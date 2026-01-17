"""
Configuration loader for GW2 PvP Tracker.
"""

import yaml
import csv
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        self._yaml_data = {}  # Strictly what's in the YAML file
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML config file and apply CSV overrides."""
        try:
            # 1. Load YAML data separately for persistence tracking
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                self._yaml_data = {}
            else:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._yaml_data = yaml.safe_load(f) or {}
                    logger.info(f"Loaded config from {self.config_path}")

            # 2. Build effective config starting with defaults
            config = self._default_config()
            
            # 3. Deep merge YAML data into the effective config
            self._deep_merge(config, self._yaml_data)

            # 4. Apply overrides from CSV (affects runtime 'config' only)
            self._apply_csv_overrides(config)
            
            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

    def _deep_merge(self, target: Dict, source: Dict):
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if (key in target and isinstance(target[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def _apply_csv_overrides(self, config: Dict[str, Any]):
        """Override configuration with values from data CSVs."""
        
        # Helper to load a CSV into a dict of regions
        def load_regions_from_csv(path: Path) -> Dict[str, Dict[str, int]]:
            regions = {}
            if not path.exists():
                return regions
                
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row.get('label_name')
                        if label:
                            regions[label] = {
                                'x': int(row['bbox_x']),
                                'y': int(row['bbox_y']),
                                'width': int(row['bbox_width']),
                                'height': int(row['bbox_height'])
                            }
                return regions
            except Exception as e:
                logger.error(f"Failed to load CSV {path}: {e}")
                return regions

        # Ensure main config structure exists
        if 'roster_regions' not in config:
            config['roster_regions'] = {}
        
        # 1. Load Arena Detection Region
        arena_type_csv = Path("data/arena_type_bounding_boxes.csv")
        arena_regions = load_regions_from_csv(arena_type_csv)
        if 'Arena Type' in arena_regions:
            config['arena_type_detection'] = arena_regions['Arena Type']

        # 2. Load Ranked/Unranked Regions
        for arena_type in ['ranked', 'unranked']:
            csv_path = Path(f"data/{arena_type}_bounding_boxes.csv")
            regions = load_regions_from_csv(csv_path)
            if not regions: continue
                
            if arena_type not in config['roster_regions']:
                config['roster_regions'][arena_type] = {}
            
            arena_config = config['roster_regions'][arena_type]

            # Update Red Team Names
            if 'Red Names' in regions:
                r = regions['Red Names']
                arena_config['red_team_names'] = {
                    'x_start': r['x'],
                    'x_end': r['x'] + r['width'],
                    'y_start': r['y'],
                    'row_height': r['height'] // 5,
                    'num_players': 5
                }

            # Update Blue Team Names
            if 'Blue Names' in regions:
                r = regions['Blue Names']
                arena_config['blue_team_names'] = {
                    'x_start': r['x'],
                    'x_end': r['x'] + r['width'],
                    'y_start': r['y'],
                    'row_height': r['height'] // 5,
                    'num_players': 5
                }
            
            # Update Scores
            if 'Red Score' in regions:
                r = regions['Red Score']
                arena_config['red_score_box'] = {
                    'x': r['x'], 'y': r['y'], 'width': r['width'], 'height': r['height']
                }
            if 'Blue Score' in regions:
                r = regions['Blue Score']
                arena_config['blue_score_box'] = {
                    'x': r['x'], 'y': r['y'], 'width': r['width'], 'height': r['height']
                }

            # Update Icons
            if 'Red Icons' in regions:
                r = regions['Red Icons']
                arena_config['red_team_icons'] = {
                     'x': r['x'], 'y': r['y'], 'width': r['width'], 'height': r['height']
                }
            if 'Blue Icons' in regions:
                r = regions['Blue Icons']
                arena_config['blue_team_icons'] = {
                     'x': r['x'], 'y': r['y'], 'width': r['width'], 'height': r['height']
                }

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "screen": {"monitor": 0, "capture_delay": 0.5},
            "cv": {
                "template_match_threshold": 0.7,
                "scales": [0.8, 0.9, 1.0, 1.1, 1.2]
            },
            "ocr": {
                "engine": "easyocr",
                "tesseract_path": None,
                "name_psm": 7,
                "score_psm": 7
            },
            "analytics": {
                "win_rate_mode": "raw",
                "bayesian_confidence": 10.0,
                "bayesian_mean": 50.0
            },
            "database": {"path": "data/pvp_tracker.db"},
            "paths": {
                "templates": "templates/",
                "screenshots": "screenshots/",
                "logs": "logs/"
            }
        }

    def set_setting(self, section: str, key: str, value: Any):
        """Update a setting in both runtime and persistence layer."""
        # 1. Update runtime config (self.data)
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value

        # 2. Update persistence config (self._yaml_data)
        if section not in self._yaml_data:
            self._yaml_data[section] = {}
        
        # Determine if we need to copy to avoid reference issues
        self._yaml_data[section][key] = copy.deepcopy(value)

    def save(self):
        """Save current explicitly configured settings to YAML file."""
        try:
            # We strictly dump self._yaml_data, which contains only:
            # 1. What was originally in the file (loaded in __init__)
            # 2. What was explicitly changed via set_setting()
            # This prevents polluting config.yaml with defaults or CSV data.
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._yaml_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key."""
        keys = key.split('.')
        value = self.data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self.get(key)
