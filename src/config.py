"""
Configuration loader for GW2 PvP Tracker.
"""

import yaml
import csv
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML config file and apply CSV overrides."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                config = self._default_config()
            else:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded config from {self.config_path}")

            # Apply overrides from CSV if available
            self._apply_csv_overrides(config)
            
            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

    def _apply_csv_overrides(self, config: Dict[str, Any]):
        """Override configuration with values from data CSVs."""
        
        # Helper to load a CSV into a dict of regions
        def load_regions_from_csv(path: Path) -> Dict[str, Dict[str, int]]:
            regions = {}
            if not path.exists():
                logger.warning(f"Bounding box CSV not found: {path}")
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
            logger.info("Loaded arena type detection region")

        # 2. Load Ranked/Unranked Regions
        for arena_type in ['ranked', 'unranked']:
            csv_path = Path(f"data/{arena_type}_bounding_boxes.csv")
            regions = load_regions_from_csv(csv_path)
            
            if not regions and arena_type == 'unranked':
                # Fallback to general file if specific ones don't exist (legacy support)
                fallback = Path("data/all_bounding_boxes.csv")
                if fallback.exists():
                    logger.info(f"Using fallback CSV for {arena_type}")
                    regions = load_regions_from_csv(fallback)
            
            if not regions:
                continue
                
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

            # Update Icons (Extra - if we use them later explicitly)
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

            logger.info(f"Updated {arena_type} regions from CSV")


    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "screen": {"monitor": 0, "capture_delay": 0.5},
            "cv": {
                "template_match_threshold": 0.7,
                "scales": [0.8, 0.9, 1.0, 1.1, 1.2]
            },
            "ocr": {
                "tesseract_path": None,
                "name_psm": 7,
                "score_psm": 7
            },
            "database": {"path": "data/pvp_tracker.db"},
            "paths": {
                "templates": "templates/",
                "screenshots": "screenshots/",
                "logs": "logs/"
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-notation key.

        Example: config.get('ocr.tesseract_path')
        """
        keys = key.split('.')
        value = self.data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self.get(key)
