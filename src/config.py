"""
Configuration loader for GW2 PvP Tracker.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML config file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return self._default_config()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded config from {self.config_path}")
                return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

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
