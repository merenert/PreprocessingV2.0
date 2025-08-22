"""
Configuration management for addrnorm package.
"""

from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for address normalization."""

    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        self._config = {}
        self._load_default_config()

        if config_path:
            self.load_config(config_path)

    def _load_default_config(self):
        """Load default configuration."""
        self._config = {
            "preprocess": {
                "normalize_case": True,
                "normalize_unicode": True,
                "expand_abbreviations": True,
                "clean_punctuation": True,
                "preserve_numbers": True,
                "token_separators": [" ", ",", ";", "/", "\\", "-", "_"],
                "punctuation_mapping": {
                    ".": " ",
                    ",": " ",
                    ";": " ",
                    ":": " ",
                    "/": " ",
                    "\\": " ",
                    "-": " ",
                    "_": " ",
                    "(": " ",
                    ")": " ",
                    "[": " ",
                    "]": " ",
                    "{": " ",
                    "}": " ",
                },
            },
            "resources": {"abbreviations_file": "data/resources/abbr_tr.yaml"},
            "patterns": {
                "default_threshold": 0.72,
                "ema_alpha": 0.1,
                "threshold_adjustment_factor": 0.2,
                "min_threshold": 0.3,
                "max_threshold": 0.9,
                "min_samples_for_adjustment": 5,
            },
        }

    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                self._merge_config(user_config)
        except FileNotFoundError:
            pass  # Use default config if file doesn't exist

    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with defaults."""

        def deep_merge(default, user):
            for key, value in user.items():
                if (
                    key in default
                    and isinstance(default[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(default[key], value)
                else:
                    default[key] = value

        deep_merge(self._config, user_config)

    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated key path."""
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value):
        """Set configuration value by dot-separated key path."""
        keys = key_path.split(".")
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
