"""
Dynamic threshold management for pattern matching.

This module implements an adaptive threshold system that adjusts pattern
acceptance thresholds based on their historical success rates using
Exponential Moving Average (EMA).
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.config import get_config


@dataclass
class PatternStats:
    """Statistics for a single pattern."""

    ema_success: float  # Exponential moving average of success rate
    seen: int  # Total number of times this pattern was used
    last_updated: float  # Timestamp of last update

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternStats":
        """Create from dictionary."""
        return cls(**data)


class ThresholdManager:
    """
    Manages dynamic thresholds for pattern matching.

    Uses Exponential Moving Average (EMA) to track pattern success rates
    and adjusts acceptance thresholds accordingly:
    - High success rate → Lower threshold (more aggressive)
    - Low success rate → Higher threshold (more conservative)
    """

    def __init__(self, cache_dir: str = ".cache", config_key: str = "patterns"):
        """
        Initialize threshold manager.

        Args:
            cache_dir: Directory for cache files
            config_key: Configuration key for pattern settings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "pattern_stats.json"
        self.config_key = config_key
        self._lock = threading.Lock()

        # Load configuration
        config = get_config()
        pattern_config = config.get(config_key, {})

        self.default_threshold = pattern_config.get("default_threshold", 0.72)
        self.ema_alpha = pattern_config.get("ema_alpha", 0.1)  # EMA smoothing factor
        self.threshold_adjustment_factor = pattern_config.get(
            "threshold_adjustment_factor", 0.2
        )
        self.min_threshold = pattern_config.get("min_threshold", 0.3)
        self.max_threshold = pattern_config.get("max_threshold", 0.9)
        self.min_samples = pattern_config.get("min_samples_for_adjustment", 5)

        # Initialize cache directory
        self.cache_dir.mkdir(exist_ok=True)

        # Load existing stats
        self.stats: Dict[str, PatternStats] = self._load_stats()

    def _load_stats(self) -> Dict[str, PatternStats]:
        """Load pattern statistics from cache file."""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            stats = {}
            for pattern_name, stat_data in data.items():
                stats[pattern_name] = PatternStats.from_dict(stat_data)

            return stats
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If cache is corrupted, start fresh
            print(f"Warning: Could not load pattern stats cache: {e}")
            return {}

    def _save_stats(self) -> None:
        """Save pattern statistics to cache file atomically."""
        # Write to temporary file first for atomic operation
        temp_file = self.cache_file.with_suffix(".tmp")

        try:
            data = {}
            for pattern_name, stats in self.stats.items():
                data[pattern_name] = stats.to_dict()

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic move (works on both Windows and Unix)
            if os.name == "nt":  # Windows
                if self.cache_file.exists():
                    self.cache_file.unlink()
                temp_file.replace(self.cache_file)
            else:  # Unix-like
                temp_file.replace(self.cache_file)

        except Exception as e:
            # Clean up temp file if something went wrong
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def get_threshold(self, pattern_name: str) -> float:
        """
        Get current threshold for a pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Current threshold value
        """
        with self._lock:
            if pattern_name not in self.stats:
                return self.default_threshold

            stats = self.stats[pattern_name]

            # Need minimum samples before adjusting threshold
            if stats.seen < self.min_samples:
                return self.default_threshold

            # Calculate adjusted threshold based on EMA success rate
            # High success rate → lower threshold (more aggressive)
            # Low success rate → higher threshold (more conservative)

            success_rate = stats.ema_success

            # Adjustment formula:
            # threshold = default - (success_rate - 0.5) * adjustment_factor
            # This means:
            # - If success_rate = 0.8, threshold decreases (more aggressive)
            # - If success_rate = 0.2, threshold increases (more conservative)

            adjustment = (success_rate - 0.5) * self.threshold_adjustment_factor
            adjusted_threshold = self.default_threshold - adjustment

            # Clamp to valid range
            return max(self.min_threshold, min(self.max_threshold, adjusted_threshold))

    def update_success(self, pattern_name: str, was_successful: bool) -> None:
        """
        Update pattern success statistics.

        Args:
            pattern_name: Name of the pattern
            was_successful: Whether the pattern matching was successful
        """
        with self._lock:
            current_time = time.time()

            if pattern_name not in self.stats:
                # Initialize with neutral success rate
                initial_success = 1.0 if was_successful else 0.0
                self.stats[pattern_name] = PatternStats(
                    ema_success=initial_success, seen=1, last_updated=current_time
                )
            else:
                stats = self.stats[pattern_name]

                # Update EMA success rate
                success_value = 1.0 if was_successful else 0.0
                stats.ema_success = (
                    self.ema_alpha * success_value
                    + (1 - self.ema_alpha) * stats.ema_success
                )

                stats.seen += 1
                stats.last_updated = current_time

            # Save to disk
            try:
                self._save_stats()
            except Exception as e:
                print(f"Warning: Could not save pattern stats: {e}")

    def get_stats(self, pattern_name: str) -> Optional[PatternStats]:
        """
        Get statistics for a pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Pattern statistics or None if not found
        """
        with self._lock:
            return self.stats.get(pattern_name)

    def get_all_stats(self) -> Dict[str, PatternStats]:
        """Get all pattern statistics."""
        with self._lock:
            return self.stats.copy()

    def reset_stats(self, pattern_name: Optional[str] = None) -> None:
        """
        Reset statistics for a pattern or all patterns.

        Args:
            pattern_name: Pattern to reset, or None to reset all
        """
        with self._lock:
            if pattern_name is None:
                self.stats.clear()
            else:
                self.stats.pop(pattern_name, None)

            try:
                self._save_stats()
            except Exception as e:
                print(f"Warning: Could not save pattern stats after reset: {e}")

    def get_threshold_info(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get detailed threshold information for debugging.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Dictionary with threshold details
        """
        with self._lock:
            threshold = self.get_threshold(pattern_name)
            stats = self.stats.get(pattern_name)

            info = {
                "pattern_name": pattern_name,
                "current_threshold": threshold,
                "default_threshold": self.default_threshold,
                "is_adjusted": threshold != self.default_threshold,
                "adjustment_factor": self.threshold_adjustment_factor,
            }

            if stats:
                info.update(
                    {
                        "ema_success": stats.ema_success,
                        "seen": stats.seen,
                        "last_updated": stats.last_updated,
                        "has_enough_samples": stats.seen >= self.min_samples,
                    }
                )
            else:
                info.update(
                    {
                        "ema_success": None,
                        "seen": 0,
                        "last_updated": None,
                        "has_enough_samples": False,
                    }
                )

            return info
