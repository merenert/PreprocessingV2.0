"""
Tests for dynamic threshold management system.

These tests verify the adaptive threshold behavior, persistence,
and integration with pattern matching.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.addrnorm.patterns.compiler import PatternCompiler
from src.addrnorm.patterns.matcher import PatternMatcher
from src.addrnorm.patterns.thresholds import PatternStats, ThresholdManager


class TestPatternStats:
    """Test PatternStats dataclass."""

    def test_pattern_stats_serialization(self):
        """Test PatternStats to/from dict conversion."""
        stats = PatternStats(ema_success=0.75, seen=10, last_updated=1234567890.0)

        # Test to_dict
        data = stats.to_dict()
        expected = {"ema_success": 0.75, "seen": 10, "last_updated": 1234567890.0}
        assert data == expected

        # Test from_dict
        restored = PatternStats.from_dict(data)
        assert restored == stats


class TestThresholdManager:
    """Test threshold manager functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create threshold manager with temporary cache."""
        return ThresholdManager(cache_dir=temp_cache_dir)

    def test_default_threshold(self, manager):
        """Test default threshold for new patterns."""
        threshold = manager.get_threshold("new_pattern")
        assert threshold == 0.72  # Default from config

    def test_threshold_persistence(self, temp_cache_dir):
        """Test that statistics persist across manager instances."""
        # Create first manager and update stats
        manager1 = ThresholdManager(cache_dir=temp_cache_dir)
        manager1.update_success("test_pattern", True)
        manager1.update_success("test_pattern", True)
        manager1.update_success("test_pattern", False)

        # Create second manager and check if stats persisted
        manager2 = ThresholdManager(cache_dir=temp_cache_dir)
        stats = manager2.get_stats("test_pattern")

        assert stats is not None
        assert stats.seen == 3
        assert 0 < stats.ema_success < 1  # Should be between 0 and 1

    def test_ema_calculation(self, manager):
        """Test EMA success rate calculation."""
        pattern_name = "test_ema"

        # First success should set EMA to 1.0
        manager.update_success(pattern_name, True)
        stats = manager.get_stats(pattern_name)
        assert stats.ema_success == 1.0

        # Add failure - EMA should decrease
        manager.update_success(pattern_name, False)
        stats = manager.get_stats(pattern_name)
        assert 0 < stats.ema_success < 1.0

        # With alpha=0.1, EMA should be: 0.1*0 + 0.9*1.0 = 0.9
        expected_ema = 0.9
        assert abs(stats.ema_success - expected_ema) < 0.001

    def test_threshold_adjustment(self, manager):
        """Test threshold adjustment based on success rate."""
        pattern_name = "adjustable_pattern"

        # Add enough samples to trigger adjustment
        for _ in range(6):  # min_samples = 5, so 6 should be enough
            manager.update_success(pattern_name, True)  # High success rate

        threshold = manager.get_threshold(pattern_name)

        # High success rate should lower threshold (more aggressive)
        assert threshold < 0.72  # Should be lower than default
        assert threshold >= 0.3  # But not below minimum

    def test_threshold_bounds(self, manager):
        """Test threshold stays within configured bounds."""
        pattern_name = "bounded_pattern"

        # Simulate many failures to push threshold up
        for _ in range(20):
            manager.update_success(pattern_name, False)

        threshold = manager.get_threshold(pattern_name)
        assert threshold <= 0.9  # Max threshold

        # Reset and simulate many successes
        manager.reset_stats(pattern_name)
        for _ in range(20):
            manager.update_success(pattern_name, True)

        threshold = manager.get_threshold(pattern_name)
        assert threshold >= 0.3  # Min threshold

    def test_insufficient_samples(self, manager):
        """Test that insufficient samples don't trigger adjustment."""
        pattern_name = "few_samples"

        # Add fewer samples than minimum required
        for _ in range(3):  # Less than min_samples = 5
            manager.update_success(pattern_name, True)

        threshold = manager.get_threshold(pattern_name)
        assert threshold == 0.72  # Should remain at default

    def test_reset_stats(self, manager):
        """Test resetting statistics."""
        pattern_name = "reset_me"

        # Add some stats
        manager.update_success(pattern_name, True)
        manager.update_success(pattern_name, False)

        assert manager.get_stats(pattern_name) is not None

        # Reset specific pattern
        manager.reset_stats(pattern_name)
        assert manager.get_stats(pattern_name) is None

        # Add stats for multiple patterns
        manager.update_success("pattern1", True)
        manager.update_success("pattern2", False)

        # Reset all
        manager.reset_stats()
        assert len(manager.get_all_stats()) == 0

    def test_corrupted_cache_handling(self, temp_cache_dir):
        """Test handling of corrupted cache files."""
        cache_file = Path(temp_cache_dir) / "pattern_stats.json"

        # Write corrupted JSON
        with open(cache_file, "w") as f:
            f.write("invalid json content {")

        # Manager should handle this gracefully
        manager = ThresholdManager(cache_dir=temp_cache_dir)
        assert len(manager.get_all_stats()) == 0

        # Should be able to save new stats
        manager.update_success("test", True)
        assert manager.get_stats("test") is not None

    def test_atomic_save(self, temp_cache_dir):
        """Test atomic saving behavior."""
        manager = ThresholdManager(cache_dir=temp_cache_dir)
        cache_file = Path(temp_cache_dir) / "pattern_stats.json"

        # Add some data
        manager.update_success("test", True)

        # Cache file should exist and be valid JSON
        assert cache_file.exists()
        with open(cache_file) as f:
            data = json.load(f)  # Should not raise exception

        assert "test" in data

    def test_thread_safety(self, manager):
        """Test basic thread safety (concurrent updates)."""
        import threading

        pattern_name = "concurrent_pattern"

        def update_worker(success_value):
            for _ in range(10):
                manager.update_success(pattern_name, success_value)
                time.sleep(0.001)  # Small delay to increase chance of conflicts

        # Run concurrent updates
        thread1 = threading.Thread(target=update_worker, args=(True,))
        thread2 = threading.Thread(target=update_worker, args=(False,))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should complete without errors and have correct count
        stats = manager.get_stats(pattern_name)
        assert stats.seen == 20
        assert 0 <= stats.ema_success <= 1

    def test_threshold_info(self, manager):
        """Test threshold info method."""
        pattern_name = "info_test"

        # Test new pattern
        info = manager.get_threshold_info(pattern_name)
        assert info["pattern_name"] == pattern_name
        assert info["current_threshold"] == 0.72
        assert info["default_threshold"] == 0.72
        assert info["is_adjusted"] is False
        assert info["seen"] == 0
        assert info["has_enough_samples"] is False

        # Add samples and test again
        for _ in range(6):
            manager.update_success(pattern_name, True)

        info = manager.get_threshold_info(pattern_name)
        assert info["seen"] == 6
        assert info["has_enough_samples"] is True
        assert info["is_adjusted"] is True  # Should be adjusted due to high success


class TestThresholdIntegration:
    """Test integration with pattern matcher."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def compiler(self):
        """Create pattern compiler."""
        return PatternCompiler("data/patterns/tr.yml")

    @pytest.fixture
    def threshold_manager(self, temp_cache_dir):
        """Create threshold manager."""
        return ThresholdManager(cache_dir=temp_cache_dir)

    @pytest.fixture
    def matcher(self, compiler, threshold_manager):
        """Create pattern matcher with threshold manager."""
        return PatternMatcher(compiler=compiler, threshold_manager=threshold_manager)

    def test_matcher_threshold_integration(self, matcher):
        """Test that matcher uses dynamic thresholds."""
        text = "moda mahalle bahariye sokak no 12"

        # Get initial matches
        initial_matches = matcher.match_text(text)
        assert len(initial_matches) > 0

        best_pattern = initial_matches[0].pattern_id

        # Simulate failures for this pattern to raise its threshold
        for _ in range(10):
            matcher.provide_feedback(best_pattern, False)

        # Get matches again - should have fewer due to higher threshold
        matcher.match_text(text)

        # Verify threshold actually changed
        threshold_info = matcher.get_threshold_info(best_pattern)
        assert threshold_info["is_adjusted"] is True
        assert threshold_info["current_threshold"] > threshold_info["default_threshold"]

    def test_feedback_mechanism(self, matcher):
        """Test feedback mechanism updates thresholds."""
        pattern_id = "mahalle_sokak_no"

        # Get initial threshold
        initial_threshold = matcher.get_current_threshold(pattern_id)

        # Provide positive feedback
        for _ in range(6):  # More than min_samples
            matcher.provide_feedback(pattern_id, True)

        # Threshold should decrease (more aggressive)
        new_threshold = matcher.get_current_threshold(pattern_id)
        assert new_threshold < initial_threshold

    def test_deterministic_behavior(self, matcher):
        """Test that matching behavior is deterministic."""
        text = "kadıköy mahalle atatürk sokak no 45"

        # Run multiple times - should get same results
        results = []
        for _ in range(5):
            matches = matcher.match_text(text)
            if matches:
                results.append((matches[0].pattern_id, matches[0].confidence))

        # All results should be identical
        assert len(set(results)) == 1  # All results are the same

    def test_threshold_persistence_across_matchers(self, temp_cache_dir, compiler):
        """Test that threshold adjustments persist across matcher instances."""
        # Create first matcher and adjust thresholds
        threshold_manager1 = ThresholdManager(cache_dir=temp_cache_dir)
        matcher1 = PatternMatcher(
            compiler=compiler, threshold_manager=threshold_manager1
        )

        pattern_id = "mahalle_sokak_no"
        for _ in range(10):
            matcher1.provide_feedback(pattern_id, False)  # Negative feedback

        threshold1 = matcher1.get_current_threshold(pattern_id)

        # Create second matcher - should use same thresholds
        threshold_manager2 = ThresholdManager(cache_dir=temp_cache_dir)
        matcher2 = PatternMatcher(
            compiler=compiler, threshold_manager=threshold_manager2
        )

        threshold2 = matcher2.get_current_threshold(pattern_id)

        # Thresholds should be the same
        assert abs(threshold1 - threshold2) < 0.001


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        manager = ThresholdManager()

        # Empty pattern name should not crash
        threshold = manager.get_threshold("")
        assert threshold == 0.72  # Default threshold

        manager.update_success("", True)  # Should not crash
        stats = manager.get_stats("")
        assert stats is not None

    def test_extreme_success_rates(self):
        """Test extreme success rates (all success/all failure)."""
        manager = ThresholdManager()
        pattern_name = "extreme_test"

        # All successes
        for _ in range(20):
            manager.update_success(pattern_name, True)

        threshold = manager.get_threshold(pattern_name)
        assert 0.3 <= threshold <= 0.9  # Should be within bounds

        # Reset and try all failures
        manager.reset_stats(pattern_name)
        for _ in range(20):
            manager.update_success(pattern_name, False)

        threshold = manager.get_threshold(pattern_name)
        assert 0.3 <= threshold <= 0.9  # Should be within bounds

    @patch("src.addrnorm.patterns.thresholds.get_config")
    def test_missing_config(self, mock_get_config):
        """Test handling of missing configuration."""
        # Mock empty config
        mock_get_config.return_value = {}

        manager = ThresholdManager()

        # Should use defaults
        assert manager.default_threshold == 0.72
        assert manager.min_threshold == 0.3
        assert manager.max_threshold == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
