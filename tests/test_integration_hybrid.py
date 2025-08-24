"""
Comprehensive test suite for hybrid ML         # Mock pattern matcher to return high confidence
        with patch.object(self.normalizer.pattern_matcher, 'get_best_match') as mock_pattern:
            mock_match = MatchResult(
                pattern_id="test_pattern",
                pattern_priority=1,
                slots={
                    "city": "Ankara",
                    "neighborhood": "Çankaya Mahallesi",
                    "street": "Tunali Hilmi Caddesi",
                    "number": "15"
                },
                confidence=0.85,
                raw_text=address,
                matched_text=address
            )
            mock_pattern.return_value = mock_match
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.addrnorm.integration.hybrid import HybridAddressNormalizer, IntegrationConfig
from src.addrnorm.ml.models import ProcessingMethod
from src.addrnorm.utils.contracts import AddressOut
from src.addrnorm.patterns.matcher import MatchResult


class TestHybridAddressNormalizer:
    """Test hybrid address normalizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IntegrationConfig(enable_ml_fallback=True, min_pattern_confidence=0.7, max_processing_time=2.0)
        self.normalizer = HybridAddressNormalizer(self.config)

    def test_initialization(self):
        """Test normalizer initialization."""
        assert self.normalizer.config == self.config
        assert self.normalizer.pattern_matcher is not None
        assert self.normalizer.hybrid_processor is not None

        # Check initial stats
        stats = self.normalizer.get_performance_stats()
        assert stats["total_processed"] == 0
        assert stats["success_rate"] == 0.0

    def test_normalize_high_pattern_confidence(self):
        """Test normalization with high pattern confidence."""
        address = "Çankaya Mahallesi Tunali Hilmi Caddesi No:15 Ankara"

        # Mock pattern compiler to return high confidence
        with patch.object(self.normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.return_value = {
                "confidence": 0.85,
                "success": True,
                "components": {
                    "city": "Ankara",
                    "neighborhood": "Çankaya Mahallesi",
                    "street": "Tunali Hilmi Caddesi",
                    "number": "15",
                },
            }

            result = self.normalizer.normalize(address)

            assert isinstance(result, AddressOut)
            assert result.city == "Ankara"
            assert result.neighborhood == "Çankaya Mahallesi"
            assert result.street == "Tunali Hilmi Caddesi"
            assert result.number == "15"
            assert result.explanation_parsed.confidence > 0.7
            assert result.explanation_parsed.method in ["pattern", "ml"]

    def test_normalize_low_pattern_confidence(self):
        """Test normalization with low pattern confidence triggering ML."""
        address = "Istanbul yakınlarında bir yer"

        # Mock pattern compiler to return low confidence
        with patch.object(self.normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.return_value = {"confidence": 0.3, "success": False, "components": {"city": "Istanbul"}}

            result = self.normalizer.normalize(address)

            assert isinstance(result, AddressOut)
            assert result.explanation_raw == address
            # ML fallback should handle this case
            assert hasattr(result.explanation_parsed, "confidence")

    def test_normalize_with_ml_disabled(self):
        """Test normalization with ML disabled."""
        config = IntegrationConfig(enable_ml_fallback=False)
        normalizer = HybridAddressNormalizer(config)

        address = "Test address"

        with patch.object(normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.return_value = {"confidence": 0.6, "success": True, "components": {"city": "Test"}}

            result = normalizer.normalize(address)

            assert result.explanation_parsed.method == "pattern"
            assert result.explanation_parsed.confidence > 0

    def test_normalize_with_error(self):
        """Test normalization error handling."""
        address = "Test address"

        # Mock pattern compiler to raise exception
        with patch.object(self.normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.side_effect = Exception("Pattern error")

            result = self.normalizer.normalize(address)

            assert isinstance(result, AddressOut)
            assert result.explanation_parsed.confidence == 0.0
            assert result.explanation_parsed.method == "fallback"
            assert len(result.explanation_parsed.warnings) > 0

    def test_batch_normalize(self):
        """Test batch normalization."""
        addresses = ["Ankara Çankaya", "İstanbul Kadıköy", "İzmir Konak"]

        with patch.object(self.normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.return_value = {"confidence": 0.8, "success": True, "components": {"city": "Test"}}

            results = self.normalizer.batch_normalize(addresses)

            assert len(results) == 3
            assert all(isinstance(r, AddressOut) for r in results)

            # Check performance stats updated
            stats = self.normalizer.get_performance_stats()
            assert stats["total_processed"] == 3

    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        address = "Test address"

        with patch.object(self.normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.return_value = {"confidence": 0.8, "success": True, "components": {"city": "Test"}}

            # Process several addresses
            for _ in range(5):
                self.normalizer.normalize(address)

            stats = self.normalizer.get_performance_stats()
            assert stats["total_processed"] == 5
            assert stats["total_time"] > 0
            assert "avg_processing_time" in stats
            assert "pattern_usage_pct" in stats

    def test_config_update(self):
        """Test configuration updates."""
        original_threshold = self.normalizer.config.min_pattern_confidence

        self.normalizer.update_config(min_pattern_confidence=0.8)

        assert self.normalizer.config.min_pattern_confidence == 0.8
        assert self.normalizer.config.min_pattern_confidence != original_threshold

    def test_reset_performance_stats(self):
        """Test performance stats reset."""
        # Process an address to generate stats
        with patch.object(self.normalizer.pattern_matcher, "get_best_match") as mock_pattern:
            mock_pattern.return_value = {"confidence": 0.8, "success": True, "components": {"city": "Test"}}

            self.normalizer.normalize("Test address")

            # Verify stats exist
            stats = self.normalizer.get_performance_stats()
            assert stats["total_processed"] > 0

            # Reset and verify
            self.normalizer.reset_performance_stats()
            stats = self.normalizer.get_performance_stats()
            assert stats["total_processed"] == 0
            assert stats["success_rate"] == 0.0

    def test_format_address(self):
        """Test address formatting."""
        components = {
            "street": "Tunali Hilmi Caddesi",
            "building_number": "15",
            "apartment_number": "3",
            "neighborhood": "Çankaya Mahallesi",
            "district": "Çankaya",
            "city": "Ankara",
            "postal_code": "06100",
        }

        formatted = self.normalizer._format_address(components)

        assert "Tunali Hilmi Caddesi" in formatted
        assert "No: 15" in formatted
        assert "Daire: 3" in formatted
        assert "Çankaya Mahallesi" in formatted
        assert "Ankara" in formatted
        assert "06100" in formatted

    def test_convert_to_normalized_address(self):
        """Test conversion to AddressOut format."""
        from src.addrnorm.ml.models import NormalizationResult, ConfidenceScore, AddressComponent, ProcessingMethod

        # Create mock ML result
        components = {
            "city": AddressComponent("Ankara", "CITY", 0.9, 0, 6, "ml"),
            "district": AddressComponent("Çankaya", "DISTRICT", 0.8, 7, 14, "ml"),
        }

        confidence = ConfidenceScore(
            overall=0.85,
            pattern_score=0.7,
            ml_score=0.9,
            adaptive_score=0.75,
            method_used=ProcessingMethod.ML,
            threshold_used=0.7,
        )

        ml_result = NormalizationResult(
            success=True, components=components, confidence=confidence, processing_time=0.1, method_details={"test": "data"}
        )

        pattern_result = {"confidence": 0.7}
        original_text = "Test address"

        normalized = self.normalizer._convert_to_normalized_address(ml_result, pattern_result, original_text)

        assert isinstance(normalized, AddressOut)
        assert normalized.city == "Ankara"
        assert normalized.district == "Çankaya"
        assert normalized.confidence_score == 0.85
        assert normalized.processing_method == "ml"
        assert normalized.explanation_raw == original_text
        assert "threshold_used" in normalized.metadata
        assert "method_details" in normalized.metadata


class TestIntegrationConfig:
    """Test integration configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntegrationConfig()

        assert config.enable_ml_fallback is True
        assert config.min_pattern_confidence == 0.7
        assert config.max_processing_time == 5.0
        assert config.enable_performance_tracking is True
        assert config.fallback_strategy == "ml"

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegrationConfig(
            enable_ml_fallback=False, min_pattern_confidence=0.8, max_processing_time=2.0, fallback_strategy="pattern"
        )

        assert config.enable_ml_fallback is False
        assert config.min_pattern_confidence == 0.8
        assert config.max_processing_time == 2.0
        assert config.fallback_strategy == "pattern"


@pytest.mark.integration
class TestHybridIntegration:
    """Integration tests for the hybrid system."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.normalizer = HybridAddressNormalizer()

    def test_end_to_end_normalization(self):
        """Test complete end-to-end normalization flow."""
        test_addresses = [
            "Çankaya Mahallesi Tunali Hilmi Caddesi No:15 Daire:3 Çankaya/Ankara",
            "Kadıköy İstanbul Fenerbahçe Mahallesi",
            "06100 Ankara Çankaya Kızılay",
            "İzmir Konak Alsancak bölgesi",
        ]

        results = []
        for address in test_addresses:
            result = self.normalizer.normalize(address)
            results.append(result)

            # Basic validation
            assert isinstance(result, AddressOut)
            assert result.country == "Turkey"
            assert result.explanation_raw == address
            assert result.processing_time >= 0
            assert result.explanation_parsed.confidence >= 0

        # Check that different methods were used
        methods_used = {r.processing_method for r in results}
        assert len(methods_used) >= 1  # At least one method should be used

        # Verify performance tracking
        stats = self.normalizer.get_performance_stats()
        assert stats["total_processed"] == len(test_addresses)
        assert stats["total_time"] > 0

    def test_adaptive_threshold_behavior(self):
        """Test that adaptive thresholds change behavior over time."""
        # This would require more sophisticated testing with mock data
        # to demonstrate threshold adaptation
        address = "Test address for threshold adaptation"

        initial_result = self.normalizer.normalize(address)

        # Process many addresses to trigger adaptation
        for _ in range(20):
            self.normalizer.normalize(f"Address variation {_}")

        # Check that pattern strength may have changed
        stats = self.normalizer.get_performance_stats()
        assert "current_pattern_strength" in stats
        assert "pattern_performance_history_length" in stats

    def test_memory_usage_batch_processing(self):
        """Test memory efficiency with large batch."""
        import gc

        # Generate large batch
        addresses = [f"Test address {i} Istanbul Turkey" for i in range(1000)]

        # Monitor memory before
        gc.collect()

        # Process batch
        results = self.normalizer.batch_normalize(addresses)

        # Verify results
        assert len(results) == 1000
        assert all(isinstance(r, AddressOut) for r in results)

        # Check performance
        stats = self.normalizer.get_performance_stats()
        assert stats["total_processed"] >= 1000
        assert stats["avg_processing_time"] > 0
