"""
Unit tests for the quality assessment system
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
test_root = Path(__file__).parent.parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.scoring.quality import (
    CompletenessScore,
    ConsistencyScore,
    GeographicConsistency,
    QualityMetrics,
    QualityAssessment,
)


class TestCompletenessScore:
    """Test CompletenessScore class"""

    def test_completeness_score_creation(self):
        """Test creating CompletenessScore"""
        cs = CompletenessScore(field_completeness=0.85, geographic_completeness=0.75, semantic_completeness=0.90)

        assert cs.field_completeness == 0.85
        assert cs.geographic_completeness == 0.75
        assert cs.semantic_completeness == 0.90

    def test_completeness_score_to_dict(self):
        """Test CompletenessScore to_dict method"""
        cs = CompletenessScore(0.85, 0.75, 0.90)
        result = cs.to_dict()

        expected = {"field_completeness": 0.85, "geographic_completeness": 0.75, "semantic_completeness": 0.90}
        assert result == expected


class TestConsistencyScore:
    """Test ConsistencyScore class"""

    def test_consistency_score_creation(self):
        """Test creating ConsistencyScore"""
        cs = ConsistencyScore(format_consistency=0.88, geographic_consistency=0.82, semantic_consistency=0.76)

        assert cs.format_consistency == 0.88
        assert cs.geographic_consistency == 0.82
        assert cs.semantic_consistency == 0.76

    def test_consistency_score_to_dict(self):
        """Test ConsistencyScore to_dict method"""
        cs = ConsistencyScore(0.88, 0.82, 0.76)
        result = cs.to_dict()

        expected = {"format_consistency": 0.88, "geographic_consistency": 0.82, "semantic_consistency": 0.76}
        assert result == expected


class TestGeographicConsistency:
    """Test GeographicConsistency class"""

    def test_geographic_consistency_creation(self):
        """Test creating GeographicConsistency"""
        gc = GeographicConsistency(hierarchy_valid=True, postal_code_match=False, region_consistency=True)

        assert gc.hierarchy_valid is True
        assert gc.postal_code_match is False
        assert gc.region_consistency is True

    def test_geographic_consistency_to_dict(self):
        """Test GeographicConsistency to_dict method"""
        gc = GeographicConsistency(True, False, True)
        result = gc.to_dict()

        expected = {"hierarchy_valid": True, "postal_code_match": False, "region_consistency": True}
        assert result == expected


class TestQualityMetrics:
    """Test QualityMetrics class"""

    def test_quality_metrics_creation(self):
        """Test creating QualityMetrics"""
        completeness = CompletenessScore(0.85, 0.75, 0.90)
        consistency = ConsistencyScore(0.88, 0.82, 0.76)

        qm = QualityMetrics(
            completeness=completeness, consistency=consistency, accuracy=0.87, usability=0.84, overall_score=8.5
        )

        assert qm.completeness == completeness
        assert qm.consistency == consistency
        assert qm.accuracy == 0.87
        assert qm.usability == 0.84
        assert qm.overall_score == 8.5

    def test_quality_metrics_to_dict(self):
        """Test QualityMetrics to_dict method"""
        completeness = CompletenessScore(0.85, 0.75, 0.90)
        consistency = ConsistencyScore(0.88, 0.82, 0.76)
        qm = QualityMetrics(completeness, consistency, 0.87, 0.84, 8.5)

        result = qm.to_dict()

        assert "completeness" in result
        assert "consistency" in result
        assert "accuracy" in result
        assert "usability" in result
        assert "overall_score" in result
        assert result["accuracy"] == 0.87


class TestQualityAssessment:
    """Test QualityAssessment class"""

    def setup_method(self):
        """Setup for each test method"""
        self.assessment = QualityAssessment()

    def test_assessment_initialization(self):
        """Test assessment initialization"""
        assert self.assessment is not None
        assert hasattr(self.assessment, "assess_quality")

    def test_assess_field_completeness(self):
        """Test field completeness assessment"""
        # Complete address components
        complete_components = {
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15",
            "daire_no": "3",
        }

        completeness = self.assessment._assess_field_completeness(complete_components)
        assert 0.8 <= completeness <= 1.0  # Should be high for complete address

        # Incomplete address components
        incomplete_components = {"il": "İstanbul", "ilce": "Kadıköy"}

        completeness_low = self.assessment._assess_field_completeness(incomplete_components)
        assert completeness_low < completeness  # Should be lower

    def test_assess_geographic_completeness(self):
        """Test geographic completeness assessment"""
        # Valid Turkish geographic hierarchy
        valid_components = {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"}

        geo_completeness = self.assessment._assess_geographic_completeness(valid_components)
        assert 0.0 <= geo_completeness <= 1.0

        # Invalid or incomplete hierarchy
        invalid_components = {"il": "InvalidCity", "ilce": "InvalidDistrict"}

        geo_completeness_low = self.assessment._assess_geographic_completeness(invalid_components)
        assert geo_completeness_low <= geo_completeness

    def test_assess_semantic_completeness(self):
        """Test semantic completeness assessment"""
        # Address with good semantic structure
        good_address = "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15"
        components = {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"}

        semantic_score = self.assessment._assess_semantic_completeness(good_address, components)
        assert 0.0 <= semantic_score <= 1.0

        # Poor semantic structure
        poor_address = "xyz123 invalid"
        poor_components = {}

        semantic_score_low = self.assessment._assess_semantic_completeness(poor_address, poor_components)
        assert semantic_score_low <= semantic_score

    def test_assess_format_consistency(self):
        """Test format consistency assessment"""
        # Well-formatted components
        consistent_components = {
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15",
        }

        format_score = self.assessment._assess_format_consistency(consistent_components)
        assert 0.0 <= format_score <= 1.0

        # Inconsistent formatting
        inconsistent_components = {
            "il": "ISTANBUL",  # Uppercase
            "ilce": "kadıköy",  # Lowercase
            "bina_no": "onbeş",  # Text instead of number
        }

        format_score_low = self.assessment._assess_format_consistency(inconsistent_components)
        assert format_score_low <= format_score

    def test_assess_geographic_consistency(self):
        """Test geographic consistency assessment"""
        # Geographically consistent
        consistent_components = {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"}

        geo_consistency = self.assessment._assess_geographic_consistency(consistent_components)
        assert isinstance(geo_consistency, GeographicConsistency)

        # Should validate hierarchy
        assert geo_consistency.hierarchy_valid in [True, False]
        assert geo_consistency.region_consistency in [True, False]

    def test_assess_accuracy(self):
        """Test accuracy assessment"""
        # Mock normalization result
        mock_result = {
            "success": True,
            "components": {"il": "İstanbul", "ilce": "Kadıköy"},
            "confidence": 0.85,
            "validation_passed": True,
        }

        accuracy = self.assessment._assess_accuracy(mock_result)
        assert 0.0 <= accuracy <= 1.0

        # Failed result should have lower accuracy
        failed_result = {"success": False, "confidence": 0.3, "validation_passed": False}

        accuracy_low = self.assessment._assess_accuracy(failed_result)
        assert accuracy_low <= accuracy

    def test_assess_usability(self):
        """Test usability assessment"""
        # Highly usable address
        usable_components = {
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15",
        }

        usability = self.assessment._assess_usability(usable_components)
        assert 0.0 <= usability <= 1.0

        # Low usability
        poor_components = {"il": "İstanbul"}

        usability_low = self.assessment._assess_usability(poor_components)
        assert usability_low <= usability

    @patch("addrnorm.scoring.quality.QualityAssessment._assess_field_completeness")
    @patch("addrnorm.scoring.quality.QualityAssessment._assess_geographic_completeness")
    @patch("addrnorm.scoring.quality.QualityAssessment._assess_semantic_completeness")
    def test_assess_quality_integration(self, mock_semantic, mock_geo, mock_field):
        """Test full quality assessment"""
        # Setup mocks
        mock_field.return_value = 0.85
        mock_geo.return_value = 0.75
        mock_semantic.return_value = 0.90

        # Mock input
        input_address = "İstanbul Kadıköy Moda Mahallesi"
        components = {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"}
        mock_result = {"success": True, "components": components, "confidence": 0.85}

        quality = self.assessment.assess_quality(input_address, components, mock_result)

        assert isinstance(quality, QualityMetrics)
        assert isinstance(quality.completeness, CompletenessScore)
        assert isinstance(quality.consistency, ConsistencyScore)
        assert 0.0 <= quality.accuracy <= 1.0
        assert 0.0 <= quality.usability <= 1.0
        assert 0.0 <= quality.overall_score <= 10.0

    def test_quality_assessment_performance(self, benchmark):
        """Test quality assessment performance"""
        input_address = "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15"
        components = {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda", "yol": "Bahariye Caddesi", "bina_no": "15"}
        mock_result = {"success": True, "components": components, "confidence": 0.85}

        # Benchmark the assessment
        result = benchmark(self.assessment.assess_quality, input_address, components, mock_result)

        assert isinstance(result, QualityMetrics)
        assert result.overall_score > 0.0


@pytest.mark.integration
class TestQualityIntegration:
    """Integration tests for quality system"""

    def test_quality_with_real_addresses(self, sample_addresses):
        """Test quality assessment with real address samples"""
        assessment = QualityAssessment()

        # Test with a subset of sample addresses
        test_samples = sample_addresses[:10]

        for sample in test_samples:
            # Mock a result based on the sample
            mock_result = {
                "success": sample.expected_success,
                "components": sample.expected_components,
                "confidence": sample.expected_confidence_min,
            }

            quality = assessment.assess_quality(sample.input_address, sample.expected_components, mock_result)

            # Verify quality metrics are reasonable
            assert isinstance(quality, QualityMetrics)
            assert 0.0 <= quality.overall_score <= 10.0

            # Successful samples should have better quality scores
            if sample.expected_success:
                assert quality.overall_score >= 3.0  # Minimum acceptable quality

            # Easy samples should have higher scores than hard ones
            if sample.difficulty_level == "easy":
                assert quality.overall_score >= 5.0

    def test_quality_consistency(self):
        """Test quality assessment consistency"""
        assessment = QualityAssessment()

        # Same input should produce same output
        input_address = "İstanbul Kadıköy Moda Mahallesi"
        components = {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"}
        mock_result = {"success": True, "components": components, "confidence": 0.85}

        # Assess multiple times
        quality1 = assessment.assess_quality(input_address, components, mock_result)
        quality2 = assessment.assess_quality(input_address, components, mock_result)

        # Should be identical
        assert quality1.overall_score == quality2.overall_score
        assert quality1.completeness.field_completeness == quality2.completeness.field_completeness

    def test_quality_edge_cases(self):
        """Test quality assessment edge cases"""
        assessment = QualityAssessment()

        # Empty input
        quality_empty = assessment.assess_quality("", {}, {"success": False})
        assert isinstance(quality_empty, QualityMetrics)
        assert quality_empty.overall_score <= 3.0  # Should be low

        # Invalid Turkish characters
        quality_invalid = assessment.assess_quality("xyz123 invalid", {}, {"success": False, "confidence": 0.1})
        assert isinstance(quality_invalid, QualityMetrics)
        assert quality_invalid.overall_score <= quality_empty.overall_score
