"""
Unit tests for the enhanced output formatter
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
test_root = Path(__file__).parent.parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.output.enhanced_formatter import (
    EnhancedFormatter,
    OutputFormat,
    ExplanationDetails,
    EnhancedNormalizationResult,
    LandmarkInfo,
)
from addrnorm.scoring.confidence import ConfidenceScores, PatternConfidence, MLConfidence
from addrnorm.scoring.quality import QualityMetrics, CompletenessScore, ConsistencyScore


class TestOutputFormat:
    """Test OutputFormat enum"""

    def test_output_format_values(self):
        """Test OutputFormat enum values"""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.CSV.value == "csv"
        assert OutputFormat.XML.value == "xml"
        assert OutputFormat.YAML.value == "yaml"


class TestExplanationDetails:
    """Test ExplanationDetails class"""

    def test_explanation_details_creation(self):
        """Test creating ExplanationDetails"""
        details = ExplanationDetails(
            method_used="pattern_primary",
            pattern_id="pattern_123",
            confidence_factors=["high_match", "valid_hierarchy"],
            processing_notes=["Applied Turkish address patterns"],
        )

        assert details.method_used == "pattern_primary"
        assert details.pattern_id == "pattern_123"
        assert "high_match" in details.confidence_factors
        assert "Applied Turkish address patterns" in details.processing_notes

    def test_explanation_details_to_dict(self):
        """Test ExplanationDetails to_dict method"""
        details = ExplanationDetails(
            method_used="ml_primary", confidence_factors=["model_confidence"], processing_notes=["Used ML model"]
        )

        result = details.to_dict()

        expected_keys = ["method_used", "pattern_id", "confidence_factors", "processing_notes"]
        for key in expected_keys:
            assert key in result


class TestLandmarkInfo:
    """Test LandmarkInfo class"""

    def test_landmark_info_creation(self):
        """Test creating LandmarkInfo"""
        landmark = LandmarkInfo(name="Amorium Hotel", category="hotel", spatial_relation="karşısı", confidence=0.85)

        assert landmark.name == "Amorium Hotel"
        assert landmark.category == "hotel"
        assert landmark.spatial_relation == "karşısı"
        assert landmark.confidence == 0.85

    def test_landmark_info_to_dict(self):
        """Test LandmarkInfo to_dict method"""
        landmark = LandmarkInfo("McDonald's", "restaurant", "yanı", 0.75)
        result = landmark.to_dict()

        expected = {"name": "McDonald's", "category": "restaurant", "spatial_relation": "yanı", "confidence": 0.75}
        assert result == expected


class TestEnhancedNormalizationResult:
    """Test EnhancedNormalizationResult class"""

    def create_sample_result(self):
        """Create a sample enhanced result for testing"""
        confidence = ConfidenceScores(
            pattern_details=PatternConfidence(0.85, 0.75, 0.90, 0.80),
            ml_details=MLConfidence(0.88, 0.65, 0.77, 0.82),
            overall=0.86,
        )

        quality = QualityMetrics(
            completeness=CompletenessScore(0.85, 0.75, 0.90),
            consistency=ConsistencyScore(0.88, 0.82, 0.76),
            accuracy=0.87,
            usability=0.84,
            overall_score=8.5,
        )

        explanation = ExplanationDetails(
            method_used="hybrid",
            pattern_id="pattern_123",
            confidence_factors=["high_pattern_match", "valid_hierarchy"],
            processing_notes=["Applied Turkish patterns", "ML validation"],
        )

        return EnhancedNormalizationResult(
            original_address="İstanbul Kadıköy Moda Mahallesi",
            normalized_address="İstanbul Kadıköy Moda Mahallesi",
            success=True,
            components={"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"},
            confidence=confidence,
            quality=quality,
            explanation=explanation,
            processing_time_ms=150.5,
            timestamp=datetime.now(),
        )

    def test_enhanced_result_creation(self):
        """Test creating EnhancedNormalizationResult"""
        result = self.create_sample_result()

        assert result.original_address == "İstanbul Kadıköy Moda Mahallesi"
        assert result.success is True
        assert isinstance(result.confidence, ConfidenceScores)
        assert isinstance(result.quality, QualityMetrics)
        assert isinstance(result.explanation, ExplanationDetails)

    def test_enhanced_result_to_dict(self):
        """Test EnhancedNormalizationResult to_dict method"""
        result = self.create_sample_result()
        result_dict = result.to_dict()

        expected_keys = [
            "original_address",
            "normalized_address",
            "success",
            "components",
            "confidence",
            "quality",
            "explanation",
            "processing_time_ms",
            "timestamp",
        ]

        for key in expected_keys:
            assert key in result_dict

        # Check nested structures
        assert "overall" in result_dict["confidence"]
        assert "overall_score" in result_dict["quality"]
        assert "method_used" in result_dict["explanation"]


class TestEnhancedFormatter:
    """Test EnhancedFormatter class"""

    def setup_method(self):
        """Setup for each test method"""
        self.formatter = EnhancedFormatter()

    def test_formatter_initialization(self):
        """Test formatter initialization"""
        assert self.formatter is not None
        assert hasattr(self.formatter, "format_single")
        assert hasattr(self.formatter, "format_batch")

    def test_formatter_with_options(self):
        """Test formatter with different options"""
        formatter = EnhancedFormatter(
            include_confidence=True, include_quality=True, include_explanations=True, enable_landmark_processing=True
        )

        assert formatter.include_confidence is True
        assert formatter.include_quality is True
        assert formatter.include_explanations is True
        assert formatter.enable_landmark_processing is True

    @patch("addrnorm.output.enhanced_formatter.EnhancedFormatter._normalize_address")
    @patch("addrnorm.output.enhanced_formatter.ConfidenceCalculator")
    @patch("addrnorm.output.enhanced_formatter.QualityAssessment")
    def test_format_single_basic(self, mock_quality, mock_confidence, mock_normalize):
        """Test basic single address formatting"""
        # Setup mocks
        mock_normalize.return_value = {
            "success": True,
            "components": {"il": "İstanbul", "ilce": "Kadıköy"},
            "normalized": "İstanbul Kadıköy",
        }

        mock_conf_instance = Mock()
        mock_conf_instance.calculate_confidence.return_value = ConfidenceScores(
            PatternConfidence(0.8, 0.7, 0.9, 0.8), MLConfidence(0.9, 0.6, 0.8, 0.8), 0.85
        )
        mock_confidence.return_value = mock_conf_instance

        mock_qual_instance = Mock()
        mock_qual_instance.assess_quality.return_value = QualityMetrics(
            CompletenessScore(0.8, 0.7, 0.9), ConsistencyScore(0.8, 0.8, 0.7), 0.8, 0.8, 8.0
        )
        mock_quality.return_value = mock_qual_instance

        # Test formatting
        result = self.formatter.format_single("İstanbul Kadıköy")

        assert isinstance(result, EnhancedNormalizationResult)
        assert result.success is True
        assert result.original_address == "İstanbul Kadıköy"

    @patch("addrnorm.output.enhanced_formatter.EnhancedFormatter.format_single")
    def test_format_batch(self, mock_format_single):
        """Test batch formatting"""
        # Mock single formatting
        mock_result = EnhancedNormalizationResult(
            original_address="test",
            normalized_address="test",
            success=True,
            components={},
            confidence=None,
            quality=None,
            explanation=None,
        )
        mock_format_single.return_value = mock_result

        # Test batch
        addresses = ["İstanbul Kadıköy", "Ankara Çankaya"]
        results = self.formatter.format_batch(addresses)

        assert len(results) == 2
        assert all(isinstance(r, EnhancedNormalizationResult) for r in results)
        assert mock_format_single.call_count == 2

    def test_export_json(self, temp_output_dir):
        """Test JSON export"""
        # Create sample results
        sample_result = EnhancedNormalizationResult(
            original_address="İstanbul Kadıköy",
            normalized_address="İstanbul Kadıköy",
            success=True,
            components={"il": "İstanbul", "ilce": "Kadıköy"},
            confidence=None,
            quality=None,
            explanation=None,
        )

        results = [sample_result]
        output_file = temp_output_dir / "test_results.json"

        # Export
        exported_file = self.formatter.export_json(results, str(output_file))

        assert Path(exported_file).exists()

        # Verify content
        with open(exported_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "results" in data
        assert "metadata" in data
        assert len(data["results"]) == 1

    def test_export_csv(self, temp_output_dir):
        """Test CSV export"""
        # Create sample results
        sample_result = EnhancedNormalizationResult(
            original_address="İstanbul Kadıköy",
            normalized_address="İstanbul Kadıköy",
            success=True,
            components={"il": "İstanbul", "ilce": "Kadıköy"},
            confidence=None,
            quality=None,
            explanation=None,
        )

        results = [sample_result]
        output_file = temp_output_dir / "test_results.csv"

        # Export
        exported_file = self.formatter.export_csv(results, str(output_file))

        assert Path(exported_file).exists()

        # Verify content
        with open(exported_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "original_address" in content
        assert "normalized_address" in content
        assert "İstanbul Kadıköy" in content

    def test_export_xml(self, temp_output_dir):
        """Test XML export"""
        # Create sample results
        sample_result = EnhancedNormalizationResult(
            original_address="İstanbul Kadıköy",
            normalized_address="İstanbul Kadıköy",
            success=True,
            components={"il": "İstanbul", "ilce": "Kadıköy"},
            confidence=None,
            quality=None,
            explanation=None,
        )

        results = [sample_result]
        output_file = temp_output_dir / "test_results.xml"

        # Export
        exported_file = self.formatter.export_xml(results, str(output_file))

        assert Path(exported_file).exists()

        # Verify content
        with open(exported_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<results>" in content
        assert "<result>" in content
        assert "İstanbul Kadıköy" in content

    def test_get_schema(self):
        """Test schema generation"""
        schema = self.formatter.get_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema

        # Check key properties exist
        properties = schema["properties"]
        expected_props = ["original_address", "normalized_address", "success", "components"]

        for prop in expected_props:
            assert prop in properties

    def test_landmark_processing(self):
        """Test landmark processing functionality"""
        # Enable landmark processing
        formatter = EnhancedFormatter(enable_landmark_processing=True)

        # Mock landmark detection
        with patch.object(formatter, "_detect_landmarks") as mock_detect:
            mock_detect.return_value = [LandmarkInfo("Amorium Hotel", "hotel", "karşısı", 0.85)]

            # Test with landmark address
            with patch.object(formatter, "_normalize_address") as mock_normalize:
                mock_normalize.return_value = {
                    "success": True,
                    "components": {"landmark": "Amorium Hotel"},
                    "normalized": "Amorium Hotel karşısı",
                }

                result = formatter.format_single("Amorium Hotel karşısı")

                assert result.landmarks is not None
                assert len(result.landmarks) == 1
                assert result.landmarks[0].name == "Amorium Hotel"

    def test_format_performance(self, benchmark):
        """Test formatting performance"""
        formatter = EnhancedFormatter(
            include_confidence=False, include_quality=False, include_explanations=False  # Disable for speed
        )

        with patch.object(formatter, "_normalize_address") as mock_normalize:
            mock_normalize.return_value = {"success": True, "components": {"il": "İstanbul"}, "normalized": "İstanbul"}

            # Benchmark single formatting
            result = benchmark(formatter.format_single, "İstanbul Kadıköy")

            assert isinstance(result, EnhancedNormalizationResult)


@pytest.mark.integration
class TestEnhancedFormatterIntegration:
    """Integration tests for enhanced formatter"""

    def test_formatter_with_real_addresses(self, sample_addresses):
        """Test formatter with real address samples"""
        formatter = EnhancedFormatter(include_confidence=True, include_quality=True, include_explanations=True)

        # Test with a subset of samples
        test_samples = sample_addresses[:5]

        for sample in test_samples:
            with patch.object(formatter, "_normalize_address") as mock_normalize:
                mock_normalize.return_value = {
                    "success": sample.expected_success,
                    "components": sample.expected_components,
                    "normalized": sample.input_address,
                }

                result = formatter.format_single(sample.input_address)

                assert isinstance(result, EnhancedNormalizationResult)
                assert result.original_address == sample.input_address
                assert result.success == sample.expected_success

    def test_full_pipeline_integration(self, temp_output_dir):
        """Test full pipeline with export"""
        formatter = EnhancedFormatter(include_confidence=True, include_quality=True, include_explanations=True)

        addresses = ["İstanbul Kadıköy Moda Mahallesi", "Ankara Çankaya", "İzmir Konak"]

        with patch.object(formatter, "_normalize_address") as mock_normalize:
            mock_normalize.return_value = {"success": True, "components": {"il": "İstanbul"}, "normalized": "Test"}

            # Format batch
            results = formatter.format_batch(addresses)
            assert len(results) == 3

            # Export to all formats
            json_file = formatter.export_json(results, str(temp_output_dir / "results.json"))
            csv_file = formatter.export_csv(results, str(temp_output_dir / "results.csv"))
            xml_file = formatter.export_xml(results, str(temp_output_dir / "results.xml"))

            # Verify all files created
            assert Path(json_file).exists()
            assert Path(csv_file).exists()
            assert Path(xml_file).exists()
