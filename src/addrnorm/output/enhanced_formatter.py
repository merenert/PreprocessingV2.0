"""
Enhanced Output Formatter

Advanced output formatting system with multiple output formats,
confidence scoring, quality metrics, and backward compatibility.
"""

from typing import Dict, List, Optional, Union, Any, IO
from enum import Enum
from dataclasses import dataclass, asdict
import json
import csv
import logging
from datetime import datetime
from pathlib import Path

from ..scoring.confidence import ConfidenceCalculator, ConfidenceScores, ProcessingMethod
from ..scoring.quality import QualityAssessment, QualityMetrics, QualityLevel

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""

    JSON = "json"
    ENHANCED_JSON = "enhanced_json"
    LEGACY_JSON = "legacy_json"
    CSV = "csv"
    TSV = "tsv"
    XML = "xml"
    YAML = "yaml"


class ValidationStatus(Enum):
    """Validation status levels"""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    NOT_VALIDATED = "not_validated"


@dataclass
class LandmarkInfo:
    """Landmark information"""

    name: str
    category: Optional[str] = None  # Landmark category/type
    spatial_relation: Optional[str] = None  # Spatial relation (yanı, karşısı, etc.)
    confidence: float = 0.0
    coordinates: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    type: Optional[str] = None  # Backward compatibility

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.coordinates is None:
            self.coordinates = {}
        # For backward compatibility - set type from category or vice versa
        if self.category and not self.type:
            self.type = self.category
        elif self.type and not self.category:
            self.category = self.type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "category": self.category,
            "spatial_relation": self.spatial_relation,
            "confidence": self.confidence,
        }
        # Only include optional fields if they have meaningful values
        if self.coordinates:
            result["coordinates"] = self.coordinates
        if self.metadata:
            result["metadata"] = self.metadata
        if self.type and self.type != self.category:
            result["type"] = self.type
        return result


@dataclass
class ExplanationDetails:
    """Detailed explanation of address processing"""

    raw: Optional[str] = None  # Original raw input
    parsed: Optional[Dict[str, Any]] = None  # Parsed structure analysis
    processing_steps: Optional[List[str]] = None  # List of processing steps taken
    method_details: Optional[Dict[str, Any]] = None  # Details about processing method
    method_used: Optional[str] = None  # Processing method used
    confidence_breakdown: Optional[Dict[str, float]] = None  # Confidence breakdown
    pattern_matches: Optional[List[str]] = None  # Pattern matches
    pattern_id: Optional[str] = None  # Pattern identifier used
    confidence_factors: Optional[Union[Dict[str, float], List[str]]] = None  # Factors affecting confidence
    processing_notes: Optional[List[str]] = None  # Processing notes

    def __post_init__(self):
        if self.raw is None:
            self.raw = ""
        if self.parsed is None:
            self.parsed = {}
        if self.processing_steps is None:
            self.processing_steps = []
        if self.method_details is None:
            self.method_details = {}
        if self.confidence_breakdown is None:
            self.confidence_breakdown = {}
        if self.pattern_matches is None:
            self.pattern_matches = []
        if self.confidence_factors is None:
            self.confidence_factors = {}
        if self.processing_notes is None:
            self.processing_notes = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "raw": self.raw,
            "parsed": self.parsed,
            "processing_steps": self.processing_steps,
            "method_details": self.method_details,
            "method_used": self.method_used,
            "confidence_breakdown": self.confidence_breakdown,
            "pattern_matches": self.pattern_matches,
            "pattern_id": self.pattern_id,
            "confidence_factors": self.confidence_factors,
            "processing_notes": self.processing_notes,
        }


@dataclass
class EnhancedNormalizationResult:
    """Enhanced normalization result with full metadata"""

    # Core normalization data
    normalized_address: str
    extracted_fields: Dict[str, Any]

    # Explanation and processing details
    explanation: ExplanationDetails

    # Confidence and quality metrics
    confidence_scores: ConfidenceScores
    quality_metrics: QualityMetrics

    # Processing metadata
    processing_method: ProcessingMethod
    validation_status: ValidationStatus

    # Timestamps and tracking
    processing_timestamp: datetime
    processing_duration_ms: float

    # Success indicator
    success: bool = True

    # Optional fields
    original_address: Optional[str] = None
    geocoding_info: Optional[Dict[str, Any]] = None
    alternative_results: Optional[List[Dict[str, Any]]] = None
    debug_info: Optional[Dict[str, Any]] = None
    landmarks: Optional[List[LandmarkInfo]] = None
    components: Optional[Dict[str, Any]] = None  # Detailed address components

    def __init__(self, **kwargs):
        """Initialize EnhancedNormalizationResult with backward compatibility"""
        # Import required classes
        from ..scoring.confidence import ConfidenceScores
        from ..scoring.quality import QualityMetrics

        # Handle test parameter mapping
        self.original_address = kwargs.get("original_address")
        self.normalized_address = kwargs.get("normalized_address", "")
        self.success = kwargs.get("success", True)
        self.components = kwargs.get("components", {})
        self.extracted_fields = kwargs.get("extracted_fields", kwargs.get("components", {}))

        # Handle confidence scores - can be ConfidenceScores object or just passed as 'confidence'
        confidence = kwargs.get("confidence")
        if isinstance(confidence, ConfidenceScores):
            self.confidence_scores = confidence
            self.confidence = confidence  # Backward compatibility
        else:
            # Create default ConfidenceScores
            self.confidence_scores = ConfidenceScores()
            self.confidence = self.confidence_scores

        # Handle quality metrics
        quality = kwargs.get("quality")
        if isinstance(quality, QualityMetrics):
            self.quality_metrics = quality
            self.quality = quality  # Backward compatibility
        else:
            # Create default QualityMetrics
            self.quality_metrics = QualityMetrics()
            self.quality = self.quality_metrics

        # Handle explanation
        explanation = kwargs.get("explanation")
        if isinstance(explanation, ExplanationDetails):
            self.explanation = explanation
        else:
            # Create default ExplanationDetails
            self.explanation = ExplanationDetails()

        # Handle processing metadata
        self.processing_method = kwargs.get("processing_method", ProcessingMethod.HYBRID)
        self.validation_status = kwargs.get("validation_status", ValidationStatus.PASSED)

        # Handle timestamps
        self.processing_timestamp = kwargs.get("timestamp", datetime.now())
        self.processing_duration_ms = kwargs.get("processing_time_ms", 0.0)

        # Optional fields
        self.geocoding_info = kwargs.get("geocoding_info")
        self.alternative_results = kwargs.get("alternative_results")
        self.debug_info = kwargs.get("debug_info")
        self.landmarks = kwargs.get("landmarks")

    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "normalized_address": self.normalized_address,
            "extracted_fields": self.extracted_fields,
            "explanation": self.explanation.to_dict(),
            "confidence_scores": self.confidence_scores.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "processing_method": self.processing_method.value,
            "validation_status": self.validation_status.value,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "processing_duration_ms": round(self.processing_duration_ms, 2),
            "success": self.success,
            # Backward compatibility aliases
            "confidence": self.confidence_scores.to_dict(),
            "quality": self.quality_metrics.to_dict(),
            "processing_time_ms": round(self.processing_duration_ms, 2),
            "timestamp": self.processing_timestamp.isoformat(),
        }

        # Add optional fields if present
        if self.original_address:
            result["original_address"] = self.original_address
        if self.geocoding_info:
            result["geocoding_info"] = self.geocoding_info
        if self.alternative_results:
            result["alternative_results"] = self.alternative_results
        if self.components:
            result["components"] = self.components
        if include_debug and self.debug_info:
            result["debug_info"] = self.debug_info

        return result

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility"""
        return {
            "normalized_address": self.normalized_address,
            "confidence": self.confidence_scores.overall,
            "method": self.processing_method.value,
            **self.extracted_fields,
        }


class EnhancedFormatter:
    """
    Enhanced output formatter with multiple formats and advanced features

    Features:
    - Multiple output formats (JSON, CSV, XML, YAML)
    - Confidence scoring integration
    - Quality metrics
    - Backward compatibility
    - Batch processing support
    - Export/import capabilities
    """

    def __init__(
        self,
        confidence_calculator: Optional[ConfidenceCalculator] = None,
        quality_assessor: Optional[QualityAssessment] = None,
        default_format: OutputFormat = OutputFormat.ENHANCED_JSON,
        include_debug: bool = False,
        enable_legacy_compatibility: bool = True,
        include_confidence: bool = True,
        include_quality: bool = True,
        enable_landmark_processing: bool = False,
        include_explanations: bool = True,
    ):
        """
        Initialize enhanced formatter

        Args:
            confidence_calculator: Confidence calculation system
            quality_assessor: Quality assessment system
            default_format: Default output format
            include_debug: Whether to include debug information
            enable_legacy_compatibility: Enable legacy format support
            include_confidence: Whether to include confidence scores
            include_quality: Whether to include quality metrics
            enable_landmark_processing: Enable landmark processing
            include_explanations: Whether to include detailed explanations
        """
        self.confidence_calculator = confidence_calculator or ConfidenceCalculator()
        self.quality_assessor = quality_assessor or QualityAssessment()
        self.default_format = default_format
        self.include_debug = include_debug
        self.enable_legacy_compatibility = enable_legacy_compatibility
        self.include_confidence = include_confidence
        self.include_quality = include_quality
        self.enable_landmark_processing = enable_landmark_processing
        self.include_explanations = include_explanations

        self.logger = logging.getLogger(__name__)

        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "format_counts": {format.value: 0 for format in OutputFormat},
            "avg_processing_time_ms": 0.0,
            "last_reset": datetime.now(),
        }

    def format_single(
        self,
        address_or_result: Union[str, Dict[str, Any]],
        original_address: Optional[str] = None,
        processing_context: Optional[Dict[str, Any]] = None,
        output_format: Optional[OutputFormat] = None,
    ) -> Union[Dict[str, Any], EnhancedNormalizationResult]:
        """
        Format single normalization result

        Args:
            address_or_result: Either address string to normalize or existing normalization result
            original_address: Original input address (required if address_or_result is a result dict)
            processing_context: Context about processing
            output_format: Desired output format

        Returns:
            Formatted result as dictionary or EnhancedNormalizationResult
        """
        if processing_context is None:
            processing_context = {}

        # Handle different input types for backward compatibility
        if isinstance(address_or_result, str):
            # Input is address string - normalize it first
            address = address_or_result
            original_address = address  # Use the address as original

            # Normalize the address
            normalization_result = self._normalize_address(address)

            # Add landmark detection if enabled
            landmarks = None
            if hasattr(self, "enable_landmark_processing") and self.enable_landmark_processing:
                landmarks = self._detect_landmarks(address)
                processing_context["landmarks"] = landmarks

            # Create enhanced result
            enhanced_result = self._create_simple_enhanced_result(normalization_result, address, processing_context)

            return enhanced_result

        else:
            # Input is normalization result - format it
            normalization_result = address_or_result
            if original_address is None:
                raise ValueError("original_address is required when providing normalization result")

            return self.format_result(
                normalization_result=normalization_result,
                original_address=original_address,
                processing_context=processing_context,
                output_format=output_format or self.default_format,
                include_alternatives=False,
            )

    def _normalize_address(self, address: str) -> Dict[str, Any]:
        """Mock normalize address method for testing"""
        # This is a placeholder for testing
        return {
            "normalized_address": address.strip().title(),
            "extracted_fields": {"street": address.split()[0] if address.split() else "", "number": "1"},
            "success": True,
        }

    def _create_simple_enhanced_result(
        self, normalization_result: Dict[str, Any], original_address: str, processing_context: Dict[str, Any]
    ) -> EnhancedNormalizationResult:
        """Create enhanced result from normalization result"""
        from ..scoring.confidence import ConfidenceScores
        from ..scoring.quality import QualityMetrics

        # Create enhanced result with all the components
        return EnhancedNormalizationResult(
            original_address=original_address,
            normalized_address=normalization_result.get("normalized_address", ""),
            success=normalization_result.get("success", True),
            components=normalization_result.get("extracted_fields", {}),
            confidence=ConfidenceScores(),  # Default confidence
            quality=QualityMetrics(),  # Default quality
            explanation=ExplanationDetails(),  # Default explanation
            processing_time_ms=processing_context.get("processing_time", 0.0),
            timestamp=datetime.now(),
            landmarks=processing_context.get("landmarks"),  # Add landmarks support
        )

    def _detect_landmarks(self, address: str) -> List[LandmarkInfo]:
        """Detect landmarks in address (placeholder)"""
        # This is a placeholder implementation for testing
        return []

    def get_schema(self) -> Dict[str, Any]:
        """Get output schema definition"""
        return {
            "type": "object",
            "properties": {
                "normalized_address": {"type": "string"},
                "extracted_fields": {"type": "object"},
                "confidence_scores": {"type": "object"},
                "quality_metrics": {"type": "object"},
                "processing_method": {"type": "string"},
                "validation_status": {"type": "string"},
                "processing_timestamp": {"type": "string"},
                "processing_duration_ms": {"type": "number"},
                "success": {"type": "boolean"},
                "original_address": {"type": "string"},
                "components": {"type": "object"},
                "geocoding_info": {"type": "object"},
                "alternative_results": {"type": "array"},
                "debug_info": {"type": "object"},
            },
            "required": ["normalized_address", "success"],
        }

    def export_json(self, results: List[EnhancedNormalizationResult], output_file: str) -> str:
        """Export results to JSON file"""
        import json

        # Convert results to dict format
        data_list = []
        for result in results:
            if hasattr(result, "to_dict"):
                data_list.append(result.to_dict())
            else:
                data_list.append(result)

        # Create structured output with metadata
        output_data = {
            "results": data_list,
            "metadata": {"total_count": len(results), "export_timestamp": datetime.now().isoformat(), "format_version": "1.0"},
        }

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_file

    def export_csv(self, results: List[EnhancedNormalizationResult], output_file: str) -> str:
        """Export results to CSV file"""
        import csv

        if not results:
            return output_file

        # Get first result to determine columns
        first_result = results[0]
        if hasattr(first_result, "to_dict"):
            sample_dict = first_result.to_dict()
        else:
            sample_dict = first_result

        # Flatten nested dictionaries for CSV
        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        flattened_sample = flatten_dict(sample_dict)
        fieldnames = list(flattened_sample.keys())

        # Write CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                if hasattr(result, "to_dict"):
                    row_dict = result.to_dict()
                else:
                    row_dict = result

                flattened_row = flatten_dict(row_dict)
                writer.writerow(flattened_row)

        return output_file

    def export_xml(self, results: List[EnhancedNormalizationResult], output_file: str) -> str:
        """Export results to XML file"""
        import xml.etree.ElementTree as ET

        # Create root element
        root = ET.Element("address_normalization_results")

        # Add results container
        results_elem = ET.SubElement(root, "results")

        for result in results:
            if hasattr(result, "to_dict"):
                result_dict = result.to_dict()
            else:
                result_dict = result

            # Create result element
            result_elem = ET.SubElement(results_elem, "result")

            def dict_to_xml(parent, dictionary):
                for key, value in dictionary.items():
                    element = ET.SubElement(parent, key)
                    if isinstance(value, dict):
                        dict_to_xml(element, value)
                    elif isinstance(value, list):
                        for item in value:
                            item_elem = ET.SubElement(element, "item")
                            if isinstance(item, dict):
                                dict_to_xml(item_elem, item)
                            else:
                                item_elem.text = str(item)
                    else:
                        element.text = str(value)

            dict_to_xml(result_elem, result_dict)

        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        return output_file

    def format_result(
        self,
        normalization_result: Dict[str, Any],
        original_address: str,
        processing_context: Dict[str, Any],
        output_format: Optional[OutputFormat] = None,
        include_alternatives: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """
        Format normalization result with enhanced metadata

        Args:
            normalization_result: Core normalization result
            original_address: Original input address
            processing_context: Context about processing
            output_format: Desired output format
            include_alternatives: Include alternative results

        Returns:
            Formatted result in requested format
        """
        start_time = datetime.now()

        try:
            # Use default format if not specified
            format_to_use = output_format or self.default_format

            # Create enhanced result
            enhanced_result = self._create_enhanced_result(
                normalization_result, original_address, processing_context, start_time
            )

            # Add alternatives if requested
            if include_alternatives and "alternative_results" in processing_context:
                enhanced_result.alternative_results = processing_context["alternative_results"]

            # Format according to requested format
            formatted_result = self._apply_format(enhanced_result, format_to_use)

            # Update statistics
            self._update_stats(format_to_use, start_time)

            return formatted_result

        except Exception as e:
            self.logger.error(f"Error formatting result: {e}")
            # Return fallback format
            return self._create_fallback_result(normalization_result, original_address)

    def format_batch(
        self,
        results: Union[List[str], List[Dict[str, Any]]],
        output_format: Optional[OutputFormat] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Format multiple results in batch

        Args:
            results: List of address strings or normalization results with context
            output_format: Desired output format
            output_file: Optional file to write results

        Returns:
            Formatted batch results
        """
        format_to_use = output_format or self.default_format
        formatted_results = []

        for result_item in results:
            try:
                # Handle different input types
                if isinstance(result_item, str):
                    # Input is address string - use format_single
                    formatted = self.format_single(result_item)
                    formatted_results.append(formatted)
                elif isinstance(result_item, dict):
                    # Input is result dict - extract components
                    normalization_result = result_item["result"]
                    original_address = result_item["original"]
                    context = result_item.get("context", {})

                    formatted = self.format_result(normalization_result, original_address, context, format_to_use)
                    formatted_results.append(formatted)
                else:
                    # Unknown input type
                    raise ValueError(f"Unsupported input type: {type(result_item)}")

            except Exception as e:
                self.logger.error(f"Error formatting batch item: {e}")
                # Add fallback for failed items
                if isinstance(result_item, str):
                    fallback = self._create_simple_enhanced_result(
                        {"success": False, "normalized_address": "", "extracted_fields": {}}, result_item, {}
                    )
                else:
                    fallback = self._create_fallback_result(result_item.get("result", {}), result_item.get("original", ""))
                formatted_results.append(fallback)

        # Handle file output
        if output_file:
            self._write_batch_to_file(formatted_results, output_file, format_to_use)

        # Return appropriate format
        if format_to_use in [OutputFormat.CSV, OutputFormat.TSV]:
            return self._format_batch_as_table(formatted_results, format_to_use)
        else:
            return formatted_results

    def export_schema(self, output_file: Union[str, Path], format: str = "json") -> None:
        """
        Export the enhanced output schema

        Args:
            output_file: File to write schema
            format: Schema format ('json', 'yaml', 'markdown')
        """
        schema = self._generate_schema()

        output_path = Path(output_file)

        if format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
        elif format.lower() == "yaml":
            try:
                import yaml

                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.dump(schema, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                self.logger.warning("PyYAML not installed, falling back to JSON")
                self.export_schema(output_file.with_suffix(".json"), "json")
        elif format.lower() == "markdown":
            markdown_content = self._schema_to_markdown(schema)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
        else:
            raise ValueError(f"Unsupported schema format: {format}")

        self.logger.info(f"Schema exported to {output_path}")

    def create_migration_guide(self, output_file: Union[str, Path]) -> None:
        """
        Create migration guide from legacy to enhanced format

        Args:
            output_file: File to write migration guide
        """
        guide_content = self._generate_migration_guide()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(guide_content)

        self.logger.info(f"Migration guide created: {output_file}")

    def _create_enhanced_result(
        self,
        normalization_result: Dict[str, Any],
        original_address: str,
        processing_context: Dict[str, Any],
        start_time: datetime,
    ) -> EnhancedNormalizationResult:
        """Create enhanced result with all metadata"""

        # Calculate processing duration
        processing_duration = (datetime.now() - start_time).total_seconds() * 1000

        # Create explanation details
        explanation = self._create_explanation(original_address, normalization_result, processing_context)

        # Calculate confidence scores
        confidence_scores = self.confidence_calculator.calculate_confidence(normalization_result, processing_context)

        # Assess quality metrics
        quality_metrics = self.quality_assessor.assess_quality(normalization_result, original_address, processing_context)

        # Determine validation status
        validation_status = self._determine_validation_status(confidence_scores, quality_metrics)

        return EnhancedNormalizationResult(
            normalized_address=normalization_result.get("normalized_address", ""),
            extracted_fields=normalization_result.get("extracted_fields", {}),
            explanation=explanation,
            confidence_scores=confidence_scores,
            quality_metrics=quality_metrics,
            processing_method=confidence_scores.processing_method,
            validation_status=validation_status,
            processing_timestamp=start_time,
            processing_duration_ms=processing_duration,
            original_address=original_address,
            geocoding_info=processing_context.get("geocoding_info"),
            debug_info=processing_context.get("debug_info") if self.include_debug else None,
        )

    def _create_explanation(self, original: str, result: Dict[str, Any], context: Dict[str, Any]) -> ExplanationDetails:
        """Create detailed explanation of processing"""

        # Parse input analysis
        parsed_analysis = {
            "type": context.get("input_type", "standard"),
            "components_detected": context.get("detected_components", []),
            "language": context.get("language", "turkish"),
            "structure": context.get("address_structure", "unknown"),
        }

        # Add specific parsing for landmarks, intersections, etc.
        if "landmark" in original.lower() or "karşısı" in original.lower():
            landmark_match = self._extract_landmark_info(original)
            if landmark_match:
                parsed_analysis.update(landmark_match)

        # Processing steps
        processing_steps = context.get(
            "processing_steps",
            ["Input validation", "Component detection", "Pattern matching", "Field extraction", "Quality validation"],
        )

        # Method details
        method_details = {
            "primary_method": context.get("method_used", "unknown"),
            "fallback_used": context.get("fallback_used", False),
            "patterns_matched": context.get("patterns_matched", []),
            "ml_models_used": context.get("ml_models_used", []),
        }

        return ExplanationDetails(
            raw=original, parsed=parsed_analysis, processing_steps=processing_steps, method_details=method_details
        )

    def _extract_landmark_info(self, address: str) -> Optional[Dict[str, Any]]:
        """Extract landmark information from address"""

        # Look for common landmark patterns
        landmark_patterns = [
            (r"(.+?)\s+karşısı", "opposite"),
            (r"(.+?)\s+yanı", "next_to"),
            (r"(.+?)\s+arkası", "behind"),
            (r"(.+?)\s+önü", "in_front_of"),
        ]

        import re

        for pattern, relation in landmark_patterns:
            match = re.search(pattern, address.lower())
            if match:
                landmark_name = match.group(1).strip()
                return {"type": "landmark", "name": landmark_name, "relation": relation, "confidence": 0.8}

        return None

    def _determine_validation_status(self, confidence: ConfidenceScores, quality: QualityMetrics) -> ValidationStatus:
        """Determine overall validation status"""

        # High confidence and quality = passed
        if confidence.overall >= 0.8 and quality.overall_score >= 0.8:
            return ValidationStatus.PASSED

        # Medium range = warning
        elif confidence.overall >= 0.6 and quality.overall_score >= 0.6:
            return ValidationStatus.WARNING

        # Low confidence or quality = failed
        elif confidence.overall < 0.4 or quality.overall_score < 0.4:
            return ValidationStatus.FAILED

        # Everything else = warning
        else:
            return ValidationStatus.WARNING

    def _apply_format(
        self, enhanced_result: EnhancedNormalizationResult, output_format: OutputFormat
    ) -> Union[Dict[str, Any], str]:
        """Apply specific output format"""

        if output_format == OutputFormat.ENHANCED_JSON:
            return enhanced_result.to_dict(include_debug=self.include_debug)

        elif output_format == OutputFormat.LEGACY_JSON:
            if self.enable_legacy_compatibility:
                return enhanced_result.to_legacy_format()
            else:
                return enhanced_result.to_dict(include_debug=False)

        elif output_format == OutputFormat.CSV:
            return self._to_csv_row(enhanced_result)

        elif output_format == OutputFormat.TSV:
            return self._to_tsv_row(enhanced_result)

        elif output_format == OutputFormat.XML:
            return self._to_xml(enhanced_result)

        elif output_format == OutputFormat.YAML:
            return self._to_yaml(enhanced_result)

        else:
            # Default to enhanced JSON
            return enhanced_result.to_dict(include_debug=self.include_debug)

    def _to_csv_row(self, result: EnhancedNormalizationResult) -> Dict[str, Any]:
        """Convert to CSV-friendly flat structure"""
        flat_result = {
            "original_address": result.original_address or "",
            "normalized_address": result.normalized_address,
            "confidence_overall": result.confidence_scores.overall,
            "confidence_pattern": result.confidence_scores.pattern,
            "confidence_ml": result.confidence_scores.ml,
            "quality_overall": result.quality_metrics.overall_score,
            "quality_completeness": result.quality_metrics.completeness,
            "quality_consistency": result.quality_metrics.consistency,
            "processing_method": result.processing_method.value,
            "validation_status": result.validation_status.value,
            "processing_duration_ms": result.processing_duration_ms,
        }

        # Add extracted fields with prefix
        for field, value in result.extracted_fields.items():
            flat_result[f"field_{field}"] = value or ""

        return flat_result

    def _detect_landmarks(self, address: str) -> List[LandmarkInfo]:
        """Detect landmarks in address text"""
        landmarks = []

        # Simple landmark detection (can be enhanced with ML models)
        landmark_patterns = {
            "mcdonald": ("restaurant", "McDonald's"),
            "starbucks": ("cafe", "Starbucks"),
            "migros": ("market", "Migros"),
            "bim": ("market", "BİM"),
            "a101": ("market", "A101"),
            "hospital": ("hospital", "Hospital"),
            "hastane": ("hospital", "Hastane"),
            "okul": ("school", "Okul"),
            "school": ("school", "School"),
        }

        address_lower = address.lower()
        for pattern, (category, name) in landmark_patterns.items():
            if pattern in address_lower:
                # Extract spatial relation
                spatial_relation = None
                if "yanı" in address_lower or "yanında" in address_lower:
                    spatial_relation = "yanı"
                elif "karşı" in address_lower or "karşısı" in address_lower:
                    spatial_relation = "karşısı"
                elif "arkası" in address_lower or "arkasında" in address_lower:
                    spatial_relation = "arkası"

                landmark = LandmarkInfo(
                    name=name, category=category, spatial_relation=spatial_relation, confidence=0.8  # Default confidence
                )
                landmarks.append(landmark)

        return landmarks

    def _to_tsv_row(self, result: EnhancedNormalizationResult) -> Dict[str, Any]:
        """Convert to TSV-friendly flat structure (same as CSV)"""
        return self._to_csv_row(result)

    def _to_xml(self, result: EnhancedNormalizationResult) -> str:
        """Convert to XML format"""

        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append("<address_normalization>")

        # Core data
        xml_parts.append(f"  <normalized_address>{self._escape_xml(result.normalized_address)}</normalized_address>")
        xml_parts.append(f'  <original_address>{self._escape_xml(result.original_address or "")}</original_address>')

        # Extracted fields
        xml_parts.append("  <extracted_fields>")
        for field, value in result.extracted_fields.items():
            xml_parts.append(f'    <{field}>{self._escape_xml(str(value or ""))}</{field}>')
        xml_parts.append("  </extracted_fields>")

        # Confidence scores
        xml_parts.append("  <confidence_scores>")
        xml_parts.append(f"    <overall>{result.confidence_scores.overall}</overall>")
        xml_parts.append(f"    <pattern>{result.confidence_scores.pattern}</pattern>")
        xml_parts.append(f"    <ml>{result.confidence_scores.ml}</ml>")
        xml_parts.append("  </confidence_scores>")

        # Quality metrics
        xml_parts.append("  <quality_metrics>")
        xml_parts.append(f"    <overall_score>{result.quality_metrics.overall_score}</overall_score>")
        xml_parts.append(f"    <completeness>{result.quality_metrics.completeness}</completeness>")
        xml_parts.append(f"    <consistency>{result.quality_metrics.consistency}</consistency>")
        xml_parts.append("  </quality_metrics>")

        # Metadata
        xml_parts.append(f"  <processing_method>{result.processing_method.value}</processing_method>")
        xml_parts.append(f"  <validation_status>{result.validation_status.value}</validation_status>")
        xml_parts.append(f"  <processing_duration_ms>{result.processing_duration_ms}</processing_duration_ms>")

        xml_parts.append("</address_normalization>")

        return "\n".join(xml_parts)

    def _to_yaml(self, result: EnhancedNormalizationResult) -> str:
        """Convert to YAML format"""
        try:
            import yaml

            result_dict = result.to_dict(include_debug=self.include_debug)
            return yaml.dump(result_dict, default_flow_style=False, allow_unicode=True)
        except ImportError:
            self.logger.warning("PyYAML not installed, falling back to JSON")
            return json.dumps(result.to_dict(include_debug=self.include_debug), indent=2, ensure_ascii=False)

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        if not text:
            return ""

        return (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;")
        )

    def _format_batch_as_table(self, results: List[Dict[str, Any]], format_type: OutputFormat) -> str:
        """Format batch results as table (CSV/TSV)"""
        if not results:
            return ""

        delimiter = "\t" if format_type == OutputFormat.TSV else ","

        # Get all possible field names
        all_fields = set()
        for result in results:
            all_fields.update(result.keys())

        # Create header
        header = delimiter.join(sorted(all_fields))

        # Create rows
        rows = [header]
        for result in results:
            row_values = []
            for field in sorted(all_fields):
                value = result.get(field, "")
                # Escape delimiter and quotes for CSV
                if delimiter in str(value) or '"' in str(value):
                    escaped_value = str(value).replace('"', '""')
                    value = f'"{escaped_value}"'
                row_values.append(str(value))
            rows.append(delimiter.join(row_values))

        return "\n".join(rows)

    def _write_batch_to_file(
        self, results: List[Dict[str, Any]], output_file: Union[str, Path], format_type: OutputFormat
    ) -> None:
        """Write batch results to file"""
        output_path = Path(output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            if format_type == OutputFormat.ENHANCED_JSON:
                json.dump(results, f, indent=2, ensure_ascii=False)
            elif format_type in [OutputFormat.CSV, OutputFormat.TSV]:
                table_content = self._format_batch_as_table(results, format_type)
                f.write(table_content)
            elif format_type == OutputFormat.XML:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write("<address_normalizations>\n")
                for result in results:
                    if isinstance(result, str):
                        # Already XML formatted
                        f.write(result + "\n")
                    else:
                        # Convert dict to XML (simplified)
                        f.write("  <item>\n")
                        for key, value in result.items():
                            f.write(f"    <{key}>{self._escape_xml(str(value))}</{key}>\n")
                        f.write("  </item>\n")
                f.write("</address_normalizations>\n")
            else:
                # Default JSON
                json.dump(results, f, indent=2, ensure_ascii=False)

    def _create_fallback_result(self, result: Dict[str, Any], original: str) -> Dict[str, Any]:
        """Create fallback result when formatting fails"""
        return {
            "normalized_address": result.get("normalized_address", original),
            "original_address": original,
            "confidence_scores": {"overall": 0.3, "pattern": 0.3, "ml": 0.3},
            "quality_metrics": {"overall_score": 0.3},
            "processing_method": "fallback",
            "validation_status": "failed",
            "error": "Formatting failed, using fallback result",
        }

    def _update_stats(self, format_type: OutputFormat, start_time: datetime) -> None:
        """Update processing statistics"""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        self.stats["total_processed"] += 1
        self.stats["format_counts"][format_type.value] += 1

        # Update average processing time
        total = self.stats["total_processed"]
        current_avg = self.stats["avg_processing_time_ms"]
        self.stats["avg_processing_time_ms"] = ((current_avg * (total - 1)) + duration_ms) / total

    def _generate_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for enhanced output format"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Enhanced Address Normalization Result",
            "type": "object",
            "properties": {
                "normalized_address": {"type": "string", "description": "Final normalized address string"},
                "extracted_fields": {
                    "type": "object",
                    "description": "Extracted address components",
                    "properties": {
                        "il": {"type": "string"},
                        "ilce": {"type": "string"},
                        "mahalle": {"type": "string"},
                        "sokak": {"type": "string"},
                        "bina_no": {"type": "string"},
                        "daire_no": {"type": "string"},
                        "posta_kodu": {"type": "string"},
                    },
                },
                "explanation": {
                    "type": "object",
                    "properties": {
                        "raw": {"type": "string"},
                        "parsed": {"type": "object"},
                        "processing_steps": {"type": "array", "items": {"type": "string"}},
                        "method_details": {"type": "object"},
                    },
                },
                "confidence_scores": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "number", "minimum": 0, "maximum": 1},
                        "ml": {"type": "number", "minimum": 0, "maximum": 1},
                        "overall": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "quality_metrics": {
                    "type": "object",
                    "properties": {
                        "completeness": {"type": "number", "minimum": 0, "maximum": 1},
                        "consistency": {"type": "number", "minimum": 0, "maximum": 1},
                        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                        "usability": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "processing_method": {
                    "type": "string",
                    "enum": ["pattern_primary", "pattern_secondary", "ml_primary", "ml_secondary", "hybrid", "fallback"],
                },
                "validation_status": {"type": "string", "enum": ["passed", "warning", "failed", "not_validated"]},
            },
            "required": ["normalized_address", "confidence_scores", "quality_metrics"],
        }

    def _schema_to_markdown(self, schema: Dict[str, Any]) -> str:
        """Convert schema to markdown documentation"""

        md_lines = [
            "# Enhanced Address Normalization Output Schema",
            "",
            "## Overview",
            "This document describes the enhanced output format for address normalization results.",
            "",
            "## Core Fields",
            "",
            "| Field | Type | Required | Description |",
            "|-------|------|----------|-------------|",
            "| `normalized_address` | string | Yes | Final normalized address string |",
            "| `extracted_fields` | object | No | Extracted address components |",
            "| `explanation` | object | No | Processing explanation details |",
            "| `confidence_scores` | object | Yes | Multi-level confidence scores |",
            "| `quality_metrics` | object | Yes | Quality assessment metrics |",
            "| `processing_method` | string | No | Processing method used |",
            "| `validation_status` | string | No | Validation status |",
            "",
            "## Example Output",
            "",
            "```json",
            json.dumps(
                {
                    "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
                    "explanation": {
                        "raw": "Amorium Hotel karşısı",
                        "parsed": {"type": "landmark", "name": "Amorium Hotel", "relation": "karşısı"},
                    },
                    "confidence_scores": {"pattern": 0.95, "ml": 0.87, "overall": 0.91},
                    "quality_metrics": {"completeness": 0.9, "consistency": 0.85},
                    "processing_method": "pattern_primary",
                    "validation_status": "passed",
                },
                indent=2,
            ),
            "```",
        ]

        return "\n".join(md_lines)

    def _generate_migration_guide(self) -> str:
        """Generate migration guide content"""

        return """# Migration Guide: Legacy to Enhanced Output Format

## Overview
This guide helps you migrate from the legacy address normalization output format to the new enhanced format.

## Key Changes

### Legacy Format
```json
{
  "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
  "confidence": 0.85,
  "method": "pattern",
  "il": "İstanbul",
  "ilce": "Kadıköy",
  "mahalle": "Moda",
  "sokak": "Bahariye Caddesi",
  "bina_no": "15"
}
```

### Enhanced Format
```json
{
  "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
  "extracted_fields": {
    "il": "İstanbul",
    "ilce": "Kadıköy",
    "mahalle": "Moda",
    "sokak": "Bahariye Caddesi",
    "bina_no": "15"
  },
  "explanation": {
    "raw": "Original address input",
    "parsed": { "type": "standard" },
    "processing_steps": ["validation", "extraction"],
    "method_details": { "primary_method": "pattern" }
  },
  "confidence_scores": {
    "pattern": 0.95,
    "ml": 0.75,
    "overall": 0.85
  },
  "quality_metrics": {
    "completeness": 0.9,
    "consistency": 0.85,
    "accuracy": 0.88,
    "usability": 0.82
  },
  "processing_method": "pattern_primary",
  "validation_status": "passed"
}
```

## Migration Steps

### 1. Update Field Access
- **Before**: `result["il"]`
- **After**: `result["extracted_fields"]["il"]`

### 2. Update Confidence Handling
- **Before**: `result["confidence"]`
- **After**: `result["confidence_scores"]["overall"]`

### 3. Update Method Detection
- **Before**: `result["method"]`
- **After**: `result["processing_method"]`

### 4. Add Quality Assessment
- **New**: `result["quality_metrics"]["overall_score"]`
- **New**: `result["validation_status"]`

## Backward Compatibility

The system supports legacy format output by setting `output_format="legacy_json"`:

```python
formatter = EnhancedFormatter(enable_legacy_compatibility=True)
result = formatter.format_result(data, original, context, OutputFormat.LEGACY_JSON)
```

## Code Examples

### Before (Legacy)
```python
def process_address(address):
    result = normalize_address(address)
    return {
        'address': result['normalized_address'],
        'confidence': result['confidence'],
        'city': result.get('il'),
        'district': result.get('ilce')
    }
```

### After (Enhanced)
```python
def process_address(address):
    result = normalize_address_enhanced(address)
    return {
        'address': result['normalized_address'],
        'confidence': result['confidence_scores']['overall'],
        'quality': result['quality_metrics']['overall_score'],
        'city': result['extracted_fields'].get('il'),
        'district': result['extracted_fields'].get('ilce'),
        'validation': result['validation_status']
    }
```

## Best Practices

1. **Always check validation_status** before using results
2. **Use quality_metrics** to filter low-quality results
3. **Leverage explanation** for debugging and user feedback
4. **Monitor confidence_scores** for performance insights
5. **Use appropriate output_format** for your use case

## Support

For additional support during migration, please refer to the API documentation or contact the development team.
"""

    def get_stats(self) -> Dict[str, Any]:
        """Get formatting statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset formatting statistics"""
        self.stats = {
            "total_processed": 0,
            "format_counts": {format.value: 0 for format in OutputFormat},
            "avg_processing_time_ms": 0.0,
            "last_reset": datetime.now(),
        }
        self.logger.info("Formatting statistics reset")


# Convenience functions for common operations
def create_enhanced_formatter(**kwargs) -> EnhancedFormatter:
    """Create enhanced formatter with default settings"""
    return EnhancedFormatter(**kwargs)


def format_address_result(
    result: Dict[str, Any], original: str, context: Dict[str, Any] = None, format_type: str = "enhanced_json"
) -> Dict[str, Any]:
    """Convenience function to format a single address result"""
    formatter = create_enhanced_formatter()
    output_format = OutputFormat(format_type)

    return formatter.format_result(result, original, context or {}, output_format)
