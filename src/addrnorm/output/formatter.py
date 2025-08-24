"""
Enhanced Output Formatter

Formats address normalization results with enhanced output schema including
confidence scores, quality metrics, and processing metadata.
"""

import json
from typing import Dict, Any, Optional, Union
from datetime import datetime

from ..utils.contracts import AddressOut
from ..scoring.confidence import ConfidenceCalculator
from ..scoring.quality import QualityAssessment
from .schema import EnhancedAddressOutput, NormalizedAddressData, ConfidenceScores, QualityMetrics, ProcessingMetadata


class EnhancedOutputFormatter:
    """
    Formats normalization results using enhanced output schema
    """

    def __init__(self):
        self.confidence_calculator = ConfidenceCalculator()
        self.quality_assessor = QualityAssessment()

    def format_enhanced_output(
        self,
        components: Dict[str, str],
        original_address: str,
        confidence_data: Optional[Dict[str, float]] = None,
        processing_method: str = "pattern_primary",
        processing_time_ms: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> EnhancedAddressOutput:
        """
        Format components into enhanced output schema

        Args:
            components: Extracted address components
            original_address: Original input address
            confidence_data: Confidence scores from processing
            processing_method: Method used for processing
            processing_time_ms: Processing time in milliseconds
            additional_metadata: Additional metadata to include

        Returns:
            EnhancedAddressOutput: Formatted enhanced output
        """

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(components, original_address, confidence_data, processing_method)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            components, original_address, confidence_scores.to_dict(), processing_method
        )

        # Create processing metadata
        processing_metadata = self._create_processing_metadata(processing_method, processing_time_ms, additional_metadata)

        # Create normalized address data (yer-yön açıklaması components'den oluşturulacak)
        normalized_data = NormalizedAddressData.from_components(components)

        # Create enhanced output
        return EnhancedAddressOutput(
            success=bool(components and any(components.values())),
            normalized_address=normalized_data,
            confidence_scores=confidence_scores,
            quality_metrics=quality_metrics,
            processing_metadata=processing_metadata,
            original_address=original_address,
        )

    def format_from_address_out(
        self,
        address_out: AddressOut,
        original_address: str,
        confidence_data: Optional[Dict[str, float]] = None,
        processing_method: str = "pattern_primary",
        processing_time_ms: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> EnhancedAddressOutput:
        """
        Convert AddressOut to enhanced output format

        Args:
            address_out: Standard AddressOut result
            original_address: Original input address
            confidence_data: Confidence scores from processing
            processing_method: Method used for processing
            processing_time_ms: Processing time in milliseconds
            additional_metadata: Additional metadata to include

        Returns:
            EnhancedAddressOutput: Formatted enhanced output
        """

        # Convert AddressOut to components dict
        components = {}
        if hasattr(address_out, "model_dump"):
            # Pydantic v2
            components = address_out.model_dump(exclude_none=True)
        elif hasattr(address_out, "dict"):
            # Pydantic v1
            components = address_out.dict(exclude_none=True)
        else:
            # Fallback to __dict__
            components = {k: v for k, v in address_out.__dict__.items() if v is not None}

        return self.format_enhanced_output(
            components=components,
            original_address=original_address,
            confidence_data=confidence_data,
            processing_method=processing_method,
            processing_time_ms=processing_time_ms,
            additional_metadata=additional_metadata,
        )

    def _calculate_confidence_scores(
        self,
        components: Dict[str, str],
        original_address: str,
        confidence_data: Optional[Dict[str, float]],
        processing_method: str,
    ) -> ConfidenceScores:
        """Calculate comprehensive confidence scores"""

        # Use provided confidence data or calculate
        if confidence_data:
            pattern_confidence = confidence_data.get("pattern", 0.0)
            ml_confidence = confidence_data.get("ml", 0.0)
            overall_confidence = confidence_data.get("overall", 0.0)
        else:
            # Calculate using ConfidenceCalculator
            enhanced_scores = self.confidence_calculator.calculate_enhanced_confidence(
                components, original_address, processing_method
            )
            pattern_confidence = enhanced_scores.get("pattern", 0.0)
            ml_confidence = enhanced_scores.get("ml", 0.0)
            overall_confidence = enhanced_scores.get("overall", 0.0)

        return ConfidenceScores(overall=overall_confidence, pattern=pattern_confidence, ml=ml_confidence)

    def _calculate_quality_metrics(
        self, components: Dict[str, str], original_address: str, confidence_scores: Dict[str, float], processing_method: str
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""

        quality_data = self.quality_assessor.assess_enhanced_quality(
            extracted_components=components,
            original_address=original_address,
            confidence_scores=confidence_scores,
            processing_method=processing_method,
        )

        return QualityMetrics(
            completeness=quality_data.get("completeness", 0.0),
            consistency=quality_data.get("consistency", 0.0),
            accuracy=quality_data.get("accuracy"),
            coverage=quality_data.get("coverage"),
        )

    def _create_processing_metadata(
        self, processing_method: str, processing_time_ms: Optional[float], additional_metadata: Optional[Dict[str, Any]]
    ) -> ProcessingMetadata:
        """Create processing metadata"""

        # Extract known fields for ProcessingMetadata
        warnings = None
        debug_info = {}
        pattern_id = None
        model_version = None

        if additional_metadata:
            warnings = additional_metadata.get("warnings")
            pattern_id = additional_metadata.get("pattern_id")
            model_version = additional_metadata.get("model_version")

            # Add processing method and timestamp to debug_info
            debug_info = {
                "processing_method": processing_method,
                "timestamp": datetime.now().isoformat(),
                **{k: v for k, v in additional_metadata.items() if k not in ["warnings", "pattern_id", "model_version"]},
            }
        else:
            debug_info = {"processing_method": processing_method, "timestamp": datetime.now().isoformat()}

        return ProcessingMetadata(
            processing_time_ms=processing_time_ms,
            pattern_id=pattern_id,
            model_version=model_version,
            warnings=warnings,
            debug_info=debug_info if debug_info else None,
        )

    def to_json(self, enhanced_output: EnhancedAddressOutput, indent: int = 2) -> str:
        """
        Convert enhanced output to JSON string

        Args:
            enhanced_output: Enhanced output to serialize
            indent: JSON indentation level

        Returns:
            JSON string representation
        """

        try:
            if hasattr(enhanced_output, "model_dump"):
                # Pydantic v2
                data = enhanced_output.model_dump(exclude_none=True)
            elif hasattr(enhanced_output, "dict"):
                # Pydantic v1
                data = enhanced_output.dict(exclude_none=True)
            else:
                # Fallback
                data = enhanced_output.to_dict()

            return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        except Exception as e:
            # Fallback to basic dict conversion
            return json.dumps(enhanced_output.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def to_legacy_format(self, enhanced_output: EnhancedAddressOutput) -> Dict[str, str]:
        """
        Convert enhanced output back to legacy format for backward compatibility

        Args:
            enhanced_output: Enhanced output to convert

        Returns:
            Dictionary in legacy format
        """

        return enhanced_output.to_legacy_format()

    def format_compact(self, enhanced_output: EnhancedAddressOutput) -> Dict[str, Any]:
        """
        Create compact representation for API responses

        Args:
            enhanced_output: Enhanced output to compact

        Returns:
            Compact dictionary representation
        """

        compact = {
            "success": enhanced_output.success,
            "address": enhanced_output.normalized_address.to_dict(),
            "confidence": round(enhanced_output.confidence_scores.overall, 3),
            "quality": round(enhanced_output.quality_metrics.completeness, 3),
        }

        # Add processing info if available
        if enhanced_output.processing_metadata.processing_time_ms:
            compact["processing_time_ms"] = enhanced_output.processing_metadata.processing_time_ms

        return compact

    def validate_output(self, enhanced_output: EnhancedAddressOutput) -> Dict[str, Any]:
        """
        Validate enhanced output and return validation results

        Args:
            enhanced_output: Enhanced output to validate

        Returns:
            Validation results dictionary
        """

        validation_results = {"is_valid": True, "errors": [], "warnings": []}

        try:
            # Basic structure validation
            if not enhanced_output.normalized_address:
                validation_results["errors"].append("Missing normalized_address")
                validation_results["is_valid"] = False

            if not enhanced_output.confidence_scores:
                validation_results["errors"].append("Missing confidence_scores")
                validation_results["is_valid"] = False

            if not enhanced_output.quality_metrics:
                validation_results["errors"].append("Missing quality_metrics")
                validation_results["is_valid"] = False

            # Confidence score validation
            if enhanced_output.confidence_scores:
                if not (0.0 <= enhanced_output.confidence_scores.overall <= 1.0):
                    validation_results["errors"].append("Overall confidence out of range [0,1]")
                    validation_results["is_valid"] = False

                if enhanced_output.confidence_scores.overall < 0.3:
                    validation_results["warnings"].append("Low overall confidence")

            # Quality metrics validation
            if enhanced_output.quality_metrics:
                if not (0.0 <= enhanced_output.quality_metrics.completeness <= 1.0):
                    validation_results["errors"].append("Completeness score out of range [0,1]")
                    validation_results["is_valid"] = False

                if enhanced_output.quality_metrics.completeness < 0.5:
                    validation_results["warnings"].append("Low completeness score")

            # Address data validation
            if enhanced_output.normalized_address:
                component_dict = enhanced_output.normalized_address.to_dict()
                if not any(component_dict.values()):
                    validation_results["warnings"].append("No address components extracted")

        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False

        return validation_results


# Legacy compatibility
class OutputFormatter(EnhancedOutputFormatter):
    """Legacy output formatter - redirects to enhanced formatter"""

    def format_output(self, components: Dict[str, str], **kwargs) -> Dict[str, str]:
        """Legacy format_output method"""
        enhanced = self.format_enhanced_output(components, **kwargs)
        return enhanced.to_legacy_format()
