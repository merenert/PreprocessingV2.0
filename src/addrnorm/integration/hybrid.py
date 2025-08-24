"""
Integration layer for hybrid ML + pattern-based address normalization.
Combines existing pattern compiler with advanced ML models.
"""

from typing import Dict, List, Optional, Union, Tuple
import time
import logging
from dataclasses import dataclass

from ..patterns.matcher import PatternMatcher
from ..ml.models import HybridProcessor, NormalizationResult, ProcessingMethod, ConfidenceScore, AddressComponent
from ..utils.contracts import AddressOut, ExplanationParsed, MethodEnum
from ..output.formatter import EnhancedOutputFormatter
from ..output.schema import EnhancedAddressOutput


logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for hybrid integration."""

    enable_ml_fallback: bool = True
    min_pattern_confidence: float = 0.7
    max_processing_time: float = 5.0  # seconds
    enable_performance_tracking: bool = True
    fallback_strategy: str = "ml"  # "ml", "pattern", "both"
    enhanced_output: bool = False  # Enable enhanced output format


class HybridAddressNormalizer:
    """
    Integrated normalizer combining pattern matching with ML models.

    Features:
    - Adaptive threshold calculation
    - Pattern-first approach with ML fallback
    - Performance tracking and optimization
    - Confidence-based method selection
    - Enhanced output format support
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()

        # Initialize components
        self.pattern_matcher = PatternMatcher()
        self.hybrid_processor = HybridProcessor()
        self.enhanced_formatter = EnhancedOutputFormatter()

        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "pattern_used": 0,
            "ml_used": 0,
            "hybrid_used": 0,
            "total_time": 0.0,
            "success_rate": 0.0,
        }

        logger.info(f"Initialized HybridAddressNormalizer with config: {self.config}")

    def normalize(
        self, address_text: str, context: Optional[Dict] = None, enhanced_output: Optional[bool] = None
    ) -> Union[AddressOut, EnhancedAddressOutput]:
        """
        Normalize address using hybrid approach.

        Args:
            address_text: Raw address text to normalize
            context: Optional context information
            enhanced_output: Override config setting for enhanced output

        Returns:
            AddressOut or EnhancedAddressOutput based on configuration
        """
        use_enhanced = enhanced_output if enhanced_output is not None else self.config.enhanced_output
        start_time = time.time()

        try:
            # Step 1: Pattern matching
            pattern_result = self._run_pattern_matching(address_text)

            # Step 2: Hybrid processing with adaptive thresholds
            ml_result = self._run_hybrid_processing(address_text, pattern_result, context)

            # Step 3: Extract components and confidence data
            components = self._extract_components_from_result(ml_result, pattern_result)
            confidence_data = self._extract_confidence_data(ml_result)
            processing_method = self._determine_processing_method(ml_result)

            # Step 4: Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Step 5: Format output based on configuration
            if use_enhanced:
                result = self.enhanced_formatter.format_enhanced_output(
                    components=components,
                    original_address=address_text,
                    confidence_data=confidence_data,
                    processing_method=processing_method,
                    processing_time_ms=processing_time,
                    additional_metadata={
                        "context": context,
                        "pattern_matched": pattern_result.get("success", False),
                        "ml_processed": ml_result.success,
                    },
                )
            else:
                # Legacy format
                result = self._convert_to_normalized_address(ml_result, pattern_result, address_text)

            # Step 6: Update performance tracking
            self._update_performance_stats(ml_result, processing_time / 1000)

            logger.debug(
                f"Normalized address using {processing_method} "
                f"in {processing_time:.1f}ms with confidence {confidence_data.get('overall', 0):.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error normalizing address '{address_text}': {e}")
            processing_time = (time.time() - start_time) * 1000

            # Return failed result in appropriate format
            if use_enhanced:
                return self._create_failed_enhanced_result(address_text, str(e), processing_time)
            else:
                return self._create_failed_result(address_text, str(e), processing_time / 1000)

    def _run_pattern_matching(self, address_text: str) -> Dict:
        """Run pattern matching and return results."""
        try:
            # Use pattern matcher to get best match
            match_result = self.pattern_matcher.get_best_match(address_text)

            if match_result:
                # Convert to expected format
                return {
                    "confidence": match_result.confidence,
                    "components": match_result.slots,  # slots contain extracted components
                    "method": "pattern",
                    "success": True,
                    "pattern_id": match_result.pattern_id,
                    "matched_text": match_result.matched_text,
                }
            else:
                return {
                    "confidence": 0.0,
                    "components": {},
                    "method": "pattern",
                    "success": False,
                    "error": "No pattern matched",
                }

        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            return {"confidence": 0.0, "components": {}, "method": "pattern", "success": False, "error": str(e)}

    def _run_hybrid_processing(
        self, address_text: str, pattern_result: Dict, context: Optional[Dict] = None
    ) -> NormalizationResult:
        """Run hybrid processing with ML fallback."""

        if not self.config.enable_ml_fallback:
            # Pattern-only mode
            return self._pattern_only_result(pattern_result, address_text)

        try:
            # Use hybrid processor for adaptive method selection
            ml_result = self.hybrid_processor.process_hybrid(address_text, pattern_result)

            # Check processing time constraints
            if ml_result.processing_time > self.config.max_processing_time:
                logger.warning(
                    f"Processing time {ml_result.processing_time:.3f}s exceeds limit " f"{self.config.max_processing_time}s"
                )

            return ml_result

        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")

            # Fallback to pattern-only result
            return self._pattern_only_result(pattern_result, address_text)

    def _pattern_only_result(self, pattern_result: Dict, address_text: str) -> NormalizationResult:
        """Create NormalizationResult from pattern matching only."""

        # Convert pattern components to AddressComponent format
        components = {}
        for label, value in pattern_result.get("components", {}).items():
            if value:
                components[label] = AddressComponent(
                    text=str(value),
                    label=label.upper(),
                    confidence=pattern_result.get("confidence", 0.0),
                    start=0,
                    end=len(str(value)),
                    source="pattern",
                )

        confidence = ConfidenceScore(
            overall=pattern_result.get("confidence", 0.0),
            pattern_score=pattern_result.get("confidence", 0.0),
            ml_score=0.0,
            adaptive_score=0.0,
            method_used=ProcessingMethod.PATTERN,
            threshold_used=self.config.min_pattern_confidence,
        )

        return NormalizationResult(
            success=pattern_result.get("success", False),
            components=components,
            confidence=confidence,
            processing_time=0.0,
            method_details={"fallback_reason": "ml_disabled_or_failed", "pattern_only": True},
        )

    def _convert_to_normalized_address(
        self, ml_result: NormalizationResult, pattern_result: Dict, original_text: str
    ) -> AddressOut:
        """Convert ML result to standard NormalizedAddress format."""

        # Extract components with proper field mapping
        components = {}

        if ml_result.success and ml_result.components:
            # Map ML components to standard contract fields
            field_mapping = {
                "city": "city",
                "il": "city",
                "district": "district",
                "ilce": "district",
                "neighborhood": "neighborhood",
                "mahalle": "neighborhood",
                "mah": "neighborhood",
                "street": "street",
                "sokak": "street",
                "sok": "street",
                "cadde": "street",
                "cad": "street",
                "bulvar": "street",
                "number": "building_number",
                "no": "building_number",
                "apartment": "apartment_number",
                "daire": "apartment_number",
                "floor": "floor",
                "kat": "floor",
                "postal_code": "postal_code",
                "posta": "postal_code",
            }

            for label, component in ml_result.components.items():
                standard_field = field_mapping.get(label.lower(), label.lower())
                if standard_field and component.text:
                    components[standard_field] = component.text

        # Create normalized address
        normalized = AddressOut(
            country="TR",
            city=components.get("city") or None,
            district=components.get("district") or None,
            neighborhood=components.get("neighborhood") or None,
            street=components.get("street") or None,
            building=components.get("building") or None,
            number=components.get("building_number") or None,
            apartment=components.get("apartment_number") or None,
            floor=components.get("floor") or None,
            postcode=components.get("postal_code") or None,
            explanation_raw=original_text.strip(),
            explanation_parsed=ExplanationParsed(
                confidence=ml_result.confidence.overall,
                method=MethodEnum.ML if ml_result.confidence.method_used.value == "ml" else MethodEnum.PATTERN,
                warnings=ml_result.warnings or [],
            ),
            normalized_address=self._format_address(components) or "Address not normalized",
        )

        return normalized

    def _format_address(self, components: Dict[str, str]) -> str:
        """Format address components into readable string."""
        parts = []

        # Order: street, neighborhood, district, city, postal_code
        order = [
            ("street", ""),
            ("building_number", "No:"),
            ("apartment_number", "Daire:"),
            ("floor", "Kat:"),
            ("neighborhood", ""),
            ("district", ""),
            ("city", ""),
            ("postal_code", ""),
        ]

        for field, prefix in order:
            value = components.get(field, "").strip()
            if value:
                if prefix:
                    parts.append(f"{prefix} {value}")
                else:
                    parts.append(value)

        return " ".join(parts)

    def _create_failed_result(self, original_text: str, error: str, processing_time: float) -> AddressOut:
        """Create failed normalization result."""
        return AddressOut(
            country="TR",
            city=None,
            district=None,
            neighborhood=None,
            street=None,
            building=None,
            number=None,
            apartment=None,
            floor=None,
            postcode=None,
            explanation_raw=original_text,
            explanation_parsed=ExplanationParsed(confidence=0.0, method=MethodEnum.FALLBACK, warnings=[error]),
            normalized_address="Address normalization failed",
        )

    def _update_performance_stats(self, result: NormalizationResult, processing_time: float):
        """Update performance tracking statistics."""
        if not self.config.enable_performance_tracking:
            return

        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_time"] += processing_time

        # Count method usage
        method = result.confidence.method_used
        if method == ProcessingMethod.PATTERN:
            self.processing_stats["pattern_used"] += 1
        elif method == ProcessingMethod.ML:
            self.processing_stats["ml_used"] += 1
        else:
            self.processing_stats["hybrid_used"] += 1

        # Update success rate (exponential moving average)
        success = float(result.success)
        alpha = 0.1  # Smoothing factor
        current_rate = self.processing_stats["success_rate"]
        self.processing_stats["success_rate"] = alpha * success + (1 - alpha) * current_rate

        # Update hybrid processor performance tracking
        self.hybrid_processor.update_performance(result.success, method)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = self.processing_stats.copy()

        total = stats["total_processed"]
        if total > 0:
            stats["avg_processing_time"] = stats["total_time"] / total
            stats["pattern_usage_pct"] = (stats["pattern_used"] / total) * 100
            stats["ml_usage_pct"] = (stats["ml_used"] / total) * 100
            stats["hybrid_usage_pct"] = (stats["hybrid_used"] / total) * 100

        # Add hybrid processor stats
        stats["current_pattern_strength"] = self.hybrid_processor.current_pattern_strength
        stats["pattern_performance_history_length"] = len(self.hybrid_processor.pattern_performance_history)

        return stats

    def reset_performance_stats(self):
        """Reset performance tracking."""
        self.processing_stats = {
            "total_processed": 0,
            "pattern_used": 0,
            "ml_used": 0,
            "hybrid_used": 0,
            "total_time": 0.0,
            "success_rate": 0.0,
        }

        # Reset hybrid processor stats
        self.hybrid_processor.pattern_performance_history = []
        self.hybrid_processor.current_pattern_strength = 0.8

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def batch_normalize(
        self, address_list: List[str], context: Optional[Dict] = None, enhanced_output: Optional[bool] = None
    ) -> List[Union[AddressOut, EnhancedAddressOutput]]:
        """
        Normalize multiple addresses efficiently.

        Args:
            address_list: List of address texts to normalize
            context: Optional context information
            enhanced_output: Override config setting for enhanced output

        Returns:
            List of normalized addresses
        """
        results = []
        start_time = time.time()

        logger.info(f"Starting batch normalization of {len(address_list)} addresses")

        for i, address in enumerate(address_list):
            try:
                result = self.normalize(address, context, enhanced_output)
                results.append(result)

                # Log progress for large batches
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (len(address_list) - i - 1)
                    logger.info(f"Processed {i + 1}/{len(address_list)} addresses. ETA: {eta:.1f}s")

            except Exception as e:
                logger.error(f"Error processing address {i}: '{address}' - {e}")
                # Add failed result
                if enhanced_output or self.config.enhanced_output:
                    failed_result = self._create_failed_enhanced_result(address, str(e), 0)
                else:
                    failed_result = self._create_failed_result(address, str(e), 0)
                results.append(failed_result)

        total_time = time.time() - start_time
        logger.info(f"Batch normalization completed in {total_time:.2f}s")

        return results

    def _extract_components_from_result(self, ml_result: NormalizationResult, pattern_result: Dict) -> Dict[str, str]:
        """Extract components from ML and pattern results"""
        components = {}

        if ml_result.success and ml_result.components:
            # Extract from ML result
            for label, component in ml_result.components.items():
                if component and component.text:
                    # Map ML labels to standard field names
                    mapped_label = self._map_component_label(label)
                    components[mapped_label] = component.text
        elif pattern_result.get("success") and pattern_result.get("components"):
            # Fall back to pattern result
            for label, value in pattern_result["components"].items():
                if value:
                    mapped_label = self._map_component_label(label)
                    components[mapped_label] = str(value)

        return components

    def _extract_confidence_data(self, ml_result: NormalizationResult) -> Dict[str, float]:
        """Extract confidence scores from ML result"""
        if ml_result.confidence:
            return {
                "overall": ml_result.confidence.overall,
                "pattern": ml_result.confidence.pattern_score,
                "ml": ml_result.confidence.ml_score,
            }
        return {"overall": 0.0, "pattern": 0.0, "ml": 0.0}

    def _determine_processing_method(self, ml_result: NormalizationResult) -> str:
        """Determine processing method used"""
        if not ml_result.confidence:
            return "fallback"

        method_mapping = {
            ProcessingMethod.PATTERN: "pattern_primary",
            ProcessingMethod.ML: "ml_primary",
            ProcessingMethod.HYBRID: "hybrid",
        }

        return method_mapping.get(ml_result.confidence.method_used, "unknown")

    def _map_component_label(self, label: str) -> str:
        """Map component labels to standard field names"""
        label_mapping = {
            # Turkish field names
            "IL": "il",
            "ILCE": "ilce",
            "MAHALLE": "mahalle",
            "SOKAK": "sokak",
            "BINA_NO": "bina_no",
            "POSTA_KODU": "posta_kodu",
            "DAIRE_NO": "daire_no",
            "KAT": "kat",
            "BLOK": "blok",
            # English field names
            "CITY": "city",
            "DISTRICT": "district",
            "NEIGHBORHOOD": "neighborhood",
            "STREET": "street",
            "BUILDING_NUMBER": "building_number",
            "POSTAL_CODE": "postal_code",
            "APARTMENT_NUMBER": "apartment_number",
            "FLOOR": "floor",
            "BLOCK": "block",
        }

        return label_mapping.get(label.upper(), label.lower())

    def _create_failed_enhanced_result(
        self, address_text: str, error_message: str, processing_time_ms: float
    ) -> EnhancedAddressOutput:
        """Create failed result in enhanced format"""
        return self.enhanced_formatter.format_enhanced_output(
            components={},
            original_address=address_text,
            confidence_data={"overall": 0.0, "pattern": 0.0, "ml": 0.0},
            processing_method="failed",
            processing_time_ms=processing_time_ms,
            additional_metadata={"error": error_message},
        )

    def set_enhanced_output(self, enabled: bool):
        """Enable or disable enhanced output format"""
        self.config.enhanced_output = enabled
        logger.info(f"Enhanced output {'enabled' if enabled else 'disabled'}")

    def get_enhanced_output_sample(self, address_text: str = None) -> str:
        """Get sample enhanced output for documentation"""
        sample_address = address_text or "Kızılay Mahallesi Atatürk Bulvarı No:123 Çankaya/ANKARA"

        # Process with enhanced output
        original_setting = self.config.enhanced_output
        self.config.enhanced_output = True

        try:
            result = self.normalize(sample_address)
            json_output = self.enhanced_formatter.to_json(result, indent=2)
            return json_output
        finally:
            self.config.enhanced_output = original_setting
