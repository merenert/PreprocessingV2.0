"""
Confidence Scoring System

Multi-level confidence calculation for address normalization results.
Provides pattern confidence, ML confidence, and overall confidence scores.
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ProcessingMethod(Enum):
    """Address processing method identifier"""

    PATTERN_PRIMARY = "pattern_primary"
    PATTERN_SECONDARY = "pattern_secondary"
    ML_PRIMARY = "ml_primary"
    ML_SECONDARY = "ml_secondary"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass
@dataclass
class ProcessingContext:
    """Context information for processing"""

    method: ProcessingMethod
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    pattern_id: Optional[str] = None
    input_length: Optional[int] = None
    method_used: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context metadata or direct attribute"""
        # Check direct attributes first
        if hasattr(self, key):
            return getattr(self, key)
        # Then check metadata
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context metadata"""
        self.metadata[key] = value


@dataclass
class PatternConfidence:
    """Pattern-based confidence metrics"""

    match_score: float  # 0.0-1.0: How well pattern matched
    pattern_quality: float  # 0.0-1.0: Quality of the pattern itself
    coverage_score: float  # 0.0-1.0: How much of address was covered
    specificity: float  # 0.0-1.0: How specific the pattern is

    @property
    def overall(self) -> float:
        """Calculate overall pattern confidence"""
        # Weighted average with emphasis on match quality
        weights = [0.4, 0.3, 0.2, 0.1]  # match, quality, coverage, specificity
        scores = [self.match_score, self.pattern_quality, self.coverage_score, self.specificity]
        return sum(w * s for w, s in zip(weights, scores))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation"""
        return {
            "match_score": self.match_score,
            "pattern_quality": self.pattern_quality,
            "coverage_score": self.coverage_score,
            "specificity": self.specificity,
            "overall": self.overall,
        }


@dataclass
class MLConfidence:
    """ML-based confidence metrics"""

    model_confidence: float  # 0.0-1.0: Model's internal confidence
    prediction_entropy: float  # 0.0-1.0: Entropy of predictions (lower = more confident)
    feature_quality: float  # 0.0-1.0: Quality of input features
    training_similarity: float  # 0.0-1.0: Similarity to training data

    @property
    def overall(self) -> float:
        """Calculate overall ML confidence"""
        # Convert entropy to confidence (1 - entropy)
        entropy_confidence = 1.0 - self.prediction_entropy

        # Weighted average
        weights = [0.4, 0.25, 0.2, 0.15]  # model, entropy, features, similarity
        scores = [self.model_confidence, entropy_confidence, self.feature_quality, self.training_similarity]
        return sum(w * s for w, s in zip(weights, scores))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_confidence": self.model_confidence,
            "prediction_entropy": self.prediction_entropy,
            "feature_quality": self.feature_quality,
            "training_similarity": self.training_similarity,
            "overall": self.overall,
        }


@dataclass
class ConfidenceScores:
    """Complete confidence scoring for address normalization"""

    pattern: Union[float, PatternConfidence]  # Overall pattern confidence or PatternConfidence object
    ml: Union[float, MLConfidence]  # Overall ML confidence or MLConfidence object
    processing_method: ProcessingMethod
    primary_method_confidence: float
    overall: Optional[float] = None  # Combined overall confidence
    secondary_method_confidence: Optional[float] = None

    # Backward compatibility - accept old parameter names
    pattern_details: Optional[PatternConfidence] = None
    ml_details: Optional[MLConfidence] = None

    # Detailed breakdowns
    pattern_breakdown: Optional[PatternConfidence] = None
    ml_breakdown: Optional[MLConfidence] = None

    def __init__(
        self,
        pattern=None,
        ml=None,
        processing_method=None,
        primary_method_confidence=None,
        overall=None,
        secondary_method_confidence=None,
        pattern_details=None,
        ml_details=None,
        **kwargs,
    ):
        """Flexible constructor for backward compatibility"""

        # Handle backward compatibility cases
        if pattern_details is not None:
            self.pattern = pattern_details.overall if hasattr(pattern_details, "overall") else 0.8
            self.pattern_details = pattern_details
            self.pattern_breakdown = pattern_details
        elif pattern is not None:
            self.pattern = pattern
            self.pattern_details = pattern if hasattr(pattern, "overall") else None
        else:
            self.pattern = 0.8  # Default

        if ml_details is not None:
            self.ml = ml_details.overall if hasattr(ml_details, "overall") else 0.8
            self.ml_details = ml_details
            self.ml_breakdown = ml_details
        elif ml is not None:
            self.ml = ml
            self.ml_details = ml if hasattr(ml, "overall") else None
        else:
            self.ml = 0.8  # Default

        # Set required fields with defaults
        self.processing_method = processing_method or ProcessingMethod.PATTERN_PRIMARY
        self.primary_method_confidence = (
            primary_method_confidence or max(self.pattern, self.ml)
            if isinstance(self.pattern, (int, float)) and isinstance(self.ml, (int, float))
            else 0.8
        )

        # Calculate overall if not provided
        if overall is not None:
            self.overall = overall
        else:
            pattern_val = self.pattern if isinstance(self.pattern, (int, float)) else self.pattern.overall
            ml_val = self.ml if isinstance(self.ml, (int, float)) else self.ml.overall
            self.overall = (pattern_val + ml_val) / 2.0

        self.secondary_method_confidence = secondary_method_confidence

    def __post_init__(self):
        """This won't be called due to custom __init__, but kept for compatibility"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        # Handle pattern value - could be float or PatternConfidence object
        pattern_val = self.pattern.overall if hasattr(self.pattern, "overall") else self.pattern
        # Handle ml value - could be float or MLConfidence object
        ml_val = self.ml.overall if hasattr(self.ml, "overall") else self.ml

        result = {"pattern": round(pattern_val, 3), "ml": round(ml_val, 3), "overall": round(self.overall, 3)}

        # Add details if available
        if self.pattern_details:
            result["pattern_breakdown"] = {
                "match_score": round(self.pattern_details.match_score, 3),
                "pattern_quality": round(self.pattern_details.pattern_quality, 3),
                "coverage_score": round(self.pattern_details.coverage_score, 3),
                "specificity": round(self.pattern_details.specificity, 3),
            }

        if self.ml_details:
            result["ml_breakdown"] = {
                "model_confidence": round(self.ml_details.model_confidence, 3),
                "prediction_entropy": round(self.ml_details.prediction_entropy, 3),
                "feature_quality": round(self.ml_details.feature_quality, 3),
                "training_similarity": round(self.ml_details.training_similarity, 3),
            }

        return result


class ConfidenceCalculator:
    """
    Multi-level confidence calculator for address normalization results

    Calculates confidence scores at multiple levels:
    - Pattern matching confidence
    - ML model confidence
    - Overall combined confidence
    """

    def __init__(
        self,
        pattern_weight: float = 0.6,
        ml_weight: float = 0.4,
        min_confidence_threshold: float = 0.3,
        enable_detailed_breakdown: bool = True,
    ):
        """
        Initialize confidence calculator

        Args:
            pattern_weight: Weight for pattern-based confidence (0.0-1.0)
            ml_weight: Weight for ML-based confidence (0.0-1.0)
            min_confidence_threshold: Minimum acceptable confidence
            enable_detailed_breakdown: Whether to include detailed breakdowns
        """
        if abs(pattern_weight + ml_weight - 1.0) > 0.001:
            raise ValueError("Pattern and ML weights must sum to 1.0")

        self.pattern_weight = pattern_weight
        self.ml_weight = ml_weight
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_detailed_breakdown = enable_detailed_breakdown

        self.logger = logging.getLogger(__name__)

        # Confidence calculation cache
        self._confidence_cache: Dict[str, ConfidenceScores] = {}

    def calculate_confidence(self, result: Dict[str, Any], context: "ProcessingContext") -> ConfidenceScores:
        """
        Calculate confidence for processing result (legacy interface)

        Args:
            result: Processing result dictionary
            context: Processing context

        Returns:
            ConfidenceScores with calculated confidence values
        """
        # Extract values from result
        components = result.get("components", {})
        success = result.get("success", True)

        # Calculate detailed confidence metrics
        result_dict = {"components": components, "success": success}
        context_dict = {"pattern_match_score": 0.8 if success else 0.3, "pattern_quality": 0.7}
        pattern_details = self._calculate_pattern_confidence(result_dict, context_dict)
        ml_details = self._calculate_ml_confidence(result_dict, context_dict)

        pattern_conf = pattern_details.overall if pattern_details else (0.8 if success else 0.3)
        ml_conf = ml_details.overall if ml_details else (0.7 if success else 0.2)

        # Use enhanced calculation method
        return self.calculate_enhanced_confidence(
            extracted_components=components,
            pattern_confidence=pattern_conf,
            ml_confidence=ml_conf,
            processing_method=getattr(context, "method", "hybrid"),
            input_text=result.get("original_text", ""),
            processing_time_ms=getattr(context, "processing_time_ms", None),
            pattern_details=pattern_details,
            ml_details=ml_details,
            pattern_id=getattr(context, "pattern_id", None),
        )

    def calculate_enhanced_confidence(
        self,
        extracted_components: Dict[str, str],
        pattern_confidence: float = 0.0,
        ml_confidence: float = 0.0,
        processing_method: str = "pattern_primary",
        input_text: str = "",
        processing_time_ms: Optional[float] = None,
        pattern_id: Optional[str] = None,
        pattern_details: Optional[PatternConfidence] = None,
        ml_details: Optional[MLConfidence] = None,
    ) -> ConfidenceScores:
        """
        Calculate confidence scores for enhanced output format

        Args:
            extracted_components: Extracted address components
            pattern_confidence: Raw pattern matching confidence
            ml_confidence: Raw ML model confidence
            processing_method: Processing method used
            input_text: Original input text
            processing_time_ms: Processing time in milliseconds
            pattern_id: Pattern ID if pattern matching was used

        Returns:
            ConfidenceScores with calculated values
        """

        # Validate and normalize confidence scores
        pattern_conf = max(0.0, min(1.0, pattern_confidence))
        ml_conf = max(0.0, min(1.0, ml_confidence))

        # Calculate overall confidence based on processing method
        if processing_method in ["pattern_primary", "pattern_secondary"]:
            if pattern_conf > 0:
                overall_conf = pattern_conf
                # Boost if ML also agrees
                if ml_conf > 0.5:
                    overall_conf = min(1.0, pattern_conf + (ml_conf - 0.5) * 0.2)
            else:
                overall_conf = ml_conf * 0.7  # Fallback to ML with penalty

        elif processing_method in ["ml_primary", "ml_secondary"]:
            if ml_conf > 0:
                overall_conf = ml_conf
                # Boost if pattern also agrees
                if pattern_conf > 0.5:
                    overall_conf = min(1.0, ml_conf + (pattern_conf - 0.5) * 0.2)
            else:
                overall_conf = pattern_conf * 0.7  # Fallback to pattern with penalty

        elif processing_method == "hybrid":
            # Weighted combination for hybrid approach
            overall_conf = pattern_conf * self.pattern_weight + ml_conf * self.ml_weight

        else:  # fallback
            overall_conf = max(pattern_conf, ml_conf) * 0.5

        # Apply component completeness bonus
        completeness_bonus = self._calculate_completeness_bonus(extracted_components)
        overall_conf = min(1.0, overall_conf + completeness_bonus)

        # Apply consistency penalty/bonus
        consistency_modifier = self._calculate_consistency_modifier(extracted_components, input_text)
        overall_conf = max(0.0, min(1.0, overall_conf + consistency_modifier))

        # Create confidence scores
        confidence_scores = ConfidenceScores(
            pattern=pattern_conf,
            ml=ml_conf,
            overall=overall_conf,
            primary_method_confidence=overall_conf,
            processing_method=(
                ProcessingMethod(processing_method)
                if processing_method
                in ["pattern_primary", "pattern_secondary", "ml_primary", "ml_secondary", "hybrid", "fallback"]
                else ProcessingMethod.FALLBACK
            ),
            pattern_details=pattern_details,
            ml_details=ml_details,
        )

        return confidence_scores

    def _calculate_completeness_bonus(self, components: Dict[str, str]) -> float:
        """Calculate bonus based on component completeness"""
        if not components:
            return 0.0

        # Define component importance weights
        component_weights = {
            "il": 0.25,
            "city": 0.25,
            "ilce": 0.20,
            "district": 0.20,
            "mahalle": 0.15,
            "neighborhood": 0.15,
            "sokak": 0.15,
            "street": 0.15,
            "bina_no": 0.10,
            "building_number": 0.10,
            "posta_kodu": 0.10,
            "postal_code": 0.10,
            "daire_no": 0.05,
            "apartment_number": 0.05,
        }

        total_weight = 0.0
        found_weight = 0.0

        for component, value in components.items():
            if component in component_weights:
                weight = component_weights[component]
                total_weight += weight
                if value and value.strip():
                    found_weight += weight

        if total_weight == 0:
            return 0.0

        completeness_ratio = found_weight / total_weight
        # Convert to bonus (max 0.1 bonus for full completeness)
        return completeness_ratio * 0.1

    def _calculate_consistency_modifier(self, components: Dict[str, str], input_text: str) -> float:
        """Calculate consistency modifier based on logical relationships"""
        modifier = 0.0

        # Check for logical consistency
        city = components.get("il") or components.get("city", "")
        district = components.get("ilce") or components.get("district", "")
        neighborhood = components.get("mahalle") or components.get("neighborhood", "")

        # Bonus for hierarchical consistency
        if city and district:
            modifier += 0.02  # Both city and district found
        if district and neighborhood:
            modifier += 0.02  # District and neighborhood found

        # Penalty for contradictions (basic check)
        components_text = " ".join([v for v in components.values() if v])
        input_lower = input_text.lower()

        # Check if extracted components make sense with input
        major_components = [city, district, neighborhood]
        major_components = [c for c in major_components if c and len(c) > 2]

        found_in_input = 0
        for component in major_components:
            if component.lower() in input_lower:
                found_in_input += 1

        if major_components and found_in_input == 0:
            modifier -= 0.05  # Penalty if major components not in input
        elif len(major_components) > 0:
            consistency_ratio = found_in_input / len(major_components)
            if consistency_ratio < 0.5:
                modifier -= 0.03  # Partial penalty

        return modifier

        try:
            # Determine processing method
            processing_method = self._determine_processing_method(processing_context)

            # Calculate pattern confidence
            pattern_confidence = self._calculate_pattern_confidence(normalization_result, processing_context)

            # Calculate ML confidence
            ml_confidence = self._calculate_ml_confidence(normalization_result, processing_context)

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(pattern_confidence, ml_confidence, processing_method)

            # Get primary/secondary method confidences
            primary_conf, secondary_conf = self._get_method_confidences(pattern_confidence, ml_confidence, processing_method)

            # Create detailed breakdowns if enabled
            pattern_details = None
            ml_details = None

            if self.enable_detailed_breakdown:
                pattern_details = self._create_pattern_details(processing_context)
                ml_details = self._create_ml_details(processing_context)

            # Create final confidence scores
            confidence_scores = ConfidenceScores(
                pattern=pattern_details.overall if pattern_details else pattern_confidence,
                ml=ml_details.overall if ml_details else ml_confidence,
                overall=overall_confidence,
                processing_method=processing_method,
                primary_method_confidence=primary_conf,
                secondary_method_confidence=secondary_conf,
                pattern_details=pattern_details,
                ml_details=ml_details,
            )

            # Cache result
            self._confidence_cache[cache_key] = confidence_scores

            self.logger.debug(f"Calculated confidence: {overall_confidence:.3f} using {processing_method.value}")

            return confidence_scores

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            # Return fallback confidence
            return self._create_fallback_confidence(processing_context)

    def _determine_processing_method(self, context: Dict[str, Any]) -> ProcessingMethod:
        """Determine which processing method was used"""
        method_used = context.get("method_used", "unknown")

        method_mapping = {
            "pattern_only": ProcessingMethod.PATTERN_PRIMARY,
            "pattern_primary": ProcessingMethod.PATTERN_PRIMARY,
            "pattern_fallback": ProcessingMethod.PATTERN_SECONDARY,
            "ml_only": ProcessingMethod.ML_PRIMARY,
            "ml_primary": ProcessingMethod.ML_PRIMARY,
            "ml_fallback": ProcessingMethod.ML_SECONDARY,
            "hybrid": ProcessingMethod.HYBRID,
            "fallback": ProcessingMethod.FALLBACK,
        }

        return method_mapping.get(method_used, ProcessingMethod.FALLBACK)

    def _calculate_pattern_confidence(self, result: Dict[str, Any], context: Dict[str, Any]) -> PatternConfidence:
        """Calculate pattern-based confidence"""
        # Pattern match score
        match_score = context.get("pattern_match_score", 0.5)

        # Pattern quality (from pattern validation)
        pattern_quality = context.get("pattern_quality", 0.7)

        # Coverage (how much of address was captured)
        extracted_fields = result.get("extracted_fields", {})
        total_fields = len(extracted_fields)
        non_empty_fields = sum(1 for v in extracted_fields.values() if v and str(v).strip())
        coverage = non_empty_fields / max(total_fields, 1) if total_fields > 0 else 0.0

        # Pattern specificity (more specific patterns = higher confidence)
        specificity = context.get("pattern_specificity", 0.6)

        return PatternConfidence(match_score, pattern_quality, coverage, specificity)

    def _calculate_ml_confidence(self, result: Dict[str, Any], context: Dict[str, Any]) -> MLConfidence:
        """Calculate ML-based confidence"""
        # Model's internal confidence
        model_confidence = context.get("ml_model_confidence", 0.5)

        # Prediction entropy (lower = more confident)
        entropy = context.get("prediction_entropy", 0.5)

        # Feature quality
        feature_quality = context.get("feature_quality", 0.7)

        # Training data similarity
        similarity = context.get("training_similarity", 0.6)

        return MLConfidence(model_confidence, entropy, feature_quality, similarity)

    def _calculate_overall_confidence(
        self, pattern_conf: Union[float, PatternConfidence], ml_conf: Union[float, MLConfidence], method: ProcessingMethod
    ) -> float:
        """Calculate overall confidence based on processing method"""

        # Extract float values from objects if needed
        pattern_val = pattern_conf.overall if hasattr(pattern_conf, "overall") else pattern_conf
        ml_val = ml_conf.overall if hasattr(ml_conf, "overall") else ml_conf

        if method == ProcessingMethod.PATTERN_PRIMARY:
            # Pattern was primary, give it more weight
            return 0.8 * pattern_val + 0.2 * ml_val
        elif method == ProcessingMethod.ML_PRIMARY:
            # ML was primary, give it more weight
            return 0.2 * pattern_val + 0.8 * ml_val
        elif method == ProcessingMethod.HYBRID:
            # Balanced hybrid approach
            return self.pattern_weight * pattern_val + self.ml_weight * ml_val
        else:
            # Fallback or secondary methods get penalty
            base_confidence = self.pattern_weight * pattern_val + self.ml_weight * ml_val
            penalty = 0.1 if method in [ProcessingMethod.PATTERN_SECONDARY, ProcessingMethod.ML_SECONDARY] else 0.2
            return max(0.0, base_confidence - penalty)

    def _get_method_confidences(
        self, pattern_conf: Union[float, PatternConfidence], ml_conf: Union[float, MLConfidence], method: ProcessingMethod
    ) -> tuple[float, Optional[float]]:
        """Get primary and secondary method confidences"""

        # Extract float values from objects if needed
        pattern_val = pattern_conf.overall if hasattr(pattern_conf, "overall") else pattern_conf
        ml_val = ml_conf.overall if hasattr(ml_conf, "overall") else ml_conf

        if method in [ProcessingMethod.PATTERN_PRIMARY, ProcessingMethod.PATTERN_SECONDARY]:
            return pattern_val, ml_val if ml_val > 0 else None
        elif method in [ProcessingMethod.ML_PRIMARY, ProcessingMethod.ML_SECONDARY]:
            return ml_val, pattern_val if pattern_val > 0 else None
        else:
            # Hybrid or fallback
            return max(pattern_val, ml_val), min(pattern_val, ml_val)

    def _create_pattern_details(self, context: Dict[str, Any]) -> PatternConfidence:
        """Create detailed pattern confidence breakdown"""
        return PatternConfidence(
            match_score=context.get("pattern_match_score", 0.5),
            pattern_quality=context.get("pattern_quality", 0.7),
            coverage_score=context.get("pattern_coverage", 0.6),
            specificity=context.get("pattern_specificity", 0.6),
        )

    def _create_ml_details(self, context: Dict[str, Any]) -> MLConfidence:
        """Create detailed ML confidence breakdown"""
        return MLConfidence(
            model_confidence=context.get("ml_model_confidence", 0.5),
            prediction_entropy=context.get("prediction_entropy", 0.5),
            feature_quality=context.get("feature_quality", 0.7),
            training_similarity=context.get("training_similarity", 0.6),
        )

    def _create_fallback_confidence(self, context: Dict[str, Any]) -> ConfidenceScores:
        """Create fallback confidence when calculation fails"""
        return ConfidenceScores(
            pattern=0.3, ml=0.3, overall=0.3, processing_method=ProcessingMethod.FALLBACK, primary_method_confidence=0.3
        )

    def _create_cache_key(self, result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create cache key for confidence calculation"""
        # Use hash of key components
        key_components = [
            str(result.get("normalized_address", "")),
            str(context.get("method_used", "")),
            str(context.get("pattern_match_score", 0)),
            str(context.get("ml_model_confidence", 0)),
        ]
        return str(hash("_".join(key_components)))

    def get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"

    def is_acceptable_confidence(self, confidence: float) -> bool:
        """Check if confidence meets minimum threshold"""
        return confidence >= self.min_confidence_threshold

    def clear_cache(self):
        """Clear confidence calculation cache"""
        self._confidence_cache.clear()
        self.logger.debug("Confidence calculation cache cleared")
