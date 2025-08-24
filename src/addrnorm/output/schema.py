"""
Enhanced Output Schema

Defines the structure for enhanced address normalization output
with confidence scores, quality metrics, and validation status.
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)


class ProcessingMethod(str, Enum):
    """Processing method identifier"""

    PATTERN_PRIMARY = "pattern_primary"
    PATTERN_SECONDARY = "pattern_secondary"
    ML_PRIMARY = "ml_primary"
    ML_SECONDARY = "ml_secondary"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


class ValidationStatus(str, Enum):
    """Validation status levels"""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    NOT_VALIDATED = "not_validated"


@dataclass
class NormalizedAddressData:
    """Core normalized address data"""

    explanation_raw: str  # Yer-yön açıklaması (ham adres verisi değil)

    # Turkish field names
    il: Optional[str] = None
    ilce: Optional[str] = None
    mahalle: Optional[str] = None
    sokak: Optional[str] = None
    bina_no: Optional[str] = None
    daire_no: Optional[str] = None
    posta_kodu: Optional[str] = None
    kat: Optional[str] = None
    blok: Optional[str] = None

    # English field names
    city: Optional[str] = None
    district: Optional[str] = None
    neighborhood: Optional[str] = None
    street: Optional[str] = None
    building_number: Optional[str] = None
    apartment_number: Optional[str] = None
    postal_code: Optional[str] = None
    blok: Optional[str] = None

    @classmethod
    def from_components(cls, components: Dict[str, str], original_address: str = "") -> "NormalizedAddressData":
        """Create NormalizedAddressData from components dictionary"""

        # Yer-yön açıklamasını bileşenlerden oluştur
        explanation_raw = cls._generate_location_explanation(components)

        # Extract all possible field mappings
        return cls(
            explanation_raw=explanation_raw,
            # Direct field mapping
            il=components.get("il"),
            ilce=components.get("ilce"),
            mahalle=components.get("mahalle"),
            sokak=components.get("sokak"),
            bina_no=components.get("bina_no"),
            daire_no=components.get("daire_no"),
            posta_kodu=components.get("posta_kodu"),
            kat=components.get("kat"),
            blok=components.get("blok"),
            # Additional mappings from English fields
            city=components.get("city"),
            district=components.get("district"),
            neighborhood=components.get("neighborhood"),
            street=components.get("street"),
            building_number=components.get("building_number"),
            apartment_number=components.get("apartment_number"),
            postal_code=components.get("postal_code"),
        )

    @staticmethod
    def _generate_location_explanation(components: Dict[str, str]) -> str:
        """Bileşenlerden yer-yön açıklaması oluştur"""

        # Öncelik sırası: mahalle -> sokak -> ilçe -> il
        explanation_parts = []

        # Mahalle bilgisi varsa ekle
        mahalle = components.get("mahalle") or components.get("neighborhood")
        if mahalle:
            explanation_parts.append(f"{mahalle} Mahallesi")

        # Sokak bilgisi varsa ekle
        sokak = components.get("sokak") or components.get("street")
        if sokak:
            # Sokak tipini kontrol et
            if any(tip in sokak.lower() for tip in ["caddesi", "cd", "cad"]):
                explanation_parts.append(sokak)
            elif any(tip in sokak.lower() for tip in ["sokağı", "sk", "sok"]):
                explanation_parts.append(sokak)
            elif any(tip in sokak.lower() for tip in ["bulvarı", "blv", "bulvar"]):
                explanation_parts.append(sokak)
            else:
                explanation_parts.append(f"{sokak} Sokağı")

        # Bina numarası varsa ekle
        bina_no = components.get("bina_no") or components.get("building_number")
        if bina_no:
            explanation_parts.append(f"No: {bina_no}")

        # İlçe bilgisi varsa ekle
        ilce = components.get("ilce") or components.get("district")
        if ilce:
            explanation_parts.append(f"{ilce} İlçesi")

        # İl bilgisi varsa ekle
        il = components.get("il") or components.get("city")
        if il:
            explanation_parts.append(f"{il} İli")

        # Kat bilgisi varsa ekle
        kat = components.get("kat")
        if kat:
            explanation_parts.append(f"{kat}. Kat")

        # Daire bilgisi varsa ekle
        daire = components.get("daire_no") or components.get("apartment_number")
        if daire:
            explanation_parts.append(f"Daire: {daire}")

        # Açıklamayı birleştir
        if explanation_parts:
            return ", ".join(explanation_parts) + " konumunda"
        else:
            return "Belirtilen konumda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ConfidenceScores:
    """Confidence scores from different processing methods"""

    pattern: float = 0.0
    ml: float = 0.0
    overall: float = 0.0
    adaptive_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        result = {"pattern": round(self.pattern, 3), "ml": round(self.ml, 3), "overall": round(self.overall, 3)}
        if self.adaptive_threshold is not None:
            result["adaptive_threshold"] = round(self.adaptive_threshold, 3)
        return result


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""

    completeness: float = 0.0  # How complete is the extracted information
    consistency: float = 0.0  # Internal consistency of extracted data
    accuracy: Optional[float] = None  # Accuracy if ground truth available
    coverage: Optional[float] = None  # Pattern coverage

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        result = {"completeness": round(self.completeness, 3), "consistency": round(self.consistency, 3)}
        if self.accuracy is not None:
            result["accuracy"] = round(self.accuracy, 3)
        if self.coverage is not None:
            result["coverage"] = round(self.coverage, 3)
        return result


@dataclass
class ProcessingMetadata:
    """Additional processing metadata"""

    processing_time_ms: Optional[float] = None
    pattern_id: Optional[str] = None
    model_version: Optional[str] = None
    warnings: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {}
        if self.processing_time_ms is not None:
            result["processing_time_ms"] = round(self.processing_time_ms, 2)
        if self.pattern_id:
            result["pattern_id"] = self.pattern_id
        if self.model_version:
            result["model_version"] = self.model_version
        if self.warnings:
            result["warnings"] = self.warnings
        if self.debug_info:
            result["debug_info"] = self.debug_info
        return result


@dataclass
class EnhancedAddressOutput:
    """Enhanced address normalization output"""

    success: bool
    normalized_address: NormalizedAddressData
    confidence_scores: ConfidenceScores
    quality_metrics: QualityMetrics
    processing_metadata: ProcessingMetadata
    original_address: str
    processing_method: Optional[ProcessingMethod] = None
    validation_status: Optional[ValidationStatus] = None

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            "success": self.success,
            "normalized_address": self.normalized_address.to_dict(),
            "confidence_scores": self.confidence_scores.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "original_address": self.original_address,
        }

        if self.processing_method:
            result["processing_method"] = self.processing_method.value
        if self.validation_status:
            result["validation_status"] = self.validation_status.value
        if include_metadata and self.processing_metadata:
            result["processing_metadata"] = self.processing_metadata.to_dict()

        return result

    def to_json(self, indent: Optional[int] = None, include_metadata: bool = True) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(include_metadata=include_metadata), ensure_ascii=False, indent=indent)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility"""
        # Legacy format: flat structure with only core fields
        legacy = self.normalized_address.to_dict()

        # Add confidence as single field for legacy compatibility
        if hasattr(self.confidence_scores, "overall"):
            legacy["confidence"] = self.confidence_scores.overall

        return legacy

    def to_legacy_format(self) -> Dict[str, str]:
        """Convert to legacy format as used by old system"""
        legacy_dict = self.to_legacy_dict()

        # Convert all values to strings for backward compatibility
        return {k: str(v) if v is not None else "" for k, v in legacy_dict.items()}


class EnhancedOutputSchema(BaseModel):
    """Pydantic model for enhanced output validation"""

    normalized_address: Dict[str, Any] = Field(..., description="Normalized address components")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores from different methods")
    processing_method: ProcessingMethod = Field(..., description="Primary processing method used")
    validation_status: ValidationStatus = Field(..., description="Validation result status")
    quality_metrics: Dict[str, float] = Field(..., description="Quality assessment metrics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional processing metadata")

    class Config:
        """Pydantic configuration"""

        use_enum_values = True
        extra = "forbid"  # Don't allow extra fields

    def to_enhanced_output(self) -> EnhancedAddressOutput:
        """Convert to EnhancedAddressOutput dataclass"""
        norm_addr = NormalizedAddressData(**self.normalized_address)
        conf_scores = ConfidenceScores(**self.confidence_scores)
        qual_metrics = QualityMetrics(**self.quality_metrics)

        metadata = None
        if self.metadata:
            metadata = ProcessingMetadata(**self.metadata)

        return EnhancedAddressOutput(
            normalized_address=norm_addr,
            confidence_scores=conf_scores,
            processing_method=self.processing_method,
            validation_status=self.validation_status,
            quality_metrics=qual_metrics,
            metadata=metadata,
        )


def create_enhanced_output(
    explanation_raw: str,
    extracted_components: Dict[str, str],
    pattern_confidence: float = 0.0,
    ml_confidence: float = 0.0,
    processing_method: ProcessingMethod = ProcessingMethod.PATTERN_PRIMARY,
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED,
    processing_time_ms: Optional[float] = None,
    pattern_id: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> EnhancedAddressOutput:
    """
    Factory function to create enhanced output from basic inputs

    Args:
        explanation_raw: Original input text
        extracted_components: Extracted address components
        pattern_confidence: Pattern matching confidence
        ml_confidence: ML model confidence
        processing_method: Processing method used
        validation_status: Validation result
        processing_time_ms: Processing time in milliseconds
        pattern_id: Pattern ID if pattern matching was used
        warnings: List of warning messages

    Returns:
        EnhancedAddressOutput object
    """

    # Create normalized address data
    normalized_address = NormalizedAddressData(
        explanation_raw=explanation_raw,
        il=extracted_components.get("il") or extracted_components.get("city"),
        ilce=extracted_components.get("ilce") or extracted_components.get("district"),
        mahalle=extracted_components.get("mahalle") or extracted_components.get("neighborhood"),
        sokak=extracted_components.get("sokak") or extracted_components.get("street"),
        bina_no=extracted_components.get("bina_no") or extracted_components.get("building_number"),
        daire_no=extracted_components.get("daire_no") or extracted_components.get("apartment_number"),
        posta_kodu=extracted_components.get("posta_kodu") or extracted_components.get("postal_code"),
        kat=extracted_components.get("kat") or extracted_components.get("floor"),
        blok=extracted_components.get("blok") or extracted_components.get("block"),
    )

    # Calculate overall confidence (weighted average)
    overall_confidence = 0.0
    if pattern_confidence > 0 and ml_confidence > 0:
        # Both methods used - weighted average
        overall_confidence = pattern_confidence * 0.6 + ml_confidence * 0.4
    elif pattern_confidence > 0:
        overall_confidence = pattern_confidence
    elif ml_confidence > 0:
        overall_confidence = ml_confidence

    confidence_scores = ConfidenceScores(pattern=pattern_confidence, ml=ml_confidence, overall=overall_confidence)

    # Calculate quality metrics
    components_found = len([v for v in normalized_address.to_dict().values() if v and v != explanation_raw])
    total_possible = 8  # il, ilce, mahalle, sokak, bina_no, daire_no, posta_kodu, kat
    completeness = components_found / total_possible if total_possible > 0 else 0.0

    # Simple consistency check - if we have city/district, consistency is higher
    consistency = 0.5
    if normalized_address.il and normalized_address.ilce:
        consistency += 0.3
    if normalized_address.mahalle:
        consistency += 0.2
    consistency = min(consistency, 1.0)

    quality_metrics = QualityMetrics(completeness=completeness, consistency=consistency)

    # Create metadata
    metadata = ProcessingMetadata(processing_time_ms=processing_time_ms, pattern_id=pattern_id, warnings=warnings or [])

    return EnhancedAddressOutput(
        normalized_address=normalized_address,
        confidence_scores=confidence_scores,
        processing_method=processing_method,
        validation_status=validation_status,
        quality_metrics=quality_metrics,
        metadata=metadata,
    )


# Example usage and validation
def validate_enhanced_output(output_dict: Dict[str, Any]) -> bool:
    """
    Validate enhanced output against schema

    Args:
        output_dict: Output dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        EnhancedOutputSchema(**output_dict)
        return True
    except Exception as e:
        logger.error(f"Enhanced output validation failed: {e}")
        return False


# Example output structure
EXAMPLE_ENHANCED_OUTPUT = {
    "normalized_address": {
        "explanation_raw": "Migros yanı apartman",
        "il": "ANKARA",
        "ilce": "ÇANKAYA",
        "mahalle": "ATATÜRK",
        "sokak": "CUMHURIYET CADDESİ",
        "bina_no": "15",
        "daire_no": "",
        "posta_kodu": "06100",
    },
    "confidence_scores": {"pattern": 0.95, "ml": 0.87, "overall": 0.91},
    "processing_method": "pattern_primary",
    "validation_status": "passed",
    "quality_metrics": {"completeness": 0.9, "consistency": 0.85},
    "metadata": {"processing_time_ms": 12.5, "pattern_id": "turkish_address_v2", "warnings": []},
}
