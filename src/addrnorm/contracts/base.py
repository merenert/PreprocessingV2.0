"""
Base contracts and data models for address normalization
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class ProcessingStatus(Enum):
    """Processing status enumeration"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ComponentConfidence:
    """Confidence score for address components"""

    value: float
    source: str = "model"
    method: str = "prediction"

    def __post_init__(self):
        """Validate confidence value"""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class AddressComponents:
    """Structured address components"""

    il: Optional[str] = None
    ilce: Optional[str] = None
    mahalle: Optional[str] = None
    sokak: Optional[str] = None
    yol: Optional[str] = None  # Alternative field name
    bina_no: Optional[str] = None
    daire_no: Optional[str] = None
    posta_kodu: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "il": self.il,
            "ilce": self.ilce,
            "mahalle": self.mahalle,
            "sokak": self.sokak,
            "yol": self.yol,
            "bina_no": self.bina_no,
            "daire_no": self.daire_no,
            "posta_kodu": self.posta_kodu,
        }


@dataclass
class ProcessingResult:
    """Result of address processing"""

    status: Optional[ProcessingStatus] = None
    components: Optional[AddressComponents] = None
    confidence: Optional[ComponentConfidence] = None
    original_text: str = ""
    processed_text: Optional[str] = None
    warnings: List[str] = None
    errors: List[str] = None
    metadata: Dict[str, Any] = None

    # Legacy interface compatibility
    success: Optional[bool] = None
    normalized_address: Optional[AddressComponents] = None
    component_confidence: Optional[ComponentConfidence] = None
    processing_method: Optional[str] = None
    processing_time: Optional[float] = None
    explanation: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values and handle legacy interface"""
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}

        # Handle legacy interface - priority for legacy params
        if self.success is not None:
            self.status = ProcessingStatus.SUCCESS if self.success else ProcessingStatus.FAILED
        elif self.status is None:
            self.status = ProcessingStatus.SUCCESS

        if self.normalized_address is not None:
            self.components = self.normalized_address
        elif self.components is None:
            self.components = AddressComponents()

        if self.component_confidence is not None:
            self.confidence = self.component_confidence
        elif self.confidence is None:
            self.confidence = ComponentConfidence(0.85)

        if self.processing_method is not None:
            self.metadata["processing_method"] = self.processing_method
        if self.processing_time is not None:
            self.metadata["processing_time"] = self.processing_time
        if self.explanation is not None:
            self.metadata["explanation"] = self.explanation
        if self.error_details is not None:
            self.metadata["error_details"] = self.error_details

    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return self.status in [ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "components": self.components.to_dict(),
            "confidence": self.confidence.value,
            "original_text": self.original_text,
            "processed_text": self.processed_text,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }
