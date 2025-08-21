"""
Contract models for address normalization output.
"""

import csv
import json
from enum import Enum
from io import StringIO
from typing import List, Optional

from pydantic import BaseModel, Field


class RelationEnum(str, Enum):
    """Spatial relation enumerations."""

    KARSISI = "karsisi"
    YANI = "yani"
    ARKASI = "arkasi"
    USTU = "ustu"
    ALTI = "alti"
    ICI = "ici"
    YANINDA = "yaninda"


class MethodEnum(str, Enum):
    """Processing method enumerations."""

    ML = "ml"
    PATTERN = "pattern"
    FALLBACK = "fallback"


class ExplanationParsed(BaseModel):
    """Parsed explanation with confidence and method info."""

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    method: MethodEnum = Field(..., description="Processing method used")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class AddressOut(BaseModel):
    """Normalized address output model."""

    country: Optional[str] = Field(
        None, pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 country code"
    )
    city: Optional[str] = Field(None, min_length=1, max_length=100)
    district: Optional[str] = Field(None, min_length=1, max_length=100)
    neighborhood: Optional[str] = Field(None, min_length=1, max_length=100)
    street: Optional[str] = Field(None, min_length=1, max_length=200)
    building: Optional[str] = Field(None, min_length=1, max_length=100)
    block: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9]+$")
    number: Optional[str] = Field(None, pattern=r"^[0-9A-Za-z/-]+$")
    entrance: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9]+$")
    floor: Optional[str] = Field(None, pattern=r"^[0-9-]+$")
    apartment: Optional[str] = Field(None, pattern=r"^[0-9A-Za-z]+$")
    postcode: Optional[str] = Field(
        None, pattern=r"^[0-9]{5}$", description="5-digit postal code"
    )
    relation: Optional[RelationEnum] = Field(
        None, description="Spatial relation to reference point"
    )
    explanation_raw: str = Field(..., description="Original raw address input")
    explanation_parsed: ExplanationParsed = Field(
        ..., description="Parsed explanation object"
    )
    normalized_address: str = Field(
        ..., min_length=1, description="Formatted normalized address string"
    )

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(**kwargs)

    def to_csv_row(self) -> List[str]:
        """Serialize to CSV row (list of strings)."""
        warnings_json = json.dumps(self.explanation_parsed.warnings, ensure_ascii=False)

        return [
            self.country or "",
            self.city or "",
            self.district or "",
            self.neighborhood or "",
            self.street or "",
            self.building or "",
            self.block or "",
            self.number or "",
            self.entrance or "",
            self.floor or "",
            self.apartment or "",
            self.postcode or "",
            self.relation.value if self.relation else "",
            self.explanation_raw,
            self.normalized_address,
            str(self.explanation_parsed.confidence),
            self.explanation_parsed.method.value,
            warnings_json,
        ]

    @classmethod
    def csv_headers(cls) -> List[str]:
        """Get CSV headers."""
        return [
            "country",
            "city",
            "district",
            "neighborhood",
            "street",
            "building",
            "block",
            "number",
            "entrance",
            "floor",
            "apartment",
            "postcode",
            "relation",
            "explanation_raw",
            "normalized_address",
            "confidence",
            "method",
            "warnings",
        ]

    @classmethod
    def to_csv_string(cls, addresses: List["AddressOut"]) -> str:
        """Convert list of addresses to CSV string."""
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(cls.csv_headers())

        # Write data rows
        for address in addresses:
            writer.writerow(address.to_csv_row())

        return output.getvalue()
