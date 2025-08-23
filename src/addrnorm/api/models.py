"""
Pydantic models for REST API requests and responses.
Pipeline uyumlu modeller - ProcessingResult yapısına uygun.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AddressRequest(BaseModel):
    """Single address normalization request."""

    text: str = Field(
        ..., min_length=1, max_length=1000, description="Address text to normalize"
    )


class BatchAddressRequest(BaseModel):
    """Batch address normalization request."""

    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of address texts to normalize",
    )


class AddressResponse(BaseModel):
    """Single address normalization response - ProcessingResult uyumlu."""

    raw_input: str = Field(..., description="Original input text")
    success: bool = Field(..., description="Whether processing was successful")
    address_out: Optional[Dict[str, Any]] = Field(
        None, description="Normalized address data (AddressOut as dict)"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_method: Optional[str] = Field(None, description="Processing method used")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )
    processing_time_ms: Optional[float] = Field(
        None, ge=0.0, description="Processing time in ms"
    )


class BatchAddressResponse(BaseModel):
    """Batch address normalization response."""

    results: List[AddressResponse] = Field(
        ..., description="List of normalization results"
    )
    total_count: int = Field(..., description="Total number of addresses processed")
    success_count: int = Field(..., description="Number of successful normalizations")
    error_count: int = Field(..., description="Number of failed normalizations")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    components: dict = Field(..., description="Component status")
