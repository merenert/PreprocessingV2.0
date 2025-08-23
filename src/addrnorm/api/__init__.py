"""
REST API module for Turkish address normalization.
"""

from .models import (
    AddressRequest,
    AddressResponse,
    BatchAddressRequest,
    BatchAddressResponse,
    HealthResponse,
)
from .server import app

__all__ = [
    "app",
    "AddressRequest",
    "AddressResponse",
    "BatchAddressRequest",
    "BatchAddressResponse",
    "HealthResponse",
]
