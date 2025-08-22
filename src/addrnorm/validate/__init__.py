"""
Address validation module for Turkish addresses.

This module provides geographic validation including:
- City/district standardization with fuzzy matching
- City-district consistency checks
- Postal code validation
- Support for typo correction
"""

from .geo import (
    ConsistencyResult,
    TurkishGeoValidator,
    ValidationResult,
    create_geo_validator,
)

__all__ = [
    "TurkishGeoValidator",
    "ValidationResult",
    "ConsistencyResult",
    "create_geo_validator",
]
