"""
Fallback normalization module for Turkish addresses.

This module provides rule-based normalization as a last resort when
pattern-based and ML-based methods fail to produce satisfactory results.

Key Features:
- Rule-based pattern extraction for common address components
- Heuristic field assignment for probable values
- Very low confidence assignments to avoid false positives
- Never generates fake/placeholder data - only real extracted values
"""

from .apply import (
    FallbackNormalizer,
    create_fallback_normalizer,
    normalize_address_fallback,
)
from .rules import NormalizationRule, TurkishAddressRules

__all__ = [
    "TurkishAddressRules",
    "NormalizationRule",
    "FallbackNormalizer",
    "create_fallback_normalizer",
    "normalize_address_fallback",
]
