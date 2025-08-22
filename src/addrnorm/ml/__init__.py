"""
Machine Learning module for Turkish address parsing.

This module provides NER-based fallback when pattern matching fails.
"""

from .infer import (
    MLAddressNormalizer,
    get_ml_normalizer,
    is_ml_available,
    normalize_with_ml_fallback,
)
from .ner_baseline import TurkishAddressNER, load_training_data, save_training_data

__all__ = [
    "TurkishAddressNER",
    "load_training_data",
    "save_training_data",
    "MLAddressNormalizer",
    "get_ml_normalizer",
    "normalize_with_ml_fallback",
    "is_ml_available",
]
