"""
Simplified Turkish Address Explanation Module

This module provides simple text processing for Turkish address explanations.
All explanations are stored as raw text without parsing landmarks or spatial relationships.

Example:
    >>> from addrnorm.explanation import process_explanation
    >>>
    >>> # Simple usage - just validates and cleans text
    >>> result = process_explanation("Migros yanı")
    >>> print(result)  # Returns cleaned text
"""


def process_explanation(text: str) -> str:
    """
    Process explanation text by cleaning and validating it.

    Args:
        text: Raw explanation text

    Returns:
        Cleaned explanation text
    """
    if not text or not isinstance(text, str):
        return ""

    # Basic cleaning
    cleaned = text.strip()

    # Remove extra whitespace
    import re

    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def parse_explanation(text: str) -> str:
    """
    Legacy compatibility function - now just returns cleaned text.

    Args:
        text: Raw explanation text

    Returns:
        Cleaned explanation text
    """
    return process_explanation(text)


__version__ = "2.0.0"
__author__ = "Turkish Address Normalization Team"

# Simplified exports
__all__ = [
    "process_explanation",
    "parse_explanation",  # For backward compatibility
]

# Quick usage examples
EXAMPLES = {
    "simple_landmark": {
        "input": "Migros yanı",
        "expected": {
            "type": "landmark",
            "landmark_name": "Migros",
            "landmark_type": "market",
            "spatial_relation": "yanı",
        },
    },
    "hotel_example": {
        "input": "Amorium Hotel karşısı",
        "expected": {
            "type": "landmark",
            "landmark_name": "Amorium Hotel",
            "landmark_type": "hotel",
            "spatial_relation": "karşısı",
        },
    },
    "business_example": {
        "input": "Şekerbank yanında",
        "expected": {
            "type": "landmark",
            "landmark_name": "Şekerbank",
            "landmark_type": "banka",
            "spatial_relation": "yanında",
        },
    },
    "complex_example": {
        "input": "Koç Holding binası arkasında",
        "expected": {
            "type": "landmark",
            "landmark_name": "Koç Holding",
            "landmark_type": "şirket",
            "spatial_relation": "arkasında",
        },
    },
}
