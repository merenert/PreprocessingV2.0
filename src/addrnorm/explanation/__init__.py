"""
Turkish Address Explanation Parser Module

This module provides tools for parsing Turkish address explanations that contain
landmark references and spatial relationships.

Example:
    >>> from addrnorm.explanation import parse_explanation, ExplanationParser
    >>> 
    >>> # Quick usage
    >>> result = parse_explanation("Migros yanı")
    >>> print(result)
    >>> 
    >>> # Advanced usage
    >>> parser = ExplanationParser()
    >>> result = parser.parse("Amorium Hotel karşısı")
    >>> print(f"Landmark: {result.landmark.name}")
    >>> print(f"Relation: {result.relation.relation}")

Main Components:
    - ExplanationParser: Main parsing class
    - LandmarkDetector: Detects businesses and points of interest
    - SpatialRelationExtractor: Extracts spatial relationships
    - ExplanationResult: Structured result format
"""

from .parser import (
    ExplanationParser,
    create_parser,
    parse_explanation,
    extract_landmark_info
)

from .models import (
    ExplanationResult,
    ExplanationConfig,
    Landmark,
    SpatialRelation
)

from .landmarks import LandmarkDetector
from .relations import SpatialRelationExtractor

__version__ = "1.0.0"
__author__ = "Turkish Address Normalization Team"

# Main exports
__all__ = [
    # Main parser
    'ExplanationParser',
    'create_parser',
    'parse_explanation',
    'extract_landmark_info',
    
    # Data models
    'ExplanationResult',
    'ExplanationConfig', 
    'Landmark',
    'SpatialRelation',
    
    # Component classes
    'LandmarkDetector',
    'SpatialRelationExtractor'
]

# Quick usage examples
EXAMPLES = {
    "simple_landmark": {
        "input": "Migros yanı",
        "expected": {
            "type": "landmark",
            "landmark_name": "Migros",
            "landmark_type": "market",
            "spatial_relation": "yanı"
        }
    },
    "hotel_example": {
        "input": "Amorium Hotel karşısı", 
        "expected": {
            "type": "landmark",
            "landmark_name": "Amorium Hotel",
            "landmark_type": "hotel",
            "spatial_relation": "karşısı"
        }
    },
    "business_example": {
        "input": "Şekerbank yanında",
        "expected": {
            "type": "landmark", 
            "landmark_name": "Şekerbank",
            "landmark_type": "banka",
            "spatial_relation": "yanında"
        }
    },
    "complex_example": {
        "input": "Koç Holding binası arkasında",
        "expected": {
            "type": "landmark",
            "landmark_name": "Koç Holding",
            "landmark_type": "şirket", 
            "spatial_relation": "arkasında"
        }
    }
}
