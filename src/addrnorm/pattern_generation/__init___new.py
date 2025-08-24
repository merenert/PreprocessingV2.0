"""
Pattern Generation Module

ML-based pattern generation and management system for address normalization.
"""

from .models import (
    PatternSuggestion,
    PatternConflict,
    PatternReview,
    AddressCluster,
    PatternGenerationConfig,
    ConflictType,
    ConflictSeverity,
    ResolutionStrategy,
    ReviewStatus,
    ValidationStatus,
)

from .ml_suggester import MLPatternSuggester, AddressFeatureExtractor
from .conflict_detector import PatternConflictDetector
from .pattern_analyzer import PatternAnalyzer
from .review_interface import PatternReviewInterface, ReviewCLI
from .cli import MLPatternCLI

__all__ = [
    # Models
    "PatternSuggestion",
    "PatternConflict",
    "PatternReview",
    "AddressCluster",
    "PatternGenerationConfig",
    "ConflictType",
    "ConflictSeverity",
    "ResolutionStrategy",
    "ReviewStatus",
    "ValidationStatus",
    # Core Components
    "MLPatternSuggester",
    "AddressFeatureExtractor",
    "PatternConflictDetector",
    "PatternAnalyzer",
    "PatternReviewInterface",
    "ReviewCLI",
    "MLPatternCLI",
]
