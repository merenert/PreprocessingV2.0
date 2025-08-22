"""
Pattern matching module for Turkish address normalization.

This module provides DSL-based pattern matching capabilities for extracting
structured information from preprocessed Turkish address text.

Main components:
- PatternCompiler: Compiles YAML pattern definitions to regex
- PatternMatcher: Matches text against patterns with confidence scoring
- Pattern DSL: YAML-based domain-specific language for defining address patterns

Example usage:
    from addrnorm.patterns import PatternMatcher
    from addrnorm.preprocess import preprocess

    # Preprocess text first
    result = preprocess("İstanbul Kadıköy Moda Mah. Bahariye Cad. No:12")

    # Match against patterns
    matcher = PatternMatcher()
    matches = matcher.match_text(result['text'])

    if matches:
        best_match = matches[0]
        print(f"Pattern: {best_match.pattern_id}")
        print(f"Confidence: {best_match.confidence}")
        print(f"Slots: {best_match.slots}")
"""

from .compiler import CompiledPattern, CompiledSlot, PatternCompiler
from .matcher import MatchResult, PatternMatcher
from .thresholds import PatternStats, ThresholdManager

__all__ = [
    "PatternCompiler",
    "CompiledPattern",
    "CompiledSlot",
    "PatternMatcher",
    "MatchResult",
    "ThresholdManager",
    "PatternStats",
]
