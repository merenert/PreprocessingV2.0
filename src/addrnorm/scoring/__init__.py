"""
Enhanced Scoring System for Address Normalization

Bu modül gelişmiş güven skorları ve kalite metrikleri sağlar.

Modules:
    confidence: Multi-level confidence scoring
    quality: Quality assessment and metrics
"""

from .confidence import ConfidenceCalculator, ConfidenceScores, ProcessingMethod, PatternConfidence, MLConfidence

from .quality import QualityAssessment, QualityMetrics, CompletenessScore, ConsistencyScore, GeographicConsistency

__all__ = [
    "ConfidenceCalculator",
    "ConfidenceScores",
    "ProcessingMethod",
    "PatternConfidence",
    "MLConfidence",
    "QualityAssessment",
    "QualityMetrics",
    "CompletenessScore",
    "ConsistencyScore",
    "GeographicConsistency",
]

# Version info
__version__ = "1.0.0"
__author__ = "Address Normalization Team"
