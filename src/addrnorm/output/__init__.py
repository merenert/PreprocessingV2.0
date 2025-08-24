"""
Enhanced Output Module

Advanced output formatting and processing for address normalization results.

This module provides:
- Multi-format output support (JSON, CSV, XML, YAML)
- Confidence scoring integration
- Quality metrics assessment
- Backward compatibility with legacy formats
- Schema validation and documentation
- Batch processing capabilities
"""

from .enhanced_formatter import (
    EnhancedFormatter,
    OutputFormat,
    ValidationStatus,
    ExplanationDetails,
    EnhancedNormalizationResult,
    create_enhanced_formatter,
    format_address_result,
)

__all__ = [
    "EnhancedFormatter",
    "OutputFormat",
    "ValidationStatus",
    "ExplanationDetails",
    "EnhancedNormalizationResult",
    "create_enhanced_formatter",
    "format_address_result",
]

# Version info
__version__ = "1.0.0"
__author__ = "Address Normalization Team"
