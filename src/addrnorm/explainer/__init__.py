"""
Explanation parsing module for Turkish address explanations.

⚠️  DEPRECATION NOTICE ⚠️
This module is deprecated and will be removed in a future version.
Please migrate to the new 'explanation' module for enhanced functionality.

Migration Guide:
    OLD: from addrnorm.explainer import parse_explanation
    NEW: from addrnorm.explanation import parse_explanation

The new module provides:
- ✅ Fuzzy matching for better landmark detection
- ✅ Compound spatial relations (e.g., "tam karşısında")
- ✅ Performance optimization (6.30ms vs 15ms)
- ✅ Better confidence scoring
- ✅ Batch processing capabilities
- ✅ Type-safe Pydantic models

For compatibility layer: from addrnorm.migration import ExplainerCompatibility

This module provides functionality to parse Turkish explanations like:
- "Amorium Hotel karşısı" → {"type": "landmark", "name": "amorium hotel",
                              "relation": "karşısı"}
- "marketin yanı" → {"type": "landmark", "name": "market", "relation": "yanı"}
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "The 'explainer' module is deprecated. Please migrate to 'explanation' module. "
    "See migration guide in src/addrnorm/migration.py",
    DeprecationWarning,
    stacklevel=2,
)

from .rules import ExplanationResult, parse_explanation

__all__ = ["parse_explanation", "ExplanationResult"]
