#!/usr/bin/env python3
"""
Migration helper for transitioning from complex explanation parsing to simple text processing.

The explanation module has been simplified to focus on basic text cleaning
rather than complex landmark detection and spatial relation parsing.
"""

import warnings
from typing import Optional, Dict, Any


def migration_guide():
    """Print migration guide for users."""
    guide = """
    📋 MIGRATION GUIDE: Complex Parser → Simple Text Processing
    =========================================================

    OLD CODE (Complex explanation parser):
    -------------------------------------
    from addrnorm.explanation import ExplanationParser

    parser = ExplanationParser()
    result = parser.parse("Migros yanı")
    print(result.landmark.name)  # "Migros"
    print(result.relation.relation)  # "yanı"
    print(result.confidence)  # 0.92


    NEW CODE (Simple text processing):
    ---------------------------------
    from addrnorm.explanation import process_explanation

    cleaned = process_explanation("Migros yanı")
    print(cleaned)  # "Migros yanı"

    # For backward compatibility:
    from addrnorm.explanation import parse_explanation
    cleaned = parse_explanation("Migros yanı")  # Same as process_explanation


    🎯 KEY CHANGES:
    --------------
    ❌ REMOVED: Landmark detection and classification
    ❌ REMOVED: Spatial relation extraction
    ❌ REMOVED: Confidence scoring and complex analysis
    ❌ REMOVED: Fuzzy matching and compound relations
    ❌ REMOVED: Performance caching and batch processing

    ✅ SIMPLIFIED: Basic text cleaning and validation
    ✅ FOCUSED: explanation_raw field processing only
    ✅ MAINTAINED: Backward compatibility with parse_explanation()


    💡 RATIONALE:
    ------------
    The explanation module was simplified to focus on the core requirement:
    cleaning and validating text for the explanation_raw field. Complex
    landmark parsing was deemed unnecessary for the primary use case.


    � INTEGRATION EXAMPLE:
    ----------------------
    # Use with main address normalization
    from addrnorm import AddressNormalizer
    from addrnorm.explanation import process_explanation

    normalizer = AddressNormalizer()

    def normalize_with_explanation(address_text):
        # Clean explanation
        explanation_cleaned = process_explanation(address_text)

        # Normalize address
        result = normalizer.normalize(address_text)

        # Add cleaned explanation to result
        if isinstance(result, dict):
            result["explanation_raw"] = explanation_cleaned

        return result


    📚 RESOURCES:
    ------------
    - Simplified module docs: docs/explanation_parser.md
    - Simple tests: tests/test_explanation_simple.py
    - Main output format: src/addrnorm/utils/contracts.py
    """
    print(guide)


# Legacy compatibility warning
def complex_parser_warning():
    """Issue warning about complex parser removal."""
    warnings.warn(
        "Complex explanation parsing features have been removed. "
        "The module now provides simple text processing only. "
        "See migration guide for details.",
        DeprecationWarning,
        stacklevel=3,
    )


if __name__ == "__main__":
    migration_guide()
