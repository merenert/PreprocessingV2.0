#!/usr/bin/env python3
"""
Comprehensive fallback system test with detailed output.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from addrnorm.fallback import create_fallback_normalizer
from addrnorm.fallback.rules import TurkishAddressRules


def test_rule_extraction():
    """Test individual rule extraction."""

    print("🔍 Rule Extraction Test")
    print("=" * 40)

    rules = TurkishAddressRules()

    test_patterns = [
        "no 12 kat 3 daire 5",
        "numara 45 a blok",
        "123 sokak 67 nolu",
        "atatürk bulvarı",
        "d8 k2 num:15",
    ]

    for pattern in test_patterns:
        print(f"\nPattern: '{pattern}'")
        result = rules.normalize_patterns(pattern)

        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Components: {result['components']}")

        if result["applied_rules"]:
            print("  Applied Rules:")
            for rule in result["applied_rules"][:3]:  # Show first 3
                print(
                    f"    - {rule['field']}: '{rule['matched']}' → "
                    f"'{rule['extracted']}'"
                )


def test_heuristics():
    """Test heuristic assignment."""

    print("\n🎯 Heuristic Assignment Test")
    print("=" * 40)

    rules = TurkishAddressRules()

    test_texts = [
        "ankara merkez",
        "buca mahallesi random text",
        "istanbul fatih district",
        "standalone number 42",
    ]

    for text in test_texts:
        print(f"\nText: '{text}'")
        heuristics = rules.apply_heuristics(text)

        print(f"  Components: {heuristics['components']}")
        print(f"  Confidence: {heuristics['confidence']:.3f}")
        if heuristics["heuristics_applied"]:
            for h in heuristics["heuristics_applied"]:
                print(f"    - {h}")


def test_complete_normalization():
    """Test complete normalization process."""

    print("\n🔧 Complete Normalization Test")
    print("=" * 50)

    normalizer = create_fallback_normalizer()

    # Edge cases and challenging examples
    test_cases = [
        {"name": "Minimal info", "raw": "no 7", "existing": {}},
        {
            "name": "Abbreviated format",
            "raw": "1234 sk no 56 k3 d8 a blok",
            "existing": {},
        },
        {
            "name": "Mixed Turkish-English",
            "raw": "apartment 5 floor 2 istanbul",
            "existing": {},
        },
        {
            "name": "Existing data priority",
            "raw": "no 15 kat 3",
            "existing": {"city": "Ankara", "district": "Çankaya"},
        },
        {
            "name": "Non-address text",
            "raw": "this is not an address at all",
            "existing": {},
        },
    ]

    for test in test_cases:
        print(f"\n【 {test['name']} 】")
        print(f"Raw: {test['raw']}")
        print(f"Existing: {test['existing']}")
        print("-" * 30)

        try:
            result = normalizer.normalize(test["existing"], test["raw"])

            print(f"✅ Normalized: {result.normalized_address}")
            print(f"📊 Confidence: {result.explanation_parsed.confidence:.3f}")

            # Extract non-null components
            components = {}
            for field in [
                "city",
                "district",
                "neighborhood",
                "street",
                "building",
                "block",
                "number",
                "entrance",
                "floor",
                "apartment",
            ]:
                value = getattr(result, field, None)
                if value:
                    components[field] = value

            if components:
                print("🎯 Components:")
                for field, value in components.items():
                    print(f"   {field}: {value}")
            else:
                print("❌ No components extracted")

            print(f"⚠️ Warnings: {len(result.explanation_parsed.warnings)}")
            for warning in result.explanation_parsed.warnings[:2]:  # Show first 2
                print(f"   - {warning}")

        except Exception as e:
            print(f"❌ Error: {e}")


def test_stats():
    """Show system statistics."""

    print("\n📊 System Statistics")
    print("=" * 30)

    normalizer = create_fallback_normalizer()
    stats = normalizer.get_stats()

    print(f"Normalizer Type: {stats['normalizer_type']}")
    print(f"Rule Engine: {stats['rule_engine']}")
    print(f"Confidence Range: {stats['confidence_range']}")
    print(f"Method: {stats['method']}")

    print("\nPattern Statistics:")
    for key, value in stats["pattern_stats"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    try:
        test_rule_extraction()
        test_heuristics()
        test_complete_normalization()
        test_stats()
        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
