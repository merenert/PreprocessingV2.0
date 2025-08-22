#!/usr/bin/env python3
"""
Demo script for fallback normalization system.
Tests rule-based normalization on various Turkish address examples.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

from addrnorm.fallback import create_fallback_normalizer, normalize_address_fallback


def test_fallback_normalization():
    """Test fallback normalization with various address examples."""

    print("🔧 Turkish Address Fallback Normalization Demo")
    print("=" * 60)

    # Test cases with varying complexity
    test_cases = [
        {
            "raw": "atatürk mahallesi cumhuriyet caddesi no 15 kat 3 daire 5 ankara",
            "existing": {},
            "description": "Complete address with all components",
        },
        {
            "raw": "a blok kat 2 daire 12 num:45",
            "existing": {},
            "description": "Partial address - building details only",
        },
        {
            "raw": "konak mahallesi 234 sokak izmir",
            "existing": {"neighborhood": "konak mahallesi"},
            "description": "With some existing components",
        },
        {
            "raw": "no 7 istanbul",
            "existing": {},
            "description": "Minimal address information",
        },
        {
            "raw": "buca merkez 45 numara",
            "existing": {},
            "description": "District and number only",
        },
        {
            "raw": "barbaros bulvarı no 123 beşiktaş",
            "existing": {},
            "description": "Boulevard address",
        },
        {
            "raw": "random text without address info",
            "existing": {},
            "description": "Non-address text (edge case)",
        },
        {
            "raw": "1234 sk no 56 k3 d8",
            "existing": {},
            "description": "Abbreviated format",
        },
    ]

    normalizer = create_fallback_normalizer()

    # Show normalizer stats
    stats = normalizer.get_stats()
    print("📊 Normalizer Stats:")
    print(f"   Total Rules: {stats['pattern_stats']['total_rules']}")
    print(f"   City Keywords: {stats['pattern_stats']['city_keywords']}")
    print(f"   Confidence Range: {stats['confidence_range']}")
    print()

    for i, test_case in enumerate(test_cases, 1):
        print(f"【 Test {i} 】- {test_case['description']}")
        print(f"Raw: {test_case['raw']}")
        print(f"Existing: {test_case['existing']}")
        print("-" * 50)

        try:
            # Normalize using fallback
            result = normalize_address_fallback(
                address_dict=test_case["existing"], raw_input=test_case["raw"]
            )

            # Display results
            print(f"✅ Normalized Address: {result.normalized_address}")
            print(f"📊 Confidence: {result.explanation_parsed.confidence:.3f}")
            print(f"🔧 Method: {result.explanation_parsed.method}")

            # Show extracted components
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
                "postcode",
            ]:
                value = getattr(result, field, None)
                if value:
                    components[field] = value

            if components:
                print("🎯 Extracted Components:")
                for field, value in components.items():
                    print(f"   {field}: {value}")
            else:
                print("❌ No components extracted")

            # Show warnings
            if result.explanation_parsed.warnings:
                print("⚠️ Warnings:")
                for warning in result.explanation_parsed.warnings:
                    print(f"   - {warning}")

            # JSON output for detailed inspection
            print("\n📄 JSON Output:")
            print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 60)
        input("Press Enter to continue...")


def test_rule_extraction():
    """Test individual rule extraction without full normalization."""

    print("\n🔍 Rule Extraction Demo")
    print("=" * 40)

    from addrnorm.fallback.rules import TurkishAddressRules

    rules = TurkishAddressRules()

    test_texts = [
        "no 12 kat 3 daire 5",
        "a blok num:45",
        "123 sokak numara 67",
        "atatürk bulvarı 234 sk",
        "d8 k2 ankara",
    ]

    for text in test_texts:
        print(f"\nText: '{text}'")

        # Test pattern normalization
        result = rules.normalize_patterns(text)
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Components: {result['components']}")

        if result["applied_rules"]:
            print("Applied Rules:")
            for rule in result["applied_rules"]:
                print(
                    f"  - {rule['field']}: '{rule['matched']}' → '{rule['extracted']}'"
                )

        # Test heuristics
        heuristics = rules.apply_heuristics(text, result["components"])
        if heuristics["components"]:
            print(f"Heuristics: {heuristics['components']}")


if __name__ == "__main__":
    try:
        test_fallback_normalization()
        test_rule_extraction()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
