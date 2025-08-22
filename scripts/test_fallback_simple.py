#!/usr/bin/env python3
"""
Simple fallback test to verify the system works.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from addrnorm.fallback import normalize_address_fallback


def quick_test():
    """Quick test of fallback normalization."""

    test_cases = [
        {
            "raw": "atatürk mahallesi cumhuriyet caddesi no 15 kat 3 daire 5 ankara",
            "existing": {},
        },
        {"raw": "a blok kat 2 daire 12 num:45", "existing": {}},
        {"raw": "konak mahallesi 234 sokak izmir", "existing": {}},
    ]

    print("🔧 Fallback Normalization Quick Test")
    print("=" * 50)

    for i, test in enumerate(test_cases, 1):
        print(f"\n【 Test {i} 】")
        print(f"Input: {test['raw']}")

        try:
            result = normalize_address_fallback(test["existing"], test["raw"])

            print("✅ Success!")
            print(f"   Normalized: {result.normalized_address}")
            print(f"   Confidence: {result.explanation_parsed.confidence:.3f}")
            print(f"   Method: {result.explanation_parsed.method}")

            # Show components
            components = []
            for field in [
                "city",
                "district",
                "neighborhood",
                "street",
                "number",
                "floor",
                "apartment",
            ]:
                value = getattr(result, field, None)
                if value:
                    components.append(f"{field}={value}")

            if components:
                print(f"   Components: {', '.join(components)}")

            if result.explanation_parsed.warnings:
                print(
                    f"   Warnings: {len(result.explanation_parsed.warnings)} warning(s)"
                )

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    quick_test()
