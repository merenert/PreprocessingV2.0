#!/usr/bin/env python3
"""
Simple demo for the Turkish address explanation processing module.
Shows basic text cleaning functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from addrnorm.explanation import process_explanation, parse_explanation


def main():
    """Demo the simplified explanation processing."""
    print("🧹 Turkish Address Explanation Text Processing Demo")
    print("=" * 55)

    # Test cases with various formatting issues
    test_cases = [
        "Migros yanı",
        "  Amorium Hotel karşısı  ",
        "Şekerbank     ATM yanında",
        "   McDonald's önü   ",
        "\t\tHastane arkası\t",
        "Okul  \n  girişi",
        "",
        "   ",
        "İstanbul Üniversitesi yanındaki kafe",
    ]

    print("\n📝 Text Cleaning Examples:")
    print("-" * 30)

    for i, text in enumerate(test_cases, 1):
        cleaned = process_explanation(text)

        # Show original vs cleaned
        original_repr = repr(text)
        cleaned_repr = repr(cleaned)

        print(f"{i:2d}. Original: {original_repr}")
        print(f"    Cleaned:  {cleaned_repr}")

        if text != cleaned:
            print(f"    ✅ Modified")
        else:
            print(f"    ⚡ No change needed")
        print()

    print("\n🔄 Backward Compatibility Test:")
    print("-" * 35)

    # Test backward compatibility function
    test_text = "Garanti Bankası karşısında"
    result1 = process_explanation(test_text)
    result2 = parse_explanation(test_text)

    print(f"process_explanation(): {repr(result1)}")
    print(f"parse_explanation():   {repr(result2)}")
    print(f"Results match: {'✅ Yes' if result1 == result2 else '❌ No'}")

    print("\n💼 Integration Example:")
    print("-" * 25)

    # Show how it would be used in practice
    raw_explanations = ["Metro yanı", "  Hospital karşısı  ", "Bank ATM önünde", "School arkasında"]

    print("Processing batch of explanations:")
    for explanation in raw_explanations:
        cleaned = process_explanation(explanation)
        print(f"  '{explanation}' → '{cleaned}'")

    print(f"\n✨ Processed {len(raw_explanations)} explanations successfully!")

    print("\n📊 Summary:")
    print("-" * 12)
    print("✅ Simple text cleaning and validation")
    print("✅ Whitespace normalization")
    print("✅ Empty input handling")
    print("✅ Backward compatibility maintained")
    print("✅ Ready for explanation_raw field usage")


if __name__ == "__main__":
    main()
