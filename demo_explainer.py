#!/usr/bin/env python3
"""
Demo script for Turkish explanation parsing.

Shows how the explainer module processes various types of spatial explanations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append("src")

from addrnorm.explainer import parse_explanation
from addrnorm.explainer.rules import load_test_explanations


def demo_basic_functionality():
    """Demonstrate basic parsing functionality."""
    print("🔍 Basic Explanation Parsing Demo")
    print("=" * 60)

    test_cases = [
        "Amorium Hotel karşısı",
        "marketin yanı",
        "parkın arkası",
        "hastane önü",
        "banka köşesi",
        "terminal girişi",
        "caminin karşısında",
        "çarşının içi",
        "otobüs durağı karşısı",
        "alışveriş merkezi yanında",
    ]

    successful = 0
    total = len(test_cases)

    for text in test_cases:
        result = parse_explanation(text)
        if result:
            print(f"✅ '{text}'")
            print(f"   Landmark: {result.name}")
            print(f"   Relation: {result.relation}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Type: {result.type}")
            successful += 1
        else:
            print(f"❌ '{text}' - Parse failed")
        print()

    accuracy = successful / total
    print(f"Success Rate: {successful}/{total} = {accuracy:.1%}")
    return accuracy


def demo_suffix_processing():
    """Demonstrate Turkish suffix processing."""
    print("\n🔧 Turkish Suffix Processing Demo")
    print("=" * 60)

    from addrnorm.explainer.rules import TurkishSuffixProcessor

    processor = TurkishSuffixProcessor()

    test_words = [
        ("marketin", "market"),
        ("okulun", "okul"),
        ("hastanesi", "hastane"),
        ("caminin", "cami"),
        ("çarşının", "çarşı"),
        ("köprünün", "köprü"),
        ("bankası", "banka"),
        ("oteli", "otel"),
    ]

    print("Original → Root Form")
    print("-" * 30)

    for original, expected in test_words:
        result = processor.strip_suffixes(original)
        status = "✅" if result == expected else "❌"
        print(f"{status} {original} → {result} (expected: {expected})")


def demo_spatial_relations():
    """Demonstrate spatial relation recognition."""
    print("\n🗺️  Spatial Relations Demo")
    print("=" * 60)

    relations_examples = {
        "across_from": ["karşısı", "karşında", "karşıda"],
        "next_to": ["yanı", "yanında", "kenarı"],
        "behind": ["arkası", "arkasında"],
        "in_front": ["önü", "önünde"],
        "above": ["üstü", "üstünde", "tepesi"],
        "below": ["altı", "altında"],
        "corner": ["köşesi", "köşesinde"],
        "entrance": ["girişi", "girişinde"],
        "inside": ["içi", "içinde"],
        "vicinity": ["civarı", "çevresi"],
    }

    for relation_type, turkish_forms in relations_examples.items():
        print(f"📍 {relation_type.upper().replace('_', ' ')}")
        for form in turkish_forms[:2]:  # Show first 2 examples
            test_text = f"market {form}"
            result = parse_explanation(test_text)
            if result:
                print(f"   '{test_text}' → {result.relation}")
            else:
                print(f"   '{test_text}' → ❌ Parse failed")
        print()


def demo_real_world_accuracy():
    """Test accuracy on real world examples."""
    print("\n📊 Real World Accuracy Test")
    print("=" * 60)

    # Load examples from file
    test_file = Path("data/examples/explanations.txt")

    if test_file.exists():
        explanations = load_test_explanations(str(test_file))
        valid_explanations = [
            line for line in explanations if line and not line.startswith("#")
        ]

        print(f"Testing {len(valid_explanations)} real examples...")
        print()

        successful = 0
        high_confidence = 0

        for i, explanation in enumerate(valid_explanations, 1):
            result = parse_explanation(explanation)
            if result:
                successful += 1
                if result.confidence >= 0.8:
                    high_confidence += 1

                print(f"{i:2d}. ✅ '{explanation}'")
                print(
                    f"     → {result.name} | {result.relation} | {result.confidence:.3f}"  # noqa: E501
                )
            else:
                print(f"{i:2d}. ❌ '{explanation}' - Parse failed")

        print("\n" + "=" * 60)
        accuracy = successful / len(valid_explanations)
        hc_rate = high_confidence / len(valid_explanations)

        print("📈 RESULTS:")
        print(f"   Total Examples: {len(valid_explanations)}")
        print(f"   Successful: {successful}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   High Confidence (≥0.8): {high_confidence} ({hc_rate:.1%})")
        print(f"   Required: ≥90% → {'✅ PASS' if accuracy >= 0.9 else '❌ FAIL'}")

        return accuracy
    else:
        print("❌ Test data file not found: data/examples/explanations.txt")
        return 0.0


def demo_edge_cases():
    """Test challenging edge cases."""
    print("\n🎯 Edge Cases & Challenging Examples")
    print("=" * 60)

    edge_cases = [
        "köprünün altı",  # Complex genitive + spatial
        "tünelin girişi",  # Genitive + entrance
        "otobüs durağı karşısı",  # Multi-word landmark
        "alışveriş merkezi yanında",  # Long compound
        "sağlık ocağı önünde",  # Medical facility
        "benzin istasyonu karşısında",  # Service station
        "polis karakolu arkasında",  # Government building
    ]

    successful = 0

    for case in edge_cases:
        result = parse_explanation(case)
        if result and result.confidence >= 0.5:
            print(f"✅ '{case}'")
            print(f"   → {result.name} | {result.relation} | {result.confidence:.3f}")
            successful += 1
        else:
            print(f"❌ '{case}' - Parse failed or low confidence")

    accuracy = successful / len(edge_cases)
    print(f"\nEdge Cases Accuracy: {successful}/{len(edge_cases)} = {accuracy:.1%}")

    return accuracy


def main():
    """Run complete explanation parsing demo."""
    print("🇹🇷 Turkish Explanation Parser - Complete Demo")
    print("=" * 80)
    print()

    # Run all demos
    basic_accuracy = demo_basic_functionality()
    demo_suffix_processing()
    demo_spatial_relations()
    real_accuracy = demo_real_world_accuracy()
    edge_accuracy = demo_edge_cases()

    print("\n" + "=" * 80)
    print("📋 FINAL SUMMARY")
    print("=" * 80)
    print(f"Basic Examples Accuracy: {basic_accuracy:.1%}")
    print(f"Real World Accuracy: {real_accuracy:.1%}")
    print(f"Edge Cases Accuracy: {edge_accuracy:.1%}")
    print()

    overall_pass = real_accuracy >= 0.9
    print("🎯 REQUIREMENT: ≥90% accuracy on test set")
    print(f"🏆 RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")

    if overall_pass:
        print("\n🎉 Explanation parsing system meets requirements!")
        print("   Ready for integration with address normalization pipeline.")
    else:
        print("\n⚠️  System needs improvement to meet 90% accuracy requirement.")


if __name__ == "__main__":
    main()
