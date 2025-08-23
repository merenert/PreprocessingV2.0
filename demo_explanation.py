#!/usr/bin/env python3
"""
Usage examples for the Turkish address explanation parser.
Demonstrates various ways to use the landmark detection and spatial relation extraction.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from addrnorm.explanation import (
    parse_explanation,
    extract_landmark_info,
    ExplanationParser,
    ExplanationConfig,
)


def basic_usage_examples():
    """Demonstrate basic usage of the explanation parser."""
    print("🏷️  BASIC USAGE EXAMPLES")
    print("=" * 50)

    # Simple cases
    test_cases = [
        "Migros yanı",
        "Amorium Hotel karşısı",
        "Şekerbank şubesi önünde",
        "McDonalds arkasında",
        "Teknosa mağazası bitişiği",
        "İş Bankası ATM'si yakınında",
    ]

    for text in test_cases:
        print(f"\n📍 Input: '{text}'")
        result = parse_explanation(text)

        print(f"   🏢 Landmark: {result.get('landmark_name', 'None')}")
        print(f"   📋 Type: {result.get('landmark_type', 'None')}")
        print(f"   📍 Relation: {result.get('spatial_relation', 'None')}")
        print(f"   ⭐ Confidence: {result.get('confidence', 0):.2f}")


def advanced_usage_examples():
    """Demonstrate advanced parser usage with configuration."""
    print("\n\n🔧 ADVANCED USAGE EXAMPLES")
    print("=" * 50)

    # Create parser with custom configuration
    config = ExplanationConfig(
        min_confidence_threshold=0.4, debug_mode=True, enable_fuzzy_matching=True
    )
    parser = ExplanationParser(config)

    complex_cases = [
        "Koç Holding binası arkasındaki küçük dükkan",
        "Mall of İstanbul AVM içindeki Starbucks yanı",
        "Beylikdüzü Devlet Hastanesi acil servisi karşısı",
        "Boğaziçi Üniversitesi ana kampüsü çevresinde",
    ]

    for text in complex_cases:
        print(f"\n📍 Complex Input: '{text}'")
        result = parser.parse(text)

        print(f"   📊 Overall Confidence: {result.confidence:.2f}")

        if result.landmark:
            print(f"   🏢 Landmark: {result.landmark.name}")
            print(f"   📋 Type: {result.landmark.type}")
            print(f"   ⭐ L.Confidence: {result.landmark.confidence:.2f}")

        if result.relation:
            print(f"   📍 Relation: {result.relation.relation}")
            print(f"   ⭐ R.Confidence: {result.relation.confidence:.2f}")

        if result.processing_notes:
            print(f"   📝 Notes: {', '.join(result.processing_notes)}")


def batch_processing_example():
    """Demonstrate batch processing of multiple explanations."""
    print("\n\n📦 BATCH PROCESSING EXAMPLE")
    print("=" * 50)

    explanations = [
        "Migros yanı",
        "Hastane karşısı",
        "Okul arkasında",
        "belirsiz bir yer",  # Low confidence case
        "123 numara",  # No landmark case
        "Hotel Grand Ankara önünde",
        "Ziraat Bankası şubesi bitişiği",
    ]

    parser = ExplanationParser()
    results = parser.parse_batch(explanations)

    print(f"Processed {len(explanations)} explanations:")

    for i, (text, result) in enumerate(zip(explanations, results), 1):
        status = (
            "✅"
            if result.confidence > 0.5
            else "⚠️" if result.confidence > 0.3 else "❌"
        )
        print(f"{status} {i:2d}. '{text}' → confidence: {result.confidence:.2f}")


def json_output_example():
    """Demonstrate JSON output format."""
    print("\n\n📄 JSON OUTPUT EXAMPLE")
    print("=" * 50)

    parser = ExplanationParser()
    text = "Migros market yanında bulunan eczane"
    result = parser.parse(text)

    # Get JSON output
    json_output = result.to_json_output()

    print(f"Input: '{text}'")
    print("JSON Output:")
    print(json.dumps(json_output, indent=2, ensure_ascii=False))


def debugging_example():
    """Demonstrate debugging capabilities."""
    print("\n\n🔍 DEBUGGING EXAMPLE")
    print("=" * 50)

    parser = ExplanationParser()
    text = "Koç Holding A.Ş. binası karşısındaki Starbucks"

    # Get debug information
    debug_info = parser.get_debug_info(text)

    print(f"Input: '{text}'")
    print(f"Preprocessed: '{debug_info['preprocessed_text']}'")

    print("\nDetected Landmarks:")
    for landmark in debug_info["detected_landmarks"]:
        print(
            f"  - {landmark['name']} ({landmark['type']}) - {landmark['confidence']:.2f}"
        )

    print("\nDetected Relations:")
    for relation in debug_info["detected_relations"]:
        print(f"  - {relation['relation']} - {relation['confidence']:.2f}")


def convenience_functions_example():
    """Demonstrate convenience functions."""
    print("\n\n⚡ CONVENIENCE FUNCTIONS")
    print("=" * 50)

    # Quick landmark extraction
    landmark_info = extract_landmark_info("Şekerbank ATM yanı")
    if landmark_info:
        print(f"Quick landmark extraction: {landmark_info}")

    # Simple parsing
    simple_result = parse_explanation("Hotel karşısı", debug=False)
    print(f"Simple parsing result: {simple_result}")


def error_handling_example():
    """Demonstrate error handling."""
    print("\n\n⚠️  ERROR HANDLING EXAMPLE")
    print("=" * 50)

    parser = ExplanationParser()

    problematic_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "xyz 123 abc",  # Nonsense text
        "çok belirsiz açıklama",  # Very vague
    ]

    for text in problematic_cases:
        try:
            result = parser.parse(text)
            print(
                f"'{text}' → confidence: {result.confidence:.2f}, notes: {result.processing_notes}"
            )
        except Exception as e:
            print(f"'{text}' → ERROR: {e}")


def performance_example():
    """Demonstrate performance characteristics."""
    print("\n\n⚡ PERFORMANCE EXAMPLE")
    print("=" * 50)

    import time

    parser = ExplanationParser()

    # Test parsing speed
    test_text = "Migros yanı"
    iterations = 1000

    start_time = time.time()
    for _ in range(iterations):
        result = parser.parse(test_text)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000  # Convert to ms

    print(f"Parsed '{test_text}' {iterations} times")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per parse: {avg_time:.2f}ms")
    print(f"Throughput: {iterations/total_time:.0f} parses/second")


def real_world_examples():
    """Test with real-world Turkish address explanations."""
    print("\n\n🌍 REAL-WORLD EXAMPLES")
    print("=" * 50)

    real_examples = [
        "Acıbadem Hastanesi ana giriş karşısı",
        "Migros Jet yanındaki eczane",
        "İstanbul Teknik Üniversitesi rektörlük binası arkası",
        "Kozyatağı Metro İstasyonu çıkışı",
        "Carrefour SA hipermarket otopark girişi",
        "Maltepe Belediyesi hizmet binası önü",
        "Pendik Marina AVM food court üstü",
        "Anadolu Adalet Sarayı yan binası",
    ]

    parser = ExplanationParser()

    for text in real_examples:
        result = parser.parse(text)
        confidence_emoji = (
            "🟢"
            if result.confidence > 0.7
            else "🟡" if result.confidence > 0.4 else "🔴"
        )

        print(f"\n{confidence_emoji} '{text}'")
        if result.landmark:
            print(f"   🏢 {result.landmark.name} ({result.landmark.type})")
        if result.relation:
            print(f"   📍 {result.relation.relation}")
        print(f"   📊 {result.confidence:.2f}")


def main():
    """Run all examples."""
    print("🇹🇷 TURKISH ADDRESS EXPLANATION PARSER")
    print("📍 Landmark Detection & Spatial Relation Extraction")
    print("=" * 80)

    try:
        basic_usage_examples()
        advanced_usage_examples()
        batch_processing_example()
        json_output_example()
        debugging_example()
        convenience_functions_example()
        error_handling_example()
        performance_example()
        real_world_examples()

        print("\n\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
