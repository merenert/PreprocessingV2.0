#!/usr/bin/env python3
"""
Test script for geographic validation system.
Tests fuzzy matching and consistency checks with intentional typos.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from addrnorm.validate import create_geo_validator


def test_city_validation():
    """Test city validation with fuzzy matching."""

    print("🏙️ City Validation Tests")
    print("=" * 40)

    validator = create_geo_validator()

    # Test cases with intentional typos
    test_cases = [
        ("istanbul", "Exact match"),
        ("İstanbul", "Case and accent test"),
        ("istanbull", "Double L typo"),
        ("ankarra", "Double R typo"),
        ("izmır", "Turkish character test"),
        ("antalyaa", "Extra A typo"),
        ("burssa", "Double S typo"),
        ("manisaa", "Extra A typo"),
        ("xyz123", "Invalid city"),
        ("", "Empty input"),
    ]

    for city, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: '{city}'")

        result = validator.validate_city(city)

        print(f"  Valid: {result.is_valid}")
        print(f"  Standardized: {result.standardized_value}")
        print(f"  Confidence: {result.confidence:.3f}")

        if result.suggestions:
            print(f"  Suggestions: {result.suggestions[:3]}")

        if result.warnings:
            print(f"  Warnings: {result.warnings}")


def test_district_validation():
    """Test district validation."""

    print("\n🏘️ District Validation Tests")
    print("=" * 40)

    validator = create_geo_validator()

    test_cases = [
        ("kadikoy", "istanbul", "Exact match with city"),
        ("kadıköy", "istanbul", "Turkish chars with city"),
        ("besiktas", "istanbul", "Missing accent"),
        ("beşiktaş", None, "Without city context"),
        ("konaak", "izmir", "Typo in district"),
        ("bucaa", "izmir", "Extra A typo"),
        ("invalid", "istanbul", "Invalid district"),
        ("", "ankara", "Empty district"),
    ]

    for district, city, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: district='{district}', city='{city}'")

        result = validator.validate_district(district, city)

        print(f"  Valid: {result.is_valid}")
        print(f"  Standardized: {result.standardized_value}")
        print(f"  Confidence: {result.confidence:.3f}")

        if result.suggestions:
            print(f"  Suggestions: {result.suggestions[:3]}")

        if result.warnings:
            print(f"  Warnings: {result.warnings}")


def test_consistency_validation():
    """Test city-district consistency."""

    print("\n🔗 City-District Consistency Tests")
    print("=" * 40)

    validator = create_geo_validator()

    test_cases = [
        ("istanbul", "kadikoy", "Correct combination"),
        ("istanbul", "konak", "Wrong district for city"),
        ("izmir", "besiktas", "Wrong district for city"),
        ("ankara", "cankaya", "Correct combination"),
        ("bursa", "buca", "Wrong district for city"),
        ("antalya", "kepez", "Correct combination"),
    ]

    for city, district, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: city='{city}', district='{district}'")

        result = validator.validate_city_district_consistency(city, district)

        print(f"  Consistent: {result.is_consistent}")
        print(f"  Std City: {result.standardized_city}")
        print(f"  Std District: {result.standardized_district}")

        if result.suggestions:
            print(f"  Suggestions: {result.suggestions[:3]}")

        if result.warnings:
            print(f"  Warnings: {result.warnings}")


def test_postcode_validation():
    """Test postal code validation."""

    print("\n📮 Postal Code Validation Tests")
    print("=" * 40)

    validator = create_geo_validator()

    test_cases = [
        ("34000", "Valid Istanbul code"),
        ("06000", "Valid Ankara code"),
        ("35000", "Valid Izmir code"),
        ("12345", "Valid format"),
        ("99999", "Out of range (too high)"),
        ("00500", "Out of range (too low)"),
        ("1234", "Too short"),
        ("123456", "Too long"),
        ("12abc", "Invalid characters"),
        ("", "Empty input"),
    ]

    for postcode, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: '{postcode}'")

        result = validator.validate_postcode(postcode)

        print(f"  Valid: {result.is_valid}")
        print(f"  Standardized: {result.standardized_value}")
        print(f"  Confidence: {result.confidence:.3f}")

        if result.warnings:
            print(f"  Warnings: {result.warnings}")


def test_fuzzy_matching():
    """Test specific fuzzy matching for 5 intentional typos."""

    print("\n🎯 Fuzzy Matching - 5 Intentional Typos Test")
    print("=" * 50)

    validator = create_geo_validator()

    # 5 intentional typos that should be corrected
    typo_tests = [
        ("istanbull", "istanbul", "City: Double L"),
        ("ankarra", "ankara", "City: Double R"),
        ("kadıkoyy", "kadikoy", "District: Double Y"),
        ("besiktass", "besiktas", "District: Double S"),
        ("izmirr", "izmir", "City: Double R"),
    ]

    corrected_count = 0

    for typo, expected, description in typo_tests:
        print(f"\n{description}")
        print(f"Input: '{typo}' → Expected: '{expected}'")

        if "City" in description:
            result = validator.validate_city(typo)
        else:
            result = validator.validate_district(
                typo, "istanbul"
            )  # Use Istanbul as context

        if result.standardized_value == expected:
            print(f"  ✅ CORRECTED: {result.standardized_value}")
            corrected_count += 1
        elif result.suggestions and expected in result.suggestions:
            print(f"  📝 SUGGESTED: {result.suggestions}")
            corrected_count += 1
        else:
            print(f"  ❌ FAILED: {result.standardized_value}")
            print(f"     Suggestions: {result.suggestions}")

        print(f"  Confidence: {result.confidence:.3f}")

    print(f"\n📊 Fuzzy Matching Results: {corrected_count}/5 typos handled")

    if corrected_count >= 5:
        print("✅ ACCEPTANCE CRITERIA MET: All 5 typos corrected!")
    else:
        print(f"⚠️ Only {corrected_count}/5 typos handled")


def test_system_stats():
    """Show system statistics."""

    print("\n📊 System Statistics")
    print("=" * 30)

    validator = create_geo_validator()
    stats = validator.get_stats()

    for key, value in stats.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
            for item in value[:3]:
                print(f"  - {item}")
            if len(value) > 3:
                print(f"  ... and {len(value) - 3} more")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    try:
        print("🔍 TURKISH GEOGRAPHIC VALIDATION SYSTEM TEST")
        print("=" * 60)

        test_city_validation()
        test_district_validation()
        test_consistency_validation()
        test_postcode_validation()
        test_fuzzy_matching()
        test_system_stats()

        print("\n🎉 ALL TESTS COMPLETED!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
