#!/usr/bin/env python3
"""
Debug script for preprocess functionality.
"""
from src.addrnorm.preprocess import (
    clean_punctuation,
    expand_abbreviations,
    load_abbreviations,
    normalize_case,
    normalize_unicode,
    tokenize,
)


def debug_preprocess():
    text = "İstanbul/Kadıköy-Moda Mh.(Bahariye Cd.) N:12/5 Apt.Blok:A"
    print(f"Original: {text}")

    # Step by step
    step1 = normalize_case(text)
    print(f"After normalize_case: {step1}")

    step2 = normalize_unicode(step1)
    print(f"After normalize_unicode: {step2}")

    abbrevs = load_abbreviations()
    step3 = expand_abbreviations(step2, abbrevs)
    print(f"After expand_abbreviations: {step3}")

    step4 = clean_punctuation(step3)
    print(f"After clean_punctuation: {step4}")

    tokens = tokenize(step4)
    print(f"Tokens: {tokens}")

    # Test specific case
    print("\nTesting 'N:12' expansion:")
    test_lower = "n:12"
    test_expanded = expand_abbreviations(test_lower, abbrevs)
    print(f"'{test_lower}' -> '{test_expanded}'")


if __name__ == "__main__":
    debug_preprocess()
