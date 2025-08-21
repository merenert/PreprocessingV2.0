#!/usr/bin/env python3
"""
Demonstration script for the preprocess module.
"""
from src.addrnorm.preprocess import (
    expand_abbreviations,
    load_abbreviations,
    normalize_case,
    preprocess,
)


def demo_preprocess():
    """Demonstrate the preprocessing functionality."""
    print("=== Turkish Address Preprocessing Demo ===\n")

    # Test cases from real Turkish addresses
    test_cases = [
        "İSTANBUL, KADIKÖY, MODA MH., BAHARİYE CD. NO:12 D:5",
        "Ankara, Çankaya, Kızılay Mah., Atatürk Bulvarı No:100",
        "İzmir, Konak, Alsancak Mah., Kıbrıs Şehitleri Cad. No:45",
        "Bursa/Nilüfer-Görükle Mh. (Üniversite Cd.) N:1 Apt.Blok:A",
        "İstanbul Kadıköy Moda mh. Bahariye cd. n:12 d:5 apt.",
        "ADANA SEYHaN CemALPAŞA MAH. ZİYAPAŞA BLV. N:23",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"Example {i}:")
        print(f"  Original: {text}")

        result = preprocess(text)
        print(f"  Processed: {result['text']}")
        print(f"  Tokens: {result['tokens']}")
        print()

    print("=== Individual Function Demos ===\n")

    # Demo individual functions
    text = "İSTANBUL KADIKÖY MODA MH."

    print(f"Original: {text}")
    print(f"normalize_case: {normalize_case(text)}")

    abbrevs = load_abbreviations()
    print(f"expand_abbreviations: {expand_abbreviations(text.lower(), abbrevs)}")

    print("\n=== Abbreviation Dictionary Sample ===")
    print(f"Total abbreviations loaded: {len(abbrevs)}")

    # Show some key abbreviations
    key_abbrevs = ["mh", "cd", "no", "n:", "apt", "blv", "d:", "sk"]
    for abbrev in key_abbrevs:
        if abbrev in abbrevs:
            print(f"  {abbrev} -> {abbrevs[abbrev]}")

    print("\n=== Edge Cases ===")

    edge_cases = [
        "mh.",  # with period
        "n:123",  # attached number
        "D:5",  # uppercase with number
        "Mehmet",  # should NOT expand 'mh' inside name
        "test N: 42 test",  # with spaces
    ]

    for case in edge_cases:
        expanded = expand_abbreviations(case.lower(), abbrevs)
        print(f"  '{case}' -> '{expanded}'")

    print("\n=== Configuration Demo ===")

    # Demo with different configurations
    text = "İSTANBUL MH. CD."

    print(f"Original: {text}")
    print(f"Default processing: {preprocess(text)['text']}")
    print(f"No case norm: {preprocess(text, normalize_case=False)['text']}")
    print(f"No abbreviations: {preprocess(text, expand_abbreviations=False)['text']}")

    print("\n=== Idempotency Test ===")

    # Test idempotency
    text = "istanbul kadıköy moda mahalle"
    result1 = preprocess(text)
    result2 = preprocess(result1["text"])

    print(f"Original: {text}")
    print(f"First pass: {result1['text']}")
    print(f"Second pass: {result2['text']}")
    print(f"Idempotent: {result1['text'] == result2['text']}")


if __name__ == "__main__":
    demo_preprocess()
