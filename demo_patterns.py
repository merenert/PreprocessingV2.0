#!/usr/bin/env python3
"""
Demo script for Turkish address pattern matching.

Shows the complete pipeline: preprocessing → pattern matching → slot extraction
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from addrnorm.patterns import PatternMatcher
from addrnorm.preprocess import preprocess
from addrnorm.utils.contracts import AddressOut, ExplanationParsed, MethodEnum


def demo_pattern_matching():
    """Demonstrate pattern matching with real addresses."""
    print("=" * 80)
    print("🎯 TURKISH ADDRESS PATTERN MATCHING DEMO")
    print("=" * 80)

    # Initialize matcher
    print("🔧 Initializing pattern matcher...")
    matcher = PatternMatcher()

    # Show loaded patterns
    stats = matcher.get_pattern_stats()
    print(f"📋 Loaded {stats['total_patterns']} patterns")
    print(f"📚 Keywords: {stats['keywords']}")
    print()

    # Real addresses from train_sample.csv
    test_addresses = [
        "Akarca Mah. Adnan Menderes Cad. 864.Sok. No:15 D.1 K.2 .",
        "Cumhuriye Mah. Hükümet Cad. Sivriler İşhanı No:3 Fethiye/Muğla",
        "İsmet inönü mahallesi 2001 sokak no:2 Çeşme belediyesi Çeşme",
        "Bitez mahallesi Adnan Menderes caddesi gündonumu mevkii 1410.sokak no 90A",
        "Dedebaşı mahallesi 6100 sokak no 10 Kat 7 daire 25",
        "Yeni sanayi mahallesi 515 sokak no 53 c3 blok emek aliminyum",
        "Dikili Güzelbelde Sitesi Müsellim Altı No:51 Kabakum Dikili / İzmir",
        "Şemikler Mh. Osmangazi Cad. No:45 Gölkent Konutları A Blok No:5",
    ]

    print("🏠 TESTING WITH REAL ADDRESSES")
    print("-" * 80)

    successful_matches = 0
    total_addresses = len(test_addresses)

    for i, address in enumerate(test_addresses, 1):
        print(f"\n📍 Address {i}/{total_addresses}")
        print(f"   Original: {address}")

        # Step 1: Preprocess
        preprocessed = preprocess(address)
        preprocessed_text = preprocessed["text"]
        print(f"   Preprocessed: {preprocessed_text}")

        # Step 2: Pattern matching
        matches = matcher.match_text(preprocessed_text, max_matches=2)

        if matches:
            successful_matches += 1
            best_match = matches[0]

            print(
                f"   ✅ Best Match: {best_match.pattern_id} (conf: {best_match.confidence:.3f})"  # noqa: E501
            )
            print("   📦 Extracted Slots:")

            for slot_name, slot_value in best_match.slots.items():
                print(f"      • {slot_name}: '{slot_value}'")

            # Show alternative matches if any
            if len(matches) > 1:
                print(
                    f"   🔄 Alternative: {matches[1].pattern_id} (conf: {matches[1].confidence:.3f})"  # noqa: E501
                )

            # Create structured output
            explanation = ExplanationParsed(
                confidence=best_match.confidence, method=MethodEnum.PATTERN, warnings=[]
            )

            # Map slots to structured address fields
            structured_address = map_slots_to_address(
                best_match.slots, address, explanation
            )
            print(f"   🏗️  Structured: {structured_address.normalized_address}")

        else:
            print("   ❌ No pattern matches found")

        print("   " + "─" * 60)

    # Summary
    success_rate = successful_matches / total_addresses
    print("\n📊 SUMMARY")
    print(
        f"   Successfully matched: {successful_matches}/{total_addresses} ({success_rate:.1%})"  # noqa: E501
    )
    print(
        f"   Success rate: {'🎉 Excellent' if success_rate >= 0.8 else '✅ Good' if success_rate >= 0.6 else '⚠️  Needs improvement'}"  # noqa: E501
    )


def map_slots_to_address(
    slots: dict, original: str, explanation: ExplanationParsed
) -> AddressOut:
    """Map extracted slots to structured address output."""
    # Initialize all fields as None
    address_fields = {
        "country": None,
        "city": None,
        "district": None,
        "neighborhood": None,
        "street": None,
        "building": None,
        "block": None,
        "number": None,
        "entrance": None,
        "floor": None,
        "apartment": None,
        "postcode": None,
        "relation": None,
    }

    # Map slots to fields
    for slot_name, slot_value in slots.items():
        if slot_name.lower() in ["mahalle", "neighborhood"]:
            address_fields["neighborhood"] = slot_value
        elif slot_name.lower() in ["sokak", "street"]:
            address_fields["street"] = slot_value
        elif slot_name.lower() in ["cadde", "cad"]:
            address_fields["street"] = slot_value + " Caddesi"
        elif slot_name.lower() in ["n", "no", "numara", "number"]:
            address_fields["number"] = slot_value
        elif slot_name.lower() in ["daire", "apartment"]:
            address_fields["apartment"] = slot_value
        elif slot_name.lower() in ["kat", "floor"]:
            address_fields["floor"] = slot_value
        elif slot_name.lower() in ["blok", "block"]:
            address_fields["block"] = slot_value
        elif slot_name.lower() in ["site", "building"]:
            address_fields["building"] = slot_value

    # Create normalized address string
    parts = []
    if address_fields["neighborhood"]:
        parts.append(address_fields["neighborhood"] + " Mahallesi")
    if address_fields["street"]:
        parts.append(address_fields["street"])
    if address_fields["building"]:
        parts.append(address_fields["building"])
    if address_fields["block"]:
        parts.append(address_fields["block"] + " Blok")
    if address_fields["number"]:
        parts.append("No:" + address_fields["number"])
    if address_fields["floor"]:
        parts.append("Kat:" + address_fields["floor"])
    if address_fields["apartment"]:
        parts.append("Daire:" + address_fields["apartment"])

    normalized = ", ".join(parts) if parts else original

    return AddressOut(
        explanation_raw=original,
        explanation_parsed=explanation,
        normalized_address=normalized,
        **address_fields,
    )


def demo_pattern_details():
    """Show details about available patterns."""
    print("\n" + "=" * 80)
    print("📋 AVAILABLE PATTERNS")
    print("=" * 80)

    matcher = PatternMatcher()
    patterns = matcher.patterns[:6]  # Show first 6 patterns

    for pattern in patterns:
        print(f"\n🔍 Pattern: {pattern.id}")
        print(f"   Priority: {pattern.priority}")
        print(f"   DSL: {pattern.pattern_text}")
        print(f"   Description: {pattern.description}")
        print(f"   Slots: {len(pattern.slots)}")

        for slot in pattern.slots:
            optional = " [optional]" if slot.is_optional else " [required]"
            print(
                f"      • {slot.name} ({slot.slot_type}){optional} weight={slot.weight}"
            )

        print("   Examples:")
        for example in pattern.examples[:2]:  # Show first 2 examples
            print(f"      • {example}")


def demo_scoring():
    """Demonstrate scoring system with different quality inputs."""
    print("\n" + "=" * 80)
    print("📊 CONFIDENCE SCORING DEMO")
    print("=" * 80)

    matcher = PatternMatcher()

    test_cases = [
        ("moda mahalle bahariye sokak no 12", "High Quality - Perfect Pattern"),
        ("istanbul kadiköy moda mahalle bahariye sokak no 12", "With City Context"),
        ("moda mahalle bahariye sokak", "Missing Number"),
        ("mahalle sokak no", "Very Minimal"),
        ("rivella evleri a blok no 77 daire 15", "Site Pattern"),
        ("random text without any structure", "Non-Address Text"),
        ("test test test", "Repetitive Text"),
        ("", "Empty Input"),
    ]

    print("Testing confidence scoring with different input qualities:\n")

    for text, description in test_cases:
        if text:
            matches = matcher.match_text(text)
            if matches:
                best = matches[0]
                confidence = best.confidence
                slots_count = len(best.slots)

                # Visual confidence indicator
                conf_bar = "█" * int(confidence * 20) + "░" * (
                    20 - int(confidence * 20)
                )

                print(f"📊 {description}")
                print(f"   Input: '{text}'")
                print(
                    f"   Confidence: {confidence:.3f} |{conf_bar}| ({slots_count} slots)"  # noqa: E501
                )
                print(f"   Pattern: {best.pattern_id}")
                print()
            else:
                print(f"❌ {description}")
                print(f"   Input: '{text}'")
                print("   Confidence: 0.000 |░░░░░░░░░░░░░░░░░░░░| (no match)")
                print()
        else:
            print(f"❌ {description}: Empty input")
            print()


if __name__ == "__main__":
    try:
        demo_pattern_matching()
        demo_pattern_details()
        demo_scoring()

        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
