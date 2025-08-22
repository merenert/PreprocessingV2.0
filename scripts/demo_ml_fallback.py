#!/usr/bin/env python3
"""
Demo script showing ML-based Turkish address normalization fallback.

This script demonstrates how the ML system works as a fallback when
pattern matching confidence is low.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from addrnorm.ml.infer import (
    get_ml_normalizer,
    is_ml_available,
    normalize_with_ml_fallback,
)


def demo_ml_fallback():
    """Demonstrate ML fallback functionality."""
    print("🤖 Turkish Address ML Fallback Demo")
    print("=" * 50)

    # Check ML availability
    print(f"ML Available: {is_ml_available()}")

    if not is_ml_available():
        print("❌ ML fallback not available. Please train a model first.")
        return

    # Get normalizer stats
    normalizer = get_ml_normalizer()
    stats = normalizer.get_stats()
    print(f"Model Path: {stats['model_path']}")
    print(f"Confidence Threshold: {stats['confidence_threshold']}")
    print()

    # Test addresses
    test_addresses = [
        "atatürk mahallesi cumhuriyet caddesi no 15 kat 3 daire 5 ankara",
        "konak mahallesi alsancak bulvarı numara 23 izmir",
        "bağdat caddesi no 125 kadıköy istanbul",
        "mehmet akif ersoy mahallesi 1425 sokak no 8 kat 2 muğla",
        "güzelyalı mahallesi iskele caddesi 45/3 izmir",
        "barbaros bulvarı no 234 kat 5 daire 12 beşiktaş istanbul",
    ]

    print("🏠 Testing Address Normalization with ML Fallback:")
    print("-" * 60)

    for i, address in enumerate(test_addresses, 1):
        print(f"\n{i}. Address: {address}")

        # Simulate different confidence scenarios
        if i <= 2:
            # High confidence pattern - use ML directly
            result = normalize_with_ml_fallback(address)
        else:
            # Simulate low confidence pattern by providing fake pattern result
            fake_pattern_result = {
                "confidence": 0.3,  # Low confidence
                "normalized": {"text": address},
            }
            result = normalize_with_ml_fallback(
                address, pattern_result=fake_pattern_result
            )

        print(f"   Method: {result['method']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Fallback Used: {result['fallback_used']}")
        print("   Normalized Components:")

        for component, value in result["normalized"].items():
            print(f"     {component}: {value}")

    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")


def interactive_demo():
    """Interactive ML demo."""
    print("\n🔄 Interactive ML Address Normalization")
    print("Enter addresses to normalize (type 'quit' to exit):")
    print("-" * 50)

    while True:
        try:
            address = input("\nAddress: ").strip()

            if address.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not address:
                continue

            # Use ML fallback
            result = normalize_with_ml_fallback(address)

            print("\n📍 Result:")
            print(f"   Method: {result['method']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print("   Components:")

            for component, value in result["normalized"].items():
                print(f"     {component}: {value}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Address Normalization Demo")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive demo mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        demo_ml_fallback()
