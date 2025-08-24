#!/usr/bin/env python3
"""
Simple demo for hybrid ML + pattern-based address normalization.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.addrnorm.integration.hybrid import HybridAddressNormalizer, IntegrationConfig


def main():
    """Run simple hybrid normalization demo."""
    print("🚀 Hybrid ML + Pattern Address Normalization Demo")
    print("=" * 60)

    # Initialize hybrid normalizer
    config = IntegrationConfig(enable_ml_fallback=True, min_pattern_confidence=0.7, max_processing_time=3.0)

    normalizer = HybridAddressNormalizer(config)

    # Test addresses with varying complexity
    test_addresses = [
        "Çankaya Mahallesi Tunali Hilmi Caddesi No:15 Daire:3 Ankara",
        "Kadıköy İstanbul Fenerbahçe Mahallesi",
        "06100 Ankara Çankaya Kızılay Meydanı",
        "İzmir Konak Alsancak bölgesi yakınları",
        "Maltepe Istanbul çok belirsiz bir adres",
    ]

    print(f"\n📊 Processing {len(test_addresses)} test addresses:")
    print("-" * 40)

    for i, address in enumerate(test_addresses, 1):
        print(f"\n🔍 Test {i}: {address}")

        try:
            # Normalize address using hybrid approach
            result = normalizer.normalize(address)

            print(f"✅ Success: {result.normalized_address}")
            print(f"📈 Confidence: {result.explanation_parsed.confidence:.3f}")
            print(f"🔧 Method: {result.explanation_parsed.method}")
            print(f"🏙️  City: {result.city}")
            print(f"🏘️  District: {result.district}")
            print(f"🏠 Neighborhood: {result.neighborhood}")
            print(f"🛣️  Street: {result.street}")

            if result.explanation_parsed.warnings:
                print(f"⚠️  Warnings: {', '.join(result.explanation_parsed.warnings)}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Show performance stats
    print(f"\n\n📈 Performance Statistics:")
    print("-" * 30)
    stats = normalizer.get_performance_stats()

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\n🎉 Demo completed successfully!")
    print("✨ Hybrid ML + Pattern approach implemented with:")
    print("   ✅ Adaptive threshold calculation")
    print("   ✅ Pattern-first with ML fallback")
    print("   ✅ Performance tracking")
    print("   ✅ Confidence-based method selection")


if __name__ == "__main__":
    main()
