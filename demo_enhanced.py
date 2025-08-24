#!/usr/bin/env python3
"""
Enhanced Demo for Turkish Address Normalization
Demonstrates the enhanced output schema with confidence scoring system
"""

import json
import argparse
import time
from typing import List

from src.addrnorm.integration.hybrid import HybridAddressNormalizer, IntegrationConfig


def demo_enhanced_output():
    """Demonstrate enhanced output format with confidence scoring"""

    print("🚀 Turkish Address Normalization - Enhanced Output Demo")
    print("=" * 60)

    # Sample Turkish addresses for testing
    test_addresses = [
        "Kızılay Mahallesi Atatürk Bulvarı No:123 Çankaya/ANKARA",
        "Beşiktaş İstanbul Türkiye",
        "Kadıköy Moda Caferağa Mh. İstanbul",
        "Levent 4. Levent Mahallesi Büyükdere Caddesi No:185 Şişli/İSTANBUL",
        "Bağcılar İstanbul 34200",
        "Maltepe Kartal İstanbul",
        "Ulus Altındağ Ankara 06030",
    ]

    # Initialize normalizer with enhanced output
    config = IntegrationConfig(
        enhanced_output=True, enable_ml_fallback=True, min_pattern_confidence=0.7, enable_performance_tracking=True
    )

    normalizer = HybridAddressNormalizer(config)

    print(f"Processing {len(test_addresses)} sample addresses...\n")

    for i, address in enumerate(test_addresses, 1):
        print(f"🏠 Address {i}: {address}")
        print("-" * 50)

        start_time = time.time()
        processing_time = 0.0  # Initialize processing_time

        try:
            # Process with enhanced output
            result = normalizer.normalize(address, enhanced_output=True)

            processing_time = (time.time() - start_time) * 1000

            # Display enhanced output in formatted JSON
            json_output = normalizer.enhanced_formatter.to_json(result, indent=2)
            print(json_output)

            # Show compact format
            compact = normalizer.enhanced_formatter.format_compact(result)
            print(f"\n📊 Compact: {compact}")

            # Show validation results
            validation = normalizer.enhanced_formatter.validate_output(result)
            if validation["warnings"]:
                print(f"⚠️  Warnings: {', '.join(validation['warnings'])}")
            if validation["errors"]:
                print(f"❌ Errors: {', '.join(validation['errors'])}")

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"❌ Error: {e}")

        print(f"\n⏱️  Processing time: {processing_time:.1f}ms")
        print("=" * 60)
        print()

    # Show performance statistics
    stats = normalizer.get_performance_stats()
    print("📈 Performance Statistics:")
    print(f"   Total processed: {stats['total_processed']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    if stats["total_processed"] > 0:
        print(f"   Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
        print(f"   Pattern usage: {stats.get('pattern_usage_pct', 0):.1f}%")
        print(f"   ML usage: {stats.get('ml_usage_pct', 0):.1f}%")
        print(f"   Hybrid usage: {stats.get('hybrid_usage_pct', 0):.1f}%")


def demo_comparison():
    """Demonstrate comparison between legacy and enhanced output"""

    print("\n🔄 Legacy vs Enhanced Output Comparison")
    print("=" * 60)

    test_address = "Kızılay Mahallesi Atatürk Bulvarı No:123 Çankaya/ANKARA"

    config = IntegrationConfig(enable_ml_fallback=True)
    normalizer = HybridAddressNormalizer(config)

    print(f"Test Address: {test_address}\n")

    # Legacy output
    print("📄 Legacy Output:")
    print("-" * 30)
    legacy_result = normalizer.normalize(test_address, enhanced_output=False)
    if hasattr(legacy_result, "model_dump"):
        print(json.dumps(legacy_result.model_dump(exclude_none=True), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(legacy_result.__dict__, indent=2, ensure_ascii=False, default=str))

    print("\n📊 Enhanced Output:")
    print("-" * 30)
    enhanced_result = normalizer.normalize(test_address, enhanced_output=True)
    enhanced_json = normalizer.enhanced_formatter.to_json(enhanced_result, indent=2)
    print(enhanced_json)

    print("\n🔄 Legacy Format Conversion:")
    print("-" * 30)
    legacy_converted = normalizer.enhanced_formatter.to_legacy_format(enhanced_result)
    print(json.dumps(legacy_converted, indent=2, ensure_ascii=False))


def demo_batch_processing():
    """Demonstrate batch processing with enhanced output"""

    print("\n📦 Batch Processing Demo")
    print("=" * 60)

    batch_addresses = [
        "Beşiktaş İstanbul",
        "Kadıköy Moda İstanbul",
        "Levent Şişli İstanbul",
        "Ulus Ankara",
        "Kızılay Çankaya Ankara",
    ]

    config = IntegrationConfig(enhanced_output=True, enable_performance_tracking=True)
    normalizer = HybridAddressNormalizer(config)

    print(f"Processing batch of {len(batch_addresses)} addresses...\n")

    start_time = time.time()
    results = normalizer.batch_normalize(batch_addresses, enhanced_output=True)
    total_time = time.time() - start_time

    print(f"✅ Batch completed in {total_time:.2f}s")
    print(f"📊 Average: {(total_time/len(batch_addresses))*1000:.1f}ms per address\n")

    # Show summary of results
    success_count = sum(1 for r in results if r.success)
    print(f"📈 Results Summary:")
    print(f"   Success rate: {success_count}/{len(results)} ({success_count/len(results):.1%})")

    # Show confidence distribution
    confidence_scores = [r.confidence_scores.overall for r in results if r.confidence_scores]
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Confidence range: {min(confidence_scores):.3f} - {max(confidence_scores):.3f}")


def demo_interactive():
    """Interactive demo for testing custom addresses"""

    print("\n💬 Interactive Demo")
    print("=" * 60)
    print("Enter Turkish addresses to test enhanced normalization")
    print("Type 'quit' to exit, 'stats' to show performance statistics")
    print()

    config = IntegrationConfig(enhanced_output=True)
    normalizer = HybridAddressNormalizer(config)

    while True:
        try:
            address = input("🏠 Enter address: ").strip()

            if address.lower() in ["quit", "exit", "q"]:
                break
            elif address.lower() == "stats":
                stats = normalizer.get_performance_stats()
                print(f"\n📊 Statistics: {json.dumps(stats, indent=2)}\n")
                continue
            elif not address:
                continue

            print(f"\nProcessing: {address}")
            print("-" * 50)

            start_time = time.time()
            result = normalizer.normalize(address)
            processing_time = (time.time() - start_time) * 1000

            # Show compact result
            compact = normalizer.enhanced_formatter.format_compact(result)
            print(f"📊 Result: {json.dumps(compact, ensure_ascii=False)}")
            print(f"⏱️  Time: {processing_time:.1f}ms")

            # Ask if user wants full output
            show_full = input("\nShow full enhanced output? (y/N): ").strip().lower()
            if show_full in ["y", "yes"]:
                full_output = normalizer.enhanced_formatter.to_json(result, indent=2)
                print(f"\n📋 Full Output:\n{full_output}")

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main demo function with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Turkish Address Normalization - Enhanced Output Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_enhanced.py                    # Run full demo
  python demo_enhanced.py --demo basic       # Basic enhanced output demo
  python demo_enhanced.py --demo comparison  # Legacy vs enhanced comparison
  python demo_enhanced.py --demo batch       # Batch processing demo
  python demo_enhanced.py --demo interactive # Interactive testing
        """,
    )

    parser.add_argument(
        "--demo",
        choices=["basic", "comparison", "batch", "interactive", "all"],
        default="all",
        help="Demo type to run (default: all)",
    )

    parser.add_argument("--address", help="Single address to test")

    parser.add_argument(
        "--format", choices=["enhanced", "legacy", "compact"], default="enhanced", help="Output format (default: enhanced)"
    )

    args = parser.parse_args()

    if args.address:
        # Single address mode
        config = IntegrationConfig(enhanced_output=(args.format == "enhanced"))
        normalizer = HybridAddressNormalizer(config)

        print(f"🏠 Processing: {args.address}")
        print("-" * 50)

        start_time = time.time()
        result = normalizer.normalize(args.address)
        processing_time = (time.time() - start_time) * 1000

        if args.format == "enhanced":
            output = normalizer.enhanced_formatter.to_json(result, indent=2)
        elif args.format == "compact":
            output = json.dumps(normalizer.enhanced_formatter.format_compact(result), indent=2, ensure_ascii=False)
        else:  # legacy
            legacy_data = normalizer.enhanced_formatter.to_legacy_format(result)
            output = json.dumps(legacy_data, indent=2, ensure_ascii=False)

        print(output)
        print(f"\n⏱️  Processing time: {processing_time:.1f}ms")

    else:
        # Demo mode
        if args.demo == "all":
            demo_enhanced_output()
            demo_comparison()
            demo_batch_processing()
            demo_interactive()
        elif args.demo == "basic":
            demo_enhanced_output()
        elif args.demo == "comparison":
            demo_comparison()
        elif args.demo == "batch":
            demo_batch_processing()
        elif args.demo == "interactive":
            demo_interactive()


if __name__ == "__main__":
    main()
