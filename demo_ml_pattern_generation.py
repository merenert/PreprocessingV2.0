"""
ML Pattern Generation Demo

ML tabanlı pattern generation sisteminin demo ve test script'i.
"""

import logging
from pathlib import Path
import sys
import json

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.pattern_generation import create_ml_pattern_system, MLPatternConfig, ClusteringAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def get_sample_addresses():
    """
    Test için örnek Türkçe adresler
    """
    return [
        # Istanbul addresses
        "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
        "İstanbul Kadıköy Fenerbahçe Mahallesi Bağdat Caddesi No: 123/A",
        "İstanbul Kadıköy Göztepe Mahallesi Özgürlük Caddesi No: 45",
        "İstanbul Beşiktaş Ortaköy Mahallesi Mecidiye Köprüsü Sokak No: 7",
        "İstanbul Beşiktaş Arnavutköy Mahallesi Bebek Yolu No: 33",
        # Ankara addresses
        "Ankara Çankaya Kızılay Mahallesi Atatürk Bulvarı No: 89",
        "Ankara Çankaya Kavaklidere Mahallesi Tunalı Hilmi Caddesi No: 156",
        "Ankara Keçiören Etlik Mahallesi Şehit Teğmen Kalmaz Caddesi No: 12",
        "Ankara Yenimahalle Ostim Mahallesi İvedik Caddesi No: 234",
        # Izmir addresses
        "İzmir Konak Alsancak Mahallesi Cumhuriyet Bulvarı No: 45",
        "İzmir Bornova Erzene Mahallesi İzmir Yolu Caddesi No: 178",
        "İzmir Karşıyaka Bostanlı Mahallesi Atatürk Caddesi No: 67",
        "İzmir Balçova Narlidere Mahallesi Mithatpaşa Caddesi No: 91",
        # Other cities
        "Bursa Osmangazi Soğanlı Mahallesi Atatürk Caddesi No: 23",
        "Antalya Muratpaşa Fener Mahallesi Cumhuriyet Caddesi No: 145",
        "Adana Seyhan Kurtuluş Mahallesi İnönü Caddesi No: 78",
        "Konya Selçuklu Sille Mahallesi Mevlana Bulvarı No: 234",
        # Different formats
        "Eskişehir Tepebaşı Akarbaşı Mah. Zübeyde Hanım Cad. No:15/3",
        "Gaziantep Şahinbey Karataş Mah. Atatürk Blv. No:67 Kat:2",
        "Mersin Yenişehir Çankaya Mah. Gazi Mustafa Kemal Blv. No:89/A",
        "Kayseri Kocasinan Osman Kavuncu Mah. Erciyes Blv. No:123 D:4",
    ]


def get_existing_patterns():
    """
    Mevcut pattern'ler (örnek)
    """
    return [
        r"^(\w+)\s+(\w+)\s+(\w+)\s+Mahallesi\s+(\w+)\s+Caddesi\s+No:\s*(\d+)$",
        r"^(\w+)\s+(\w+)\s+(\w+)\s+Mah\.\s+(\w+)\s+Cad\.\s+No:(\d+)/(\w+)$",
        r"^(\w+)\s+(\w+)\s+(\w+)\s+Mahallesi\s+(\w+)\s+Bulvarı\s+No:\s*(\d+)$",
    ]


def demo_basic_pattern_generation():
    """
    Temel pattern generation demo
    """
    print("\n" + "=" * 60)
    print("🎯 BASIC PATTERN GENERATION DEMO")
    print("=" * 60)

    # Create system with custom config
    config = MLPatternConfig(
        clustering_algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3,
        min_cluster_size=2,
        min_pattern_confidence=0.1,  # Çok düşük threshold
        confidence_threshold=0.1,
        quality_threshold=0.1,
        min_generalizability=0.1,  # Çok düşük threshold
        min_coverage=0.01,  # Çok düşük threshold
        max_collision_risk=0.9,
    )

    system = create_ml_pattern_system(config=config, output_dir="demo_output")

    # Get sample data
    addresses = get_sample_addresses()

    print(f"📄 Sample addresses: {len(addresses)}")
    for i, addr in enumerate(addresses[:5], 1):
        print(f"  {i}. {addr}")
    print(f"  ... and {len(addresses)-5} more")

    # Generate patterns
    print(f"\n🤖 Generating patterns...")
    patterns = system.generate_patterns_from_addresses(
        address_samples=addresses, min_cluster_size=2, max_patterns=10  # Düşürüldü
    )

    print(f"✅ Generated {len(patterns)} patterns")

    # Show sample patterns
    for i, pattern in enumerate(patterns[:3], 1):
        print(f"\n📋 Pattern {i}:")
        print(f"  Type: {pattern.pattern_type.value}")
        print(f"  Confidence: {pattern.confidence:.2f}")
        print(f"  Coverage: {pattern.coverage:.2f}")
        print(f"  Regex: {pattern.regex_pattern}")
        print(f"  Template: {pattern.template.template}")
        if pattern.examples:
            print(f"  Example: {pattern.examples[0]}")

    return system, patterns


def demo_validation_and_conflicts():
    """
    Validation ve conflict detection demo
    """
    print("\n" + "=" * 60)
    print("🔍 VALIDATION & CONFLICT DETECTION DEMO")
    print("=" * 60)

    system, patterns = demo_basic_pattern_generation()

    # Validate patterns
    print(f"\n🎯 Validating {len(patterns)} patterns...")
    validation_results = system.validate_patterns(patterns=patterns, existing_patterns=get_existing_patterns())

    # Show validation summary
    valid_count = sum(1 for r in validation_results if r.is_valid)
    avg_quality = sum(r.quality_score for r in validation_results) / len(validation_results) if validation_results else 0.0

    print(f"✅ Validation complete:")
    print(f"  Valid patterns: {valid_count}/{len(patterns)}")
    print(f"  Average quality: {avg_quality:.2f}")

    # Show sample validation results
    for i, result in enumerate(validation_results[:3], 1):
        print(f"\n📊 Validation {i}:")
        print(f"  Valid: {'✅' if result.is_valid else '❌'}")
        print(f"  Quality: {result.quality_score:.2f}")
        print(f"  Generalizability: {result.generalizability_score:.2f}")
        print(f"  Collision Risk: {result.collision_risk:.2f}")
        if result.issues:
            print(f"  Issues: {', '.join(result.issues[:2])}")

    # Detect conflicts
    print(f"\n⚔️ Detecting conflicts...")
    conflict_reports = system.detect_conflicts(patterns=patterns, existing_patterns=get_existing_patterns())

    print(f"🚨 Found {len(conflict_reports)} conflicts")

    # Show conflict summary
    critical_conflicts = sum(1 for c in conflict_reports if c.severity.name == "CRITICAL")
    print(f"  Critical conflicts: {critical_conflicts}")

    for i, conflict in enumerate(conflict_reports[:2], 1):
        print(f"\n⚠️ Conflict {i}:")
        print(f"  Severity: {conflict.severity.value}")
        print(f"  Overlap Score: {conflict.overlap_score:.2f}")
        print(f"  Conflicting patterns: {len(conflict.conflicting_patterns)}")
        if conflict.ambiguity_examples:
            print(f"  Ambiguous example: {conflict.ambiguity_examples[0]}")

    return system, patterns, validation_results, conflict_reports


def demo_batch_review():
    """
    Batch review demo (automatic approval/rejection)
    """
    print("\n" + "=" * 60)
    print("🤖 BATCH REVIEW DEMO")
    print("=" * 60)

    system, patterns, validation_results, conflict_reports = demo_validation_and_conflicts()

    # Batch review
    print(f"\n🔄 Running batch review for {len(patterns)} patterns...")
    review_decisions = system.review_patterns(
        patterns=patterns,
        validation_results=validation_results,
        conflict_reports=conflict_reports,
        reviewer_name="demo_batch_reviewer",
        batch_mode=True,  # Automatic decisions
    )

    # Review summary
    approved = sum(1 for d in review_decisions if d.decision.value == "approved")
    rejected = sum(1 for d in review_decisions if d.decision.value == "rejected")
    pending = sum(1 for d in review_decisions if d.decision.value == "pending")

    print(f"📊 Batch review complete:")
    print(f"  ✅ Approved: {approved}")
    print(f"  ❌ Rejected: {rejected}")
    print(f"  🤔 Pending: {pending}")

    # Get approved patterns
    approved_patterns = system.get_approved_patterns()
    print(f"\n🎉 Final approved patterns: {len(approved_patterns)}")

    for i, pattern in enumerate(approved_patterns, 1):
        print(f"  {i}. {pattern.template.template} (confidence: {pattern.confidence:.2f})")

    return system


def demo_full_pipeline():
    """
    Full pipeline demo
    """
    print("\n" + "=" * 70)
    print("🚀 FULL ML PATTERN GENERATION PIPELINE DEMO")
    print("=" * 70)

    # Create system
    system = create_ml_pattern_system(output_dir="full_pipeline_demo")

    # Get data
    addresses = get_sample_addresses()
    existing_patterns = get_existing_patterns()

    print(f"📊 Pipeline input:")
    print(f"  Address samples: {len(addresses)}")
    print(f"  Existing patterns: {len(existing_patterns)}")

    # Run full pipeline
    print(f"\n🔄 Running full pipeline...")
    results = system.full_pipeline(
        address_samples=addresses,
        existing_patterns=existing_patterns,
        reviewer_name="demo_reviewer",
        batch_review=True,  # Use batch review for demo
        auto_export=True,
    )

    # Show results
    print(f"\n🎉 Pipeline complete!")
    print(f"  Duration: {results['pipeline_duration']:.2f} seconds")
    print(f"  Patterns generated: {results['total_patterns_generated']}")
    print(f"  Patterns approved: {results['approved_patterns']}")
    print(f"  Patterns needing modification: {results['patterns_needing_modification']}")
    print(f"  Patterns rejected: {results['rejected_patterns']}")
    print(f"  Total conflicts: {results['total_conflicts']}")
    print(f"  Critical conflicts: {results['critical_conflicts']}")

    if results["report_file"]:
        print(f"  📄 Report saved: {results['report_file']}")

    # Show system metrics
    metrics = results["system_metrics"]
    print(f"\n📈 System metrics:")
    print(f"  Addresses processed: {metrics['total_addresses_processed']}")
    print(f"  Avg generation time: {metrics['avg_generation_time']:.2f}s")
    print(f"  Avg validation time: {metrics['avg_validation_time']:.2f}s")

    return system, results


def main():
    """
    Ana demo function
    """
    print("🎯 ML PATTERN GENERATION SYSTEM DEMO")
    print("=" * 50)

    try:
        # Run demos
        demo_basic_pattern_generation()
        demo_validation_and_conflicts()
        demo_batch_review()
        system, results = demo_full_pipeline()

        print(f"\n✅ All demos completed successfully!")
        print(f"🔗 Check output files in: {system.output_dir}")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
