"""
ML Pattern Generation Debug Script

Pattern üretim sürecini debug etmek için detaylı analiz.
"""

import logging
from pathlib import Path
import sys
import re  # Eklendi

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.pattern_generation import create_ml_pattern_system, MLPatternConfig, ClusteringAlgorithm

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def debug_pattern_generation():
    """
    Pattern generation sürecini debug et
    """
    print("🔍 DEBUG: ML Pattern Generation")
    print("=" * 50)

    # Sample addresses
    addresses = [
        "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
        "İstanbul Kadıköy Fenerbahçe Mahallesi Bağdat Caddesi No: 123",
        "İstanbul Beşiktaş Ortaköy Mahallesi Mecidiye Sokak No: 7",
        "Ankara Çankaya Kızılay Mahallesi Atatürk Bulvarı No: 89",
        "Ankara Çankaya Kavaklidere Mahallesi Tunalı Caddesi No: 156",
    ]

    # Very relaxed config
    config = MLPatternConfig(
        clustering_algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=2,
        min_cluster_size=2,
        min_pattern_confidence=0.01,  # Çok düşük
        confidence_threshold=0.01,
        quality_threshold=0.01,
        min_generalizability=0.01,  # Çok düşük
        min_coverage=0.001,  # Çok düşük
        max_collision_risk=0.99,
    )

    print(f"📄 Sample addresses: {len(addresses)}")
    for i, addr in enumerate(addresses, 1):
        print(f"  {i}. {addr}")

    print(f"\n🔧 Config:")
    print(f"  min_pattern_confidence: {config.min_pattern_confidence}")
    print(f"  min_generalizability: {config.min_generalizability}")
    print(f"  min_coverage: {config.min_coverage}")

    # Create system
    system = create_ml_pattern_system(config=config, output_dir="debug_output")

    # Generate patterns with detailed logging
    print(f"\n🤖 Generating patterns...")
    patterns = system.generate_patterns_from_addresses(address_samples=addresses, min_cluster_size=2, max_patterns=5)

    print(f"\n📊 Results:")
    print(f"  Final patterns: {len(patterns)}")

    # Access internal suggester to check extracted patterns
    suggester = system.ml_suggester
    extractor = suggester.extractor

    # Manually run clustering to check intermediate results
    clusterer = suggester.clusterer
    features = clusterer.extract_features(addresses)
    clusters = clusterer.cluster_addresses(addresses)

    print(f"\n🔍 Clustering Results:")
    print(f"  Valid clusters: {len(clusters)}")

    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {cluster.size} addresses")
        for j, addr in enumerate(cluster.addresses[:3]):
            print(f"    {j+1}. {addr}")

        # Debug component detection for this cluster
        print(f"    Components detected:")
        for component, pattern in extractor.component_patterns.items():
            matches = []
            for addr in cluster.addresses:
                match = re.findall(pattern, addr, re.IGNORECASE)
                if match:
                    matches.extend(match)
            if matches:
                print(f"      {component}: {matches[:3]}")

    # Extract patterns manually
    print(f"\n🎯 Pattern Extraction:")
    raw_suggestions = extractor.extract_patterns(clusters)
    print(f"  Raw suggestions: {len(raw_suggestions)}")

    for i, suggestion in enumerate(raw_suggestions):
        print(f"\n  Suggestion {i+1}:")
        print(f"    Pattern ID: {suggestion.pattern_id}")
        print(f"    Confidence: {suggestion.confidence:.3f}")
        print(f"    Coverage: {suggestion.coverage:.3f}")
        print(f"    Regex: {suggestion.regex_pattern}")
        print(f"    Template: {suggestion.template.template}")
        print(f"    Generalizability: {suggestion.template.generalizability:.3f}")

        # Check quality filters
        conf_ok = suggestion.confidence >= config.min_pattern_confidence
        coverage_ok = suggestion.coverage >= config.min_coverage
        gen_ok = suggestion.template.generalizability >= config.min_generalizability

        print(f"    Quality Check:")
        print(f"      Confidence >= {config.min_pattern_confidence}: {conf_ok}")
        print(f"      Coverage >= {config.min_coverage}: {coverage_ok}")
        print(f"      Generalizability >= {config.min_generalizability}: {gen_ok}")
        print(f"      Overall: {'✅ PASS' if (conf_ok and coverage_ok and gen_ok) else '❌ FAIL'}")

        # Debug regex matching
        print(f"    Regex Test:")
        try:
            import re as regex_module

            compiled_regex = regex_module.compile(suggestion.regex_pattern, regex_module.IGNORECASE)
            for j, addr in enumerate(cluster.addresses[:3]):
                match = compiled_regex.search(addr)
                print(f"      '{addr}' -> {'✅ MATCH' if match else '❌ NO MATCH'}")
        except Exception as e:
            print(f"      Regex error: {e}")

    return system, patterns


if __name__ == "__main__":
    system, patterns = debug_pattern_generation()
    print(f"\n🎉 Debug completed!")
    print(f"Final result: {len(patterns)} patterns passed quality filter")
