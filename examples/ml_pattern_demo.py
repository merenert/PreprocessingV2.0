"""
Example usage of ML Pattern Generation System

This script demonstrates the complete ML pattern generation workflow.
"""

import sys

sys.path.append("src")

from addrnorm.pattern_generation import (
    MLPatternSuggester,
    PatternConflictDetector,
    PatternAnalyzer,
    PatternReviewInterface,
    PatternGenerationConfig,
)


def main():
    """Demonstrate ML pattern generation system"""

    # Sample Turkish addresses
    sample_addresses = [
        "Atatürk Mahallesi, Cumhuriyet Caddesi No:25 Daire:3",
        "Merkez Mah. İstiklal Sok. 15/4",
        "Yenişehir Mahallesi Gazi Bulvarı 42",
        "Kültür Mah, Barış Sk, No:8",
        "Bahçelievler Mahallesi 15. Sokak No:23 Daire:4",
        "Atatürk Mahallesi Vatan Caddesi 67/2",
        "Merkez Mahallesi Özgürlük Sokağı No:12",
        "Yenişehir Mah. Atatürk Bulvarı 89 Kat:2",
    ]

    print("🚀 ML Pattern Generation System Demo")
    print("=" * 50)

    # Step 1: Initialize components
    print("\n1️⃣ Initializing ML components...")

    config = PatternGenerationConfig(
        min_pattern_confidence=0.2, max_clusters=5, min_cluster_size=2  # Lower threshold for demo
    )

    suggester = MLPatternSuggester(config)
    conflict_detector = PatternConflictDetector()
    pattern_analyzer = PatternAnalyzer()
    review_interface = PatternReviewInterface()

    print("✅ Components initialized")

    # Step 2: Generate pattern suggestions
    print(f"\n2️⃣ Generating patterns from {len(sample_addresses)} addresses...")

    try:
        suggestions = suggester.suggest_patterns(sample_addresses, max_patterns=5)
        print(f"✅ Generated {len(suggestions)} pattern suggestions")

        for i, suggestion in enumerate(suggestions, 1):
            print(f"   Pattern {i}: {suggestion.regex_pattern[:50]}...")
            print(f"   Confidence: {suggestion.confidence:.3f}")
            print(f"   Description: ML-generated pattern from cluster")
            print()

    except Exception as e:
        print(f"❌ Error generating patterns: {e}")
        return

    # Step 3: Analyze pattern quality
    print("3️⃣ Analyzing pattern quality...")

    try:
        for suggestion in suggestions:
            quality_analysis = pattern_analyzer.analyze_pattern_quality(suggestion.regex_pattern, sample_addresses)
            print(f"   {suggestion.pattern_id}: Quality Score = {quality_analysis.overall_quality_score:.3f}")

        print("✅ Quality analysis completed")

    except Exception as e:
        print(f"❌ Error analyzing quality: {e}")

    # Step 4: Detect conflicts (simulate existing patterns)
    print("\n4️⃣ Detecting conflicts with existing patterns...")

    try:
        # Simulate some existing patterns
        existing_patterns = [
            {
                "id": "existing_1",
                "pattern": r"(\w+)\s+Mahallesi.*No:(\d+)",
                "fields": ["mahalle", "no"],
                "description": "Existing mahalle pattern",
                "confidence": 0.7,
            }
        ]

        conflicts = conflict_detector.detect_conflicts(existing_patterns, suggestions)
        print(f"✅ Detected {len(conflicts)} conflicts")

        for conflict in conflicts:
            print(f"   Conflict: {conflict.conflict_type.value} ({conflict.severity.value})")
            print(f"   Description: {conflict.description}")
            print()

    except Exception as e:
        print(f"❌ Error detecting conflicts: {e}")

    # Step 5: Submit for review
    print("5️⃣ Submitting for human review...")

    try:
        review_id = review_interface.submit_for_review(
            suggestions=suggestions,
            conflicts=conflicts if "conflicts" in locals() else [],
            reviewer_id="demo_user",
            priority="normal",
        )

        print(f"✅ Review session created: {review_id}")

        # Show review summary
        session = review_interface.get_review_session(review_id)
        if session and "review_summary" in session:
            summary = session["review_summary"]
            print(f"   Patterns to review: {summary['pattern_summary']['total_patterns']}")
            print(f"   Conflicts to resolve: {summary['conflict_summary']['total_conflicts']}")
            print(f"   Estimated time: {review_interface._estimate_review_time(session)}")

    except Exception as e:
        print(f"❌ Error submitting review: {e}")

    # Step 6: Show statistics
    print("\n6️⃣ System Statistics:")

    try:
        stats = review_interface.get_review_statistics()
        print(f"   Total review sessions: {stats['review_sessions']['total']}")
        print(f"   Pending reviews: {stats['review_sessions']['pending']}")
        print(f"   Completed reviews: {stats['review_sessions']['completed']}")

    except Exception as e:
        print(f"❌ Error getting statistics: {e}")

    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("   - Use the CLI interface for real-world usage")
    print("   - Review and approve/reject patterns")
    print("   - Integrate approved patterns into normalization system")


if __name__ == "__main__":
    main()
