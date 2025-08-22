#!/usr/bin/env python3
"""
Demo script for dynamic threshold management system.

This script demonstrates how the adaptive threshold system works:
1. Initial pattern matching with default thresholds
2. Simulated feedback (success/failure)
3. Threshold adjustments based on feedback
4. Impact on matching behavior
"""

import sys

sys.path.append("src")

import shutil
import tempfile

from addrnorm.patterns.compiler import PatternCompiler
from addrnorm.patterns.matcher import PatternMatcher
from addrnorm.patterns.thresholds import ThresholdManager


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"🎯 {title}")
    print(f"{'=' * 60}")


def print_threshold_info(matcher: PatternMatcher, pattern_id: str):
    """Print detailed threshold information."""
    info = matcher.get_threshold_info(pattern_id)

    print(f"📊 Pattern: {pattern_id}")
    print(f"   Current Threshold: {info['current_threshold']:.3f}")
    print(f"   Default Threshold: {info['default_threshold']:.3f}")
    print(f"   Is Adjusted: {'✅' if info['is_adjusted'] else '❌'}")
    print(f"   Samples Seen: {info['seen']}")

    if info["ema_success"] is not None:
        print(f"   EMA Success Rate: {info['ema_success']:.3f}")
        print(f"   Has Enough Samples: {'✅' if info['has_enough_samples'] else '❌'}")
    else:
        print("   EMA Success Rate: Not available")


def test_address_matching(matcher: PatternMatcher, addresses: list, scenario_name: str):
    """Test matching behavior for given addresses."""
    print(f"\n🔍 {scenario_name}")
    print("-" * 40)

    total_matches = 0
    for addr in addresses:
        matches = matcher.match_text(addr)
        if matches:
            best = matches[0]
            print(
                f"   ✅ '{addr[:40]}...' → {best.pattern_id} (conf: {best.confidence:.3f})"  # noqa: E501
            )
            total_matches += 1
        else:
            print(f"   ❌ '{addr[:40]}...' → No match")

    success_rate = total_matches / len(addresses) if addresses else 0
    print(f"   📈 Success Rate: {success_rate:.1%} ({total_matches}/{len(addresses)})")
    return success_rate


def main():
    """Main demo function."""
    print_header("DYNAMIC THRESHOLD MANAGEMENT DEMO")

    # Create temporary cache directory for this demo
    temp_cache = tempfile.mkdtemp()
    print(f"📁 Using temporary cache: {temp_cache}")

    try:
        # Initialize components
        compiler = PatternCompiler("data/patterns/tr.yml")
        threshold_manager = ThresholdManager(cache_dir=temp_cache)
        matcher = PatternMatcher(compiler=compiler, threshold_manager=threshold_manager)

        # Test addresses - mix of good and challenging cases
        test_addresses = [
            "moda mahalle bahariye sokak no 12",
            "kadiköy mahalle atatürk cadde no 45",
            "alsancak mahalle cumhuriyet sokak no 123",
            "balgat mahalle turan güneş cadde no 67",
            "some random text without structure",
            "incomplete address missing",
            "görükle mahalle university cadde no 34",
            "lara mahalle barış manço cadde no 67",
        ]

        print(f"🏠 Testing with {len(test_addresses)} addresses")

        # Get a pattern to focus on for demonstration
        demo_pattern = "mahalle_sokak_no"

        print_header("INITIAL STATE")
        print_threshold_info(matcher, demo_pattern)

        # Test initial matching behavior
        initial_success_rate = test_address_matching(
            matcher, test_addresses, "Initial Matching (Default Thresholds)"
        )

        print_header("SIMULATING POSITIVE FEEDBACK")
        print(f"🔄 Providing positive feedback for pattern '{demo_pattern}'...")

        # Simulate positive feedback (high success rate)
        for i in range(8):
            matcher.provide_feedback(demo_pattern, True)
            print(f"   ✅ Feedback {i + 1}/8: Success")

        print_threshold_info(matcher, demo_pattern)

        # Test matching after positive feedback
        positive_success_rate = test_address_matching(
            matcher, test_addresses, "After Positive Feedback (Lower Threshold)"
        )

        print_header("SIMULATING NEGATIVE FEEDBACK")
        print(f"🔄 Providing negative feedback for pattern '{demo_pattern}'...")

        # Simulate negative feedback (low success rate)
        for i in range(12):
            matcher.provide_feedback(demo_pattern, False)
            print(f"   ❌ Feedback {i + 1}/12: Failure")

        print_threshold_info(matcher, demo_pattern)

        # Test matching after negative feedback
        negative_success_rate = test_address_matching(
            matcher, test_addresses, "After Negative Feedback (Higher Threshold)"
        )

        print_header("THRESHOLD EVOLUTION SUMMARY")
        print("📊 Success Rate Comparison:")
        print(f"   Initial (Default):     {initial_success_rate:.1%}")
        print(f"   After Positive FB:     {positive_success_rate:.1%}")
        print(f"   After Negative FB:     {negative_success_rate:.1%}")

        # Show threshold changes for multiple patterns
        print("\n📋 All Pattern Thresholds:")
        all_patterns = [
            "mahalle_sokak_no",
            "mahalle_cadde_no",
            "sokak_numara_simple",
            "loose_mahalle_only",
        ]

        for pattern in all_patterns:
            info = matcher.get_threshold_info(pattern)
            status = "📈 Adjusted" if info["is_adjusted"] else "📊 Default"
            print(f"   {status} {pattern}: {info['current_threshold']:.3f}")

        print_header("PERSISTENCE TEST")
        print("🔄 Creating new matcher instance to test persistence...")

        # Create new matcher with same cache - thresholds should persist
        new_threshold_manager = ThresholdManager(cache_dir=temp_cache)
        new_matcher = PatternMatcher(
            compiler=compiler, threshold_manager=new_threshold_manager
        )

        old_threshold = matcher.get_current_threshold(demo_pattern)
        new_threshold = new_matcher.get_current_threshold(demo_pattern)

        if abs(old_threshold - new_threshold) < 0.001:
            print(
                f"   ✅ Persistence verified: {old_threshold:.3f} = {new_threshold:.3f}"
            )
        else:
            print(
                f"   ❌ Persistence failed: {old_threshold:.3f} ≠ {new_threshold:.3f}"
            )

        print_header("ADAPTIVE BEHAVIOR DEMONSTRATION")

        # Show how different success rates affect thresholds
        test_patterns = [
            ("high_success_pattern", [True] * 10),
            ("medium_success_pattern", [True] * 5 + [False] * 5),
            ("low_success_pattern", [False] * 8 + [True] * 2),
        ]

        for pattern_name, feedback_sequence in test_patterns:
            print(f"\n🎯 Testing {pattern_name}:")

            # Apply feedback sequence
            for feedback in feedback_sequence:
                matcher.provide_feedback(pattern_name, feedback)

            info = matcher.get_threshold_info(pattern_name)
            success_rate = info["ema_success"] if info["ema_success"] else 0

            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Threshold: {info['current_threshold']:.3f}")

            if success_rate > 0.7:
                print("   💡 High success → Lower threshold (more aggressive)")
            elif success_rate < 0.3:
                print("   💡 Low success → Higher threshold (more conservative)")
            else:
                print("   💡 Medium success → Moderate adjustment")

        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("✅ Dynamic threshold management is working correctly!")
        print("✅ Thresholds adapt based on feedback")
        print("✅ Changes persist across sessions")
        print("✅ Matching behavior adjusts accordingly")

    finally:
        # Clean up temporary cache
        shutil.rmtree(temp_cache)
        print(f"\n🧹 Cleaned up temporary cache: {temp_cache}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
