"""
Regression testing for Turkish address normalization.
Tests for unexpected behavior changes when pattern thresholds or configurations change.
"""

import json
import time
from typing import Any, Dict

import pytest

from addrnorm.pipeline import PipelineConfig, create_pipeline


class RegressionTestRunner:
    """Runner for regression tests with threshold variation analysis."""

    def __init__(self):
        """Initialize the regression test runner."""
        self.baseline_results = {}
        self.test_addresses = [
            "İstanbul Beşiktaş Levent Mahallesi Büyükdere Caddesi No:100",
            "Ankara Çankaya Kızılay Meydanı No:1",
            "İzmir Konak Alsancak Mahallesi Kordon Boyu Sok. 12A",
            "Bursa Osmangazi Çekirge Mah. Mürsel Paşa Bulvarı 15/A",
            "Beşiktaş İlçesi Levent Mah. Büyükdere Cad. Yapı Kredi Plaza A Blok",
            "Kadıköy İlçesi Moda Mahallesi Caferağa Mah. Şair Nedim Cad. No:15",
            "Şişli Mecidiyeköy Büyükdere Cad. No:78 K:5 D:12 İstanbul",
            "Ankara Keçiören Aktepe Mahallesi Atatürk Bulvarı 45/7",
            "İzmir Bornova Ege Üniversitesi Kampüsü Mühendislik Fakültesi",
            "İstanbul Atatürk Havalimanı Dış Hatlar Terminali",
            "Galata Kulesi civarı Karaköy Meydanı yakını",
            "34394 İstanbul Beşiktaş Levent Mahallesi Büyükdere Caddesi No:100",
            "İstanbul/Beşiktaş/Levent - Büyükdere Cad. : 100",
            "ISTANBUL BESIKTAS LEVENT BUYUKDERE CAD. NO:100",
            "istanbul beşiktaş levent büyükdere caddesi no:100",
        ]

    def create_baseline(self, config: PipelineConfig = None) -> Dict[str, Any]:
        """Create baseline results with default configuration."""

        if config is None:
            config = PipelineConfig(
                pattern_threshold_high=0.8,
                pattern_threshold_medium=0.6,
                pattern_threshold_low=0.4,
                ml_confidence_threshold=0.7,
                enable_validation=True,
                log_level="ERROR",
            )

        pipeline = create_pipeline(config)
        baseline = {}

        print(f"🎯 Creating baseline with {len(self.test_addresses)} addresses...")

        for i, address in enumerate(self.test_addresses):
            print(
                f"  [{i + 1:2d}/{len(self.test_addresses)}] "
                f"Processing: {address[:50]}..."
            )

            start_time = time.time()
            result = pipeline.process_single(address)
            processing_time = (time.time() - start_time) * 1000

            baseline[address] = {
                "success": result.success,
                "normalized_address": (
                    result.address_out.normalized_address
                    if result.address_out
                    else None
                ),
                "processing_method": result.processing_method,
                "confidence": result.confidence,
                "processing_time_ms": processing_time,
                "error": result.error,
                "components": (
                    result.address_out.to_dict() if result.address_out else None
                ),
            }

        self.baseline_results = baseline
        return baseline

    def test_threshold_variations(self) -> Dict[str, Any]:
        """Test various threshold configurations against baseline."""

        if not self.baseline_results:
            self.create_baseline()

        # Define threshold variations to test
        threshold_configs = [
            {"name": "high_precision", "high": 0.9, "medium": 0.8, "low": 0.6},
            {"name": "low_precision", "high": 0.6, "medium": 0.4, "low": 0.2},
            {"name": "medium_only", "high": 0.6, "medium": 0.6, "low": 0.6},
            {"name": "strict", "high": 0.95, "medium": 0.9, "low": 0.8},
            {"name": "permissive", "high": 0.5, "medium": 0.3, "low": 0.1},
        ]

        regression_results = {}

        for config_def in threshold_configs:
            config_name = config_def["name"]
            print(f"\n🔧 Testing configuration: {config_name}")

            # Create configuration
            config = PipelineConfig(
                pattern_threshold_high=config_def["high"],
                pattern_threshold_medium=config_def["medium"],
                pattern_threshold_low=config_def["low"],
                ml_confidence_threshold=0.7,
                enable_validation=True,
                log_level="ERROR",
            )

            pipeline = create_pipeline(config)
            config_results = {}

            for address in self.test_addresses:
                result = pipeline.process_single(address)

                config_results[address] = {
                    "success": result.success,
                    "normalized_address": (
                        result.address_out.normalized_address
                        if result.address_out
                        else None
                    ),
                    "processing_method": result.processing_method,
                    "confidence": result.confidence,
                    "error": result.error,
                    "components": (
                        result.address_out.to_dict() if result.address_out else None
                    ),
                }

            # Compare with baseline
            comparison = self._compare_with_baseline(config_results)

            regression_results[config_name] = {
                "config": config_def,
                "results": config_results,
                "comparison": comparison,
            }

            print(f"  Changes: {comparison['total_changes']}")
            print(f"  Method changes: {comparison['method_changes']}")
            print(f"  Output changes: {comparison['output_changes']}")

        return regression_results

    def _compare_with_baseline(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare test results with baseline."""

        comparison = {
            "total_changes": 0,
            "method_changes": 0,
            "output_changes": 0,
            "success_changes": 0,
            "confidence_changes": 0,
            "detailed_changes": [],
        }

        for address in self.test_addresses:
            baseline = self.baseline_results[address]
            test_result = test_results[address]

            changes = {}

            # Check success status change
            if baseline["success"] != test_result["success"]:
                changes["success"] = {
                    "baseline": baseline["success"],
                    "test": test_result["success"],
                }
                comparison["success_changes"] += 1

            # Check processing method change
            if baseline["processing_method"] != test_result["processing_method"]:
                changes["method"] = {
                    "baseline": baseline["processing_method"],
                    "test": test_result["processing_method"],
                }
                comparison["method_changes"] += 1

            # Check normalized address change
            if baseline["normalized_address"] != test_result["normalized_address"]:
                changes["output"] = {
                    "baseline": baseline["normalized_address"],
                    "test": test_result["normalized_address"],
                }
                comparison["output_changes"] += 1

            # Check significant confidence change (>0.1 difference)
            if (
                baseline["confidence"]
                and test_result["confidence"]
                and abs(baseline["confidence"] - test_result["confidence"]) > 0.1
            ):
                changes["confidence"] = {
                    "baseline": baseline["confidence"],
                    "test": test_result["confidence"],
                    "difference": test_result["confidence"] - baseline["confidence"],
                }
                comparison["confidence_changes"] += 1

            if changes:
                comparison["total_changes"] += 1
                comparison["detailed_changes"].append(
                    {"address": address, "changes": changes}
                )

        return comparison

    def test_ml_threshold_variations(self) -> Dict[str, Any]:
        """Test various ML confidence thresholds."""

        if not self.baseline_results:
            self.create_baseline()

        ml_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        ml_results = {}

        for threshold in ml_thresholds:
            print(f"\n🤖 Testing ML threshold: {threshold}")

            config = PipelineConfig(
                pattern_threshold_high=0.8,
                pattern_threshold_medium=0.6,
                pattern_threshold_low=0.4,
                ml_confidence_threshold=threshold,
                enable_validation=True,
                log_level="ERROR",
            )

            pipeline = create_pipeline(config)
            threshold_results = {}

            for address in self.test_addresses:
                result = pipeline.process_single(address)
                threshold_results[address] = {
                    "success": result.success,
                    "normalized_address": (
                        result.address_out.normalized_address
                        if result.address_out
                        else None
                    ),
                    "processing_method": result.processing_method,
                    "confidence": result.confidence,
                    "error": result.error,
                }

            comparison = self._compare_with_baseline(threshold_results)
            ml_results[f"ml_{threshold}"] = {
                "threshold": threshold,
                "results": threshold_results,
                "comparison": comparison,
            }

            print(f"  Changes: {comparison['total_changes']}")
            print(f"  Method changes: {comparison['method_changes']}")

        return ml_results

    def detect_performance_regressions(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect performance regressions by comparing processing times."""

        performance_issues = []

        for config_name, config_data in test_results.items():
            if "results" not in config_data:
                continue

            for address in self.test_addresses:
                baseline_time = self.baseline_results[address]["processing_time_ms"]
                test_time = config_data["results"][address].get("processing_time_ms", 0)

                if test_time > baseline_time * 2:  # 2x slower threshold
                    performance_issues.append(
                        {
                            "config": config_name,
                            "address": address,
                            "baseline_time_ms": baseline_time,
                            "test_time_ms": test_time,
                            "slowdown_factor": (
                                test_time / baseline_time
                                if baseline_time > 0
                                else float("inf")
                            ),
                        }
                    )

        return {
            "performance_regressions": performance_issues,
            "total_regressions": len(performance_issues),
        }

    def run_comprehensive_regression_tests(self) -> Dict[str, Any]:
        """Run all regression tests and compile comprehensive report."""

        print("🔄 Starting comprehensive regression testing...")

        # Create baseline
        baseline = self.create_baseline()

        # Test threshold variations
        threshold_results = self.test_threshold_variations()

        # Test ML threshold variations
        ml_results = self.test_ml_threshold_variations()

        # Detect performance regressions
        all_results = {**threshold_results, **ml_results}
        performance_analysis = self.detect_performance_regressions(all_results)

        # Compile summary
        summary = {
            "baseline_config": {
                "pattern_threshold_high": 0.8,
                "pattern_threshold_medium": 0.6,
                "pattern_threshold_low": 0.4,
                "ml_confidence_threshold": 0.7,
            },
            "test_addresses_count": len(self.test_addresses),
            "threshold_variations": len(threshold_results),
            "ml_variations": len(ml_results),
            "total_configurations_tested": len(all_results),
            "performance_regressions": performance_analysis["total_regressions"],
            "detailed_results": {
                "baseline": baseline,
                "threshold_tests": threshold_results,
                "ml_tests": ml_results,
                "performance_analysis": performance_analysis,
            },
        }

        # Calculate stability metrics
        total_changes = sum(
            result["comparison"]["total_changes"] for result in all_results.values()
        )

        method_changes = sum(
            result["comparison"]["method_changes"] for result in all_results.values()
        )

        output_changes = sum(
            result["comparison"]["output_changes"] for result in all_results.values()
        )

        summary["stability_metrics"] = {
            "avg_changes_per_config": total_changes / len(all_results),
            "avg_method_changes_per_config": method_changes / len(all_results),
            "avg_output_changes_per_config": output_changes / len(all_results),
            "total_test_runs": len(all_results) * len(self.test_addresses),
        }

        return summary


def test_regression_threshold_changes():
    """Pytest wrapper for regression testing."""

    runner = RegressionTestRunner()
    summary = runner.run_comprehensive_regression_tests()

    # Print summary
    print("\n📊 Regression Test Results:")
    print(f"   Test Addresses: {summary['test_addresses_count']}")
    print(f"   Configurations Tested: {summary['total_configurations_tested']}")
    print(f"   Performance Regressions: {summary['performance_regressions']}")

    # Stability metrics
    metrics = summary["stability_metrics"]
    print("\n📈 Stability Metrics:")
    print(f"   Avg Changes per Config: {metrics['avg_changes_per_config']:.1f}")
    print(f"   Avg Method Changes: {metrics['avg_method_changes_per_config']:.1f}")
    print(f"   Avg Output Changes: {metrics['avg_output_changes_per_config']:.1f}")

    # Save detailed results
    with open("regression_test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n💾 Detailed results saved to: regression_test_results.json")

    # Assert thresholds
    assert (
        summary["performance_regressions"] < 5
    ), f"Too many performance regressions: {summary['performance_regressions']}"
    assert (
        metrics["avg_changes_per_config"] < 10
    ), f"Too many changes per config: {metrics['avg_changes_per_config']}"


def test_deterministic_behavior():
    """Test that same input produces same output consistently."""

    config = PipelineConfig(log_level="ERROR")
    pipeline = create_pipeline(config)

    test_address = "İstanbul Beşiktaş Levent Mahallesi Büyükdere Caddesi No:100"

    # Run same address multiple times
    results = []
    for _ in range(5):
        result = pipeline.process_single(test_address)
        if result.success and result.address_out:
            results.append(result.address_out.normalized_address)
        else:
            results.append(None)

    # All results should be identical
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        assert (
            result == first_result
        ), f"Non-deterministic behavior: run {i + 1} differs from run 1"


def test_configuration_edge_cases():
    """Test edge case configurations."""

    edge_configs = [
        # Extremely high thresholds
        PipelineConfig(
            pattern_threshold_high=0.99,
            pattern_threshold_medium=0.95,
            pattern_threshold_low=0.9,
            ml_confidence_threshold=0.95,
        ),
        # Extremely low thresholds
        PipelineConfig(
            pattern_threshold_high=0.1,
            pattern_threshold_medium=0.05,
            pattern_threshold_low=0.01,
            ml_confidence_threshold=0.1,
        ),
        # Inverted thresholds (should be handled gracefully)
        PipelineConfig(
            pattern_threshold_high=0.4,
            pattern_threshold_medium=0.6,
            pattern_threshold_low=0.8,
            ml_confidence_threshold=0.7,
        ),
    ]

    test_address = "İstanbul Beşiktaş Levent"

    for i, config in enumerate(edge_configs):
        try:
            pipeline = create_pipeline(config)
            result = pipeline.process_single(test_address)

            # Should not crash
            assert result is not None

        except Exception as e:
            pytest.fail(f"Edge case config {i + 1} caused crash: {e}")


if __name__ == "__main__":
    # Run standalone regression tests
    runner = RegressionTestRunner()
    summary = runner.run_comprehensive_regression_tests()

    print("\n📊 Regression Test Summary:")
    print(f"Configurations Tested: {summary['total_configurations_tested']}")
    print(f"Performance Regressions: {summary['performance_regressions']}")
    avg_changes = summary["stability_metrics"]["avg_changes_per_config"]
    print(f"Avg Changes per Config: {avg_changes:.1f}")
