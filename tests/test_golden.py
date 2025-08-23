"""
Golden test runner for Turkish address normalization.
Tests the pipeline against 50 golden examples.
"""

import json
import time
from typing import Any, Dict

from addrnorm.pipeline import PipelineConfig, create_pipeline
from tests.golden.test_cases import GOLDEN_TEST_CASES


class GoldenTestRunner:
    """Runner for golden tests with detailed reporting."""

    def __init__(self):
        """Initialize the test runner."""
        # Create pipeline with default config
        config = PipelineConfig(
            enable_validation=True,
            log_level="WARNING",  # Reduce noise during testing
        )
        self.pipeline = create_pipeline(config)
        self.results = []

    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single golden test case."""

        test_id = test_case["id"]
        input_address = test_case["input"]
        expected = test_case["expected"]

        start_time = time.time()

        try:
            # Process the address
            result = self.pipeline.process_single(input_address)
            processing_time = (time.time() - start_time) * 1000  # ms

            if not result.success:
                return {
                    "test_id": test_id,
                    "status": "FAILED",
                    "error": result.error,
                    "input": input_address,
                    "expected": expected,
                    "actual": None,
                    "processing_time_ms": processing_time,
                    "category": test_case["category"],
                }

            # Extract actual values
            actual = {}
            if result.address_out:
                actual = result.address_out.to_dict()

            # Compare with expected
            matches = self._compare_address(expected, actual)

            return {
                "test_id": test_id,
                "status": "PASSED" if matches["overall"] else "FAILED",
                "input": input_address,
                "expected": expected,
                "actual": actual,
                "matches": matches,
                "processing_time_ms": processing_time,
                "processing_method": result.processing_method,
                "confidence": result.confidence,
                "category": test_case["category"],
                "description": test_case["description"],
            }

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                "test_id": test_id,
                "status": "ERROR",
                "error": str(e),
                "input": input_address,
                "expected": expected,
                "actual": None,
                "processing_time_ms": processing_time,
                "category": test_case["category"],
            }

    def _compare_address(self, expected: Dict, actual: Dict) -> Dict[str, Any]:
        """Compare expected vs actual address components."""

        matches = {}
        critical_fields = ["city", "district", "neighborhood", "street", "number"]
        optional_fields = [
            "building",
            "block",
            "floor",
            "apartment",
            "postcode",
            "country",
        ]

        critical_matches = 0
        critical_total = 0

        # Check critical fields
        for field in critical_fields:
            if field in expected:
                critical_total += 1
                expected_val = self._normalize_value(expected[field])
                actual_val = self._normalize_value(actual.get(field))

                match = expected_val == actual_val
                matches[field] = {
                    "expected": expected_val,
                    "actual": actual_val,
                    "match": match,
                }

                if match:
                    critical_matches += 1

        # Check optional fields
        optional_matches = 0
        optional_total = 0

        for field in optional_fields:
            if field in expected:
                optional_total += 1
                expected_val = self._normalize_value(expected[field])
                actual_val = self._normalize_value(actual.get(field))

                match = expected_val == actual_val
                matches[field] = {
                    "expected": expected_val,
                    "actual": actual_val,
                    "match": match,
                }

                if match:
                    optional_matches += 1

        # Calculate scores
        critical_score = (
            critical_matches / critical_total if critical_total > 0 else 1.0
        )
        optional_score = (
            optional_matches / optional_total if optional_total > 0 else 1.0
        )

        # Overall pass if critical fields ≥ 80% and optional ≥ 60%
        overall_pass = critical_score >= 0.8 and optional_score >= 0.6

        matches["scores"] = {
            "critical": critical_score,
            "optional": optional_score,
            "critical_matches": critical_matches,
            "critical_total": critical_total,
            "optional_matches": optional_matches,
            "optional_total": optional_total,
        }
        matches["overall"] = overall_pass

        return matches

    def _normalize_value(self, value: Any) -> str:
        """Normalize value for comparison."""
        if value is None:
            return ""
        return str(value).strip().lower()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all golden tests and return comprehensive results."""

        print(f"🧪 Running {len(GOLDEN_TEST_CASES)} golden tests...")

        self.results = []
        category_stats = {}

        for i, test_case in enumerate(GOLDEN_TEST_CASES):
            print(
                f"  [{i + 1:2d}/{len(GOLDEN_TEST_CASES)}] {test_case['id']}: "
                f"{test_case['description'][:50]}..."
            )

            result = self.run_single_test(test_case)
            self.results.append(result)

            # Track category statistics
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                }

            category_stats[category]["total"] += 1
            if result["status"] == "PASSED":
                category_stats[category]["passed"] += 1
            elif result["status"] == "FAILED":
                category_stats[category]["failed"] += 1
            else:
                category_stats[category]["errors"] += 1

        # Calculate overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.results if r["status"] == "FAILED")
        error_tests = sum(1 for r in self.results if r["status"] == "ERROR")

        avg_processing_time = (
            sum(r["processing_time_ms"] for r in self.results) / total_tests
        )

        # Method distribution
        method_stats = {}
        for result in self.results:
            if result["status"] == "PASSED" and "processing_method" in result:
                method = result["processing_method"]
                method_stats[method] = method_stats.get(method, 0) + 1

        # Confidence distribution
        confidences = [r["confidence"] for r in self.results if "confidence" in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": passed_tests / total_tests,
            "avg_processing_time_ms": avg_processing_time,
            "avg_confidence": avg_confidence,
            "category_stats": category_stats,
            "method_distribution": method_stats,
            "detailed_results": self.results,
        }

        return summary


def test_golden_examples():
    """Pytest wrapper for golden tests."""
    runner = GoldenTestRunner()
    summary = runner.run_all_tests()

    # Print summary
    print("\n📊 Golden Test Results:")
    print(f"   Total: {summary['total_tests']}")
    print(f"   Passed: {summary['passed']} ({summary['success_rate']:.1%})")
    print(f"   Failed: {summary['failed']}")
    print(f"   Errors: {summary['errors']}")
    print(f"   Avg Time: {summary['avg_processing_time_ms']:.1f}ms")
    print(f"   Avg Confidence: {summary['avg_confidence']:.3f}")

    # Category breakdown
    print("\n📋 By Category:")
    for category, stats in summary["category_stats"].items():
        rate = stats["passed"] / stats["total"]
        print(
            f"   {category:15s}: {stats['passed']:2d}/{stats['total']:2d} ({rate:.1%})"
        )

    # Method distribution
    print("\n🔧 Processing Methods:")
    for method, count in summary["method_distribution"].items():
        print(f"   {method:15s}: {count:2d}")

    # Save detailed results
    with open("golden_test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n💾 Detailed results saved to: golden_test_results.json")

    # Assert success rate
    assert (
        summary["success_rate"] >= 0.80
    ), f"Golden test success rate {summary['success_rate']:.1%} below 80%"


if __name__ == "__main__":
    # Run standalone
    runner = GoldenTestRunner()
    summary = runner.run_all_tests()

    print("\n📊 Golden Test Summary:")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Failed Tests: {summary['failed']}")

    if summary["failed"] > 0:
        print("\n❌ Failed Tests:")
        for result in summary["detailed_results"]:
            if result["status"] == "FAILED":
                print(f"  {result['test_id']}: {result['description']}")
                if "matches" in result:
                    critical_score = result["matches"]["scores"]["critical"]
                    optional_score = result["matches"]["scores"]["optional"]
                    print(f"    Critical Score: {critical_score:.1%}")
                    print(f"    Optional Score: {optional_score:.1%}")
