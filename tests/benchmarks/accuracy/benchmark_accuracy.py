"""
Accuracy Benchmark Suite for Address Normalization

Comprehensive accuracy testing including:
- Precision/Recall evaluation
- F1 score calculations
- Component-level accuracy
- Confidence calibration
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from addrnorm.preprocess.api import AddressNormalizer
from tests.fixtures.address_samples import get_accuracy_test_data, get_edge_case_data, AddressSample


@dataclass
class ComponentAccuracy:
    """Accuracy metrics for a specific address component"""

    component: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_predictions: int = 0
    total_ground_truth: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class AccuracyResult:
    """Accuracy evaluation result"""

    test_name: str
    overall_accuracy: float
    component_accuracies: Dict[str, ComponentAccuracy]
    confidence_calibration: Dict[str, float]
    sample_size: int
    success_rate: float
    error_analysis: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracySuite:
    """Complete accuracy evaluation suite"""

    timestamp: str
    results: List[AccuracyResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class AccuracyBenchmark:
    """Accuracy benchmark runner"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.normalizer = None

        # Component mapping for evaluation
        self.component_mapping = {
            "il": ["province", "city", "il"],
            "ilce": ["district", "ilce", "county"],
            "mahalle": ["neighborhood", "mahalle", "quarter"],
            "yol": ["street", "road", "avenue", "boulevard", "sokak", "cadde"],
            "bina_no": ["building_number", "house_number", "bina_no"],
            "daire_no": ["apartment_number", "flat_number", "daire_no"],
            "posta_kodu": ["postal_code", "zip_code", "posta_kodu"],
        }

    def setup(self):
        """Setup accuracy benchmark environment"""
        self.normalizer = AddressNormalizer()

    def evaluate_component_accuracy(self, test_samples: List[AddressSample]) -> Dict[str, ComponentAccuracy]:
        """Evaluate accuracy for each address component"""
        component_metrics = {}

        for component in self.component_mapping:
            component_metrics[component] = ComponentAccuracy(component=component)

        for sample in test_samples:
            try:
                # Process address
                result = self.normalizer.process(sample.input_address)

                if not result.success:
                    # Count all expected components as false negatives
                    for component, expected_value in sample.expected_components.items():
                        if component in component_metrics and expected_value:
                            component_metrics[component].false_negatives += 1
                    continue

                # Extract normalized components
                normalized_components = self._extract_components(result)

                # Evaluate each component
                for component in self.component_mapping:
                    expected_value = sample.expected_components.get(component, "").strip().lower()
                    predicted_value = normalized_components.get(component, "").strip().lower()

                    metrics = component_metrics[component]

                    # Update totals
                    if expected_value:
                        metrics.total_ground_truth += 1
                    if predicted_value:
                        metrics.total_predictions += 1

                    # Calculate confusion matrix elements
                    if expected_value and predicted_value:
                        if self._components_match(expected_value, predicted_value):
                            metrics.true_positives += 1
                        else:
                            metrics.false_positives += 1
                            metrics.false_negatives += 1
                    elif expected_value and not predicted_value:
                        metrics.false_negatives += 1
                    elif not expected_value and predicted_value:
                        metrics.false_positives += 1
                    # True negative case: both empty (not counted explicitly)

            except Exception as e:
                # Count as false negatives for all expected components
                for component, expected_value in sample.expected_components.items():
                    if component in component_metrics and expected_value:
                        component_metrics[component].false_negatives += 1

        return component_metrics

    def _extract_components(self, result) -> Dict[str, str]:
        """Extract components from normalization result"""
        components = {}

        # This is a simplified extraction - adjust based on actual result structure
        if hasattr(result, "normalized_address"):
            addr = result.normalized_address
            if hasattr(addr, "__dict__"):
                for key, value in addr.__dict__.items():
                    components[key] = str(value) if value else ""

        # If result has components directly
        if hasattr(result, "components"):
            for key, value in result.components.items():
                components[key] = str(value) if value else ""

        return components

    def _components_match(self, expected: str, predicted: str) -> bool:
        """Check if two component values match (with fuzzy matching)"""
        if not expected or not predicted:
            return False

        expected = expected.lower().strip()
        predicted = predicted.lower().strip()

        # Exact match
        if expected == predicted:
            return True

        # Partial match for certain components
        if len(expected) > 3 and len(predicted) > 3:
            # Check if one is contained in the other
            if expected in predicted or predicted in expected:
                return True

        return False

    def evaluate_confidence_calibration(self, test_samples: List[AddressSample]) -> Dict[str, float]:
        """Evaluate confidence score calibration"""
        confidence_buckets = defaultdict(list)

        for sample in test_samples:
            try:
                result = self.normalizer.process(sample.input_address)

                # Get confidence score
                confidence = getattr(result, "confidence", 0.0)

                # Determine if prediction is correct
                is_correct = self._is_prediction_correct(result, sample)

                # Bucket by confidence ranges
                bucket = self._get_confidence_bucket(confidence)
                confidence_buckets[bucket].append(is_correct)

            except Exception:
                # Low confidence bucket for errors
                confidence_buckets["0.0-0.1"].append(False)

        # Calculate calibration metrics
        calibration_metrics = {}
        for bucket, predictions in confidence_buckets.items():
            if predictions:
                accuracy = sum(predictions) / len(predictions)
                calibration_metrics[bucket] = accuracy
            else:
                calibration_metrics[bucket] = 0.0

        return calibration_metrics

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket label"""
        if confidence < 0.1:
            return "0.0-0.1"
        elif confidence < 0.3:
            return "0.1-0.3"
        elif confidence < 0.5:
            return "0.3-0.5"
        elif confidence < 0.7:
            return "0.5-0.7"
        elif confidence < 0.9:
            return "0.7-0.9"
        else:
            return "0.9-1.0"

    def _is_prediction_correct(self, result, sample: AddressSample) -> bool:
        """Determine if overall prediction is correct"""
        if not result.success:
            return False

        # Extract and compare components
        predicted_components = self._extract_components(result)
        correct_components = 0
        total_components = 0

        for component, expected_value in sample.expected_components.items():
            if expected_value:  # Only check non-empty expected values
                total_components += 1
                predicted_value = predicted_components.get(component, "")
                if self._components_match(expected_value, predicted_value):
                    correct_components += 1

        # Consider correct if 70% or more components match
        if total_components == 0:
            return True  # No components to check

        return (correct_components / total_components) >= 0.7

    def analyze_errors(self, test_samples: List[AddressSample]) -> Dict[str, Any]:
        """Analyze common error patterns"""
        error_patterns = defaultdict(int)
        component_errors = defaultdict(lambda: defaultdict(int))
        confidence_errors = []

        for sample in test_samples:
            try:
                result = self.normalizer.process(sample.input_address)

                if not result.success:
                    error_patterns["processing_failed"] += 1
                    continue

                # Analyze component-level errors
                predicted_components = self._extract_components(result)

                for component, expected_value in sample.expected_components.items():
                    if expected_value:
                        predicted_value = predicted_components.get(component, "")
                        if not self._components_match(expected_value, predicted_value):
                            component_errors[component]["mismatch"] += 1

                            # Categorize error type
                            if not predicted_value:
                                component_errors[component]["missing"] += 1
                            elif len(predicted_value) < len(expected_value) * 0.5:
                                component_errors[component]["too_short"] += 1
                            else:
                                component_errors[component]["wrong_value"] += 1

                # Track confidence for incorrect predictions
                if not self._is_prediction_correct(result, sample):
                    confidence = getattr(result, "confidence", 0.0)
                    confidence_errors.append(confidence)

            except Exception as e:
                error_patterns["exception"] += 1

        # Calculate error statistics
        return {
            "error_patterns": dict(error_patterns),
            "component_errors": {k: dict(v) for k, v in component_errors.items()},
            "high_confidence_errors": len([c for c in confidence_errors if c > 0.8]),
            "avg_error_confidence": statistics.mean(confidence_errors) if confidence_errors else 0.0,
            "total_errors": len(confidence_errors),
        }

    def run_standard_accuracy_test(self, sample_size: int = 500) -> AccuracyResult:
        """Run standard accuracy evaluation"""
        test_samples = get_accuracy_test_data(sample_size)

        # Component accuracy
        component_accuracies = self.evaluate_component_accuracy(test_samples)

        # Overall accuracy
        correct_predictions = 0
        total_predictions = len(test_samples)
        successful_predictions = 0

        for sample in test_samples:
            try:
                result = self.normalizer.process(sample.input_address)
                if result.success:
                    successful_predictions += 1
                    if self._is_prediction_correct(result, sample):
                        correct_predictions += 1
            except Exception:
                pass

        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        success_rate = successful_predictions / total_predictions if total_predictions > 0 else 0.0

        # Confidence calibration
        confidence_calibration = self.evaluate_confidence_calibration(test_samples)

        # Error analysis
        error_analysis = self.analyze_errors(test_samples)

        return AccuracyResult(
            test_name="standard_accuracy_test",
            overall_accuracy=overall_accuracy,
            component_accuracies=component_accuracies,
            confidence_calibration=confidence_calibration,
            sample_size=sample_size,
            success_rate=success_rate,
            error_analysis=error_analysis,
            details={
                "correct_predictions": correct_predictions,
                "successful_predictions": successful_predictions,
                "avg_component_f1": statistics.mean([acc.f1_score for acc in component_accuracies.values()]),
                "best_component": max(component_accuracies.keys(), key=lambda k: component_accuracies[k].f1_score),
                "worst_component": min(component_accuracies.keys(), key=lambda k: component_accuracies[k].f1_score),
            },
        )

    def run_edge_case_accuracy_test(self) -> AccuracyResult:
        """Run accuracy test on edge cases"""
        edge_cases = get_edge_case_data()

        processed_count = 0
        error_count = 0
        graceful_failures = 0

        for sample in edge_cases:
            try:
                result = self.normalizer.process(sample.input_address)
                processed_count += 1

                if not result.success:
                    graceful_failures += 1

            except Exception as e:
                error_count += 1

        # For edge cases, we mainly check robustness
        robustness_score = (processed_count - error_count) / len(edge_cases) if edge_cases else 1.0
        graceful_failure_rate = graceful_failures / processed_count if processed_count > 0 else 0.0

        return AccuracyResult(
            test_name="edge_case_accuracy_test",
            overall_accuracy=robustness_score,
            component_accuracies={},  # Not applicable for edge cases
            confidence_calibration={},
            sample_size=len(edge_cases),
            success_rate=1.0 - (error_count / len(edge_cases)) if edge_cases else 1.0,
            error_analysis={
                "total_edge_cases": len(edge_cases),
                "processing_errors": error_count,
                "graceful_failures": graceful_failures,
                "robustness_score": robustness_score,
                "graceful_failure_rate": graceful_failure_rate,
            },
            details={"edge_case_categories": Counter(sample.category for sample in edge_cases)},
        )

    def run_category_accuracy_test(self) -> List[AccuracyResult]:
        """Run accuracy tests for different address categories"""
        categories = ["simple", "complex", "multilingual", "edge_cases"]
        results = []

        for category in categories:
            # Get category-specific test data
            # This would need to be implemented based on available test data
            # For now, use a subset of standard test data
            test_samples = get_accuracy_test_data(100)  # Simplified

            # Filter by category if possible
            category_samples = [s for s in test_samples if getattr(s, "category", "standard") == category]

            if not category_samples:
                continue

            # Run accuracy evaluation
            component_accuracies = self.evaluate_component_accuracy(category_samples)
            confidence_calibration = self.evaluate_confidence_calibration(category_samples)
            error_analysis = self.analyze_errors(category_samples)

            # Calculate overall accuracy
            correct_predictions = 0
            successful_predictions = 0

            for sample in category_samples:
                try:
                    result = self.normalizer.process(sample.input_address)
                    if result.success:
                        successful_predictions += 1
                        if self._is_prediction_correct(result, sample):
                            correct_predictions += 1
                except Exception:
                    pass

            overall_accuracy = correct_predictions / len(category_samples) if category_samples else 0.0
            success_rate = successful_predictions / len(category_samples) if category_samples else 0.0

            result = AccuracyResult(
                test_name=f"category_accuracy_test_{category}",
                overall_accuracy=overall_accuracy,
                component_accuracies=component_accuracies,
                confidence_calibration=confidence_calibration,
                sample_size=len(category_samples),
                success_rate=success_rate,
                error_analysis=error_analysis,
                details={
                    "category": category,
                    "correct_predictions": correct_predictions,
                    "successful_predictions": successful_predictions,
                },
            )

            results.append(result)

        return results

    def run_all_accuracy_tests(self, sample_size: int = 500) -> AccuracySuite:
        """Run complete accuracy test suite"""
        self.setup()

        suite = AccuracySuite(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

        print("Running standard accuracy test...")
        result = self.run_standard_accuracy_test(sample_size)
        suite.results.append(result)

        print("Running edge case accuracy test...")
        result = self.run_edge_case_accuracy_test()
        suite.results.append(result)

        print("Running category-specific accuracy tests...")
        category_results = self.run_category_accuracy_test()
        suite.results.extend(category_results)

        # Generate summary
        suite.summary = self._generate_accuracy_summary(suite.results)

        return suite

    def _generate_accuracy_summary(self, results: List[AccuracyResult]) -> Dict[str, Any]:
        """Generate accuracy summary statistics"""
        if not results:
            return {}

        # Overall metrics
        overall_accuracies = [r.overall_accuracy for r in results if r.overall_accuracy > 0]
        success_rates = [r.success_rate for r in results if r.success_rate > 0]

        # Component metrics
        all_component_f1s = []
        component_performance = defaultdict(list)

        for result in results:
            if result.component_accuracies:
                for component, accuracy in result.component_accuracies.items():
                    f1 = accuracy.f1_score
                    all_component_f1s.append(f1)
                    component_performance[component].append(f1)

        # Best and worst performing components
        avg_component_performance = {
            component: statistics.mean(scores) for component, scores in component_performance.items() if scores
        }

        best_component = (
            max(avg_component_performance.keys(), key=lambda k: avg_component_performance[k])
            if avg_component_performance
            else None
        )
        worst_component = (
            min(avg_component_performance.keys(), key=lambda k: avg_component_performance[k])
            if avg_component_performance
            else None
        )

        return {
            "avg_overall_accuracy": statistics.mean(overall_accuracies) if overall_accuracies else 0.0,
            "max_overall_accuracy": max(overall_accuracies) if overall_accuracies else 0.0,
            "min_overall_accuracy": min(overall_accuracies) if overall_accuracies else 0.0,
            "avg_success_rate": statistics.mean(success_rates) if success_rates else 0.0,
            "avg_component_f1": statistics.mean(all_component_f1s) if all_component_f1s else 0.0,
            "best_performing_component": best_component,
            "worst_performing_component": worst_component,
            "component_avg_performance": avg_component_performance,
            "total_samples_tested": sum(r.sample_size for r in results),
            "recommendations": self._generate_accuracy_recommendations(results),
        }

    def _generate_accuracy_recommendations(self, results: List[AccuracyResult]) -> List[str]:
        """Generate recommendations based on accuracy results"""
        recommendations = []

        # Analyze overall accuracy
        overall_accuracies = [r.overall_accuracy for r in results if r.overall_accuracy > 0]
        if overall_accuracies:
            avg_accuracy = statistics.mean(overall_accuracies)
            if avg_accuracy < 0.8:
                recommendations.append("Overall accuracy is below 80% - consider model retraining")
            elif avg_accuracy < 0.9:
                recommendations.append("Overall accuracy could be improved - review component extraction logic")

        # Analyze component performance
        component_issues = []
        for result in results:
            if result.component_accuracies:
                for component, accuracy in result.component_accuracies.items():
                    if accuracy.f1_score < 0.7:
                        component_issues.append(component)

        if component_issues:
            most_common_issue = Counter(component_issues).most_common(1)[0][0]
            recommendations.append(f"Focus on improving {most_common_issue} extraction accuracy")

        # Analyze confidence calibration
        for result in results:
            if result.confidence_calibration:
                high_conf_buckets = [
                    k for k, v in result.confidence_calibration.items() if k.startswith("0.9") or k.startswith("0.8")
                ]
                for bucket in high_conf_buckets:
                    if result.confidence_calibration[bucket] < 0.9:
                        recommendations.append(
                            "High confidence predictions have lower accuracy - review confidence calculation"
                        )
                        break

        return recommendations

    def save_results(self, suite: AccuracySuite, filename: str = None) -> Path:
        """Save accuracy results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_benchmark_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert to dict for JSON serialization
        suite_dict = {
            "timestamp": suite.timestamp,
            "results": [
                {
                    "test_name": r.test_name,
                    "overall_accuracy": r.overall_accuracy,
                    "component_accuracies": {
                        comp: {
                            "precision": acc.precision,
                            "recall": acc.recall,
                            "f1_score": acc.f1_score,
                            "true_positives": acc.true_positives,
                            "false_positives": acc.false_positives,
                            "false_negatives": acc.false_negatives,
                        }
                        for comp, acc in r.component_accuracies.items()
                    },
                    "confidence_calibration": r.confidence_calibration,
                    "sample_size": r.sample_size,
                    "success_rate": r.success_rate,
                    "error_analysis": r.error_analysis,
                    "details": r.details,
                }
                for r in suite.results
            ],
            "summary": suite.summary,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(suite_dict, f, indent=2, ensure_ascii=False)

        return output_path


def run_accuracy_benchmark(sample_size: int = 500, output_dir: str = "benchmark_results") -> Path:
    """Convenience function to run accuracy benchmark"""
    benchmark = AccuracyBenchmark(output_dir)
    suite = benchmark.run_all_accuracy_tests(sample_size)
    return benchmark.save_results(suite)


if __name__ == "__main__":
    # Example usage
    print("Running accuracy benchmark suite...")
    result_path = run_accuracy_benchmark(500)
    print(f"Results saved to: {result_path}")
