"""
Test configuration and utilities for the address normalization test suite
"""

import sys
import os
from pathlib import Path
import pytest
import logging
from typing import Any, Dict, List, Optional

# Add src to path for imports
test_root = Path(__file__).parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Test configuration
TEST_CONFIG = {
    "timeout": 30,  # seconds
    "performance_threshold": 200,  # ms
    "confidence_threshold": 0.5,
    "memory_limit_mb": 500,
    "parallel_workers": 4,
}

# Pytest fixtures and markers
pytest_plugins = []  # Remove benchmark plugin for now


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "accuracy: Accuracy evaluation tests")


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def sample_addresses():
    """Load address samples for testing"""
    from tests.fixtures.address_samples import get_all_samples

    return get_all_samples()


@pytest.fixture(scope="session")
def performance_addresses():
    """Load performance test addresses"""
    from tests.fixtures.address_samples import get_performance_test_samples

    return get_performance_test_samples(1000)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


class TestMetrics:
    """Collect and track test metrics"""

    def __init__(self):
        self.metrics = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_results": [],
            "accuracy_results": [],
            "memory_usage": [],
        }

    def record_test_result(self, passed: bool, duration: float = 0.0):
        """Record test result"""
        self.metrics["tests_run"] += 1
        if passed:
            self.metrics["tests_passed"] += 1
        else:
            self.metrics["tests_failed"] += 1

    def record_performance(self, operation: str, duration: float, throughput: float = 0.0):
        """Record performance metric"""
        self.metrics["performance_results"].append(
            {"operation": operation, "duration_ms": duration * 1000, "throughput": throughput}
        )

    def record_accuracy(self, test_type: str, accuracy: float, confidence: float = 0.0):
        """Record accuracy metric"""
        self.metrics["accuracy_results"].append({"test_type": test_type, "accuracy": accuracy, "confidence": confidence})

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        pass_rate = self.metrics["tests_passed"] / max(self.metrics["tests_run"], 1)

        avg_performance = 0.0
        if self.metrics["performance_results"]:
            avg_performance = sum(r["duration_ms"] for r in self.metrics["performance_results"]) / len(
                self.metrics["performance_results"]
            )

        avg_accuracy = 0.0
        if self.metrics["accuracy_results"]:
            avg_accuracy = sum(r["accuracy"] for r in self.metrics["accuracy_results"]) / len(self.metrics["accuracy_results"])

        return {
            "pass_rate": pass_rate,
            "total_tests": self.metrics["tests_run"],
            "avg_performance_ms": avg_performance,
            "avg_accuracy": avg_accuracy,
            "metrics": self.metrics,
        }


# Global test metrics instance
test_metrics = TestMetrics()


@pytest.fixture
def metrics():
    """Test metrics fixture"""
    return test_metrics


def assert_performance(duration: float, threshold: float = None):
    """Assert performance is within threshold"""
    threshold = threshold or TEST_CONFIG["performance_threshold"] / 1000.0
    assert duration <= threshold, f"Performance test failed: {duration:.3f}s > {threshold:.3f}s"


def assert_accuracy(actual: float, expected: float, tolerance: float = 0.1):
    """Assert accuracy is within tolerance"""
    diff = abs(actual - expected)
    assert diff <= tolerance, f"Accuracy test failed: {actual:.3f} vs {expected:.3f} (diff: {diff:.3f})"


def assert_confidence(confidence: float, min_threshold: float = None):
    """Assert confidence is above threshold"""
    threshold = min_threshold or TEST_CONFIG["confidence_threshold"]
    assert confidence >= threshold, f"Confidence test failed: {confidence:.3f} < {threshold:.3f}"


class MockAddressNormalizer:
    """Mock normalizer for testing"""

    def __init__(self, success_rate: float = 0.8, avg_confidence: float = 0.7):
        self.success_rate = success_rate
        self.avg_confidence = avg_confidence
        self.call_count = 0

    def normalize(self, address: str) -> Dict[str, Any]:
        """Mock normalization"""
        self.call_count += 1

        import random

        success = random.random() < self.success_rate
        confidence = self.avg_confidence + random.uniform(-0.2, 0.2)
        confidence = max(0.0, min(1.0, confidence))

        if success:
            return {
                "success": True,
                "confidence": confidence,
                "components": {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda"},
                "processed_address": address.title(),
            }
        else:
            return {"success": False, "confidence": confidence * 0.5, "components": {}, "error": "Normalization failed"}
