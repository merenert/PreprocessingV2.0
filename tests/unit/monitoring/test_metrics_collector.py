"""
Unit tests for the monitoring metrics collector
"""

import pytest
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src to path
test_root = Path(__file__).parent.parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.monitoring.metrics_collector import (
    MetricsCollector,
    MetricEvent,
    MetricType,
    ProcessingMethod,
    PerformanceMetrics,
    PatternMetrics,
    SystemMetrics,
)


class TestMetricType:
    """Test MetricType enum"""

    def test_metric_type_values(self):
        """Test MetricType enum values"""
        assert MetricType.PERFORMANCE.value == "performance"
        assert MetricType.PATTERN_USAGE.value == "pattern_usage"
        assert MetricType.SUCCESS_RATE.value == "success_rate"
        assert MetricType.CONFIDENCE.value == "confidence"


class TestProcessingMethod:
    """Test ProcessingMethod enum"""

    def test_processing_method_values(self):
        """Test ProcessingMethod enum values"""
        assert ProcessingMethod.PATTERN_PRIMARY.value == "pattern_primary"
        assert ProcessingMethod.ML_PRIMARY.value == "ml_primary"
        assert ProcessingMethod.HYBRID.value == "hybrid"
        assert ProcessingMethod.FALLBACK.value == "fallback"


class TestMetricEvent:
    """Test MetricEvent class"""

    def test_metric_event_creation(self):
        """Test creating MetricEvent"""
        event = MetricEvent(
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
            method=ProcessingMethod.PATTERN_PRIMARY,
            pattern_id="pattern_123",
            success=True,
            confidence=0.85,
            processing_time_ms=150.5,
        )

        assert event.metric_type == MetricType.PERFORMANCE
        assert event.method == ProcessingMethod.PATTERN_PRIMARY
        assert event.pattern_id == "pattern_123"
        assert event.success is True
        assert event.confidence == 0.85
        assert event.processing_time_ms == 150.5

    def test_metric_event_defaults(self):
        """Test MetricEvent with default values"""
        event = MetricEvent(
            timestamp=datetime.now(), metric_type=MetricType.PERFORMANCE, method=ProcessingMethod.PATTERN_PRIMARY
        )

        assert event.success is True
        assert event.confidence == 0.0
        assert event.processing_time_ms == 0.0
        assert event.address_length == 0
        assert event.components_extracted == 0

    def test_metric_event_to_dict(self):
        """Test MetricEvent to_dict method"""
        timestamp = datetime.now()
        event = MetricEvent(
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            method=ProcessingMethod.PATTERN_PRIMARY,
            success=True,
            confidence=0.85,
        )

        result = event.to_dict()

        expected_keys = [
            "timestamp",
            "metric_type",
            "method",
            "pattern_id",
            "success",
            "confidence",
            "processing_time_ms",
            "address_length",
            "components_extracted",
            "error_type",
            "metadata",
        ]

        for key in expected_keys:
            assert key in result

        assert result["timestamp"] == timestamp.isoformat()
        assert result["metric_type"] == MetricType.PERFORMANCE.value
        assert result["method"] == ProcessingMethod.PATTERN_PRIMARY.value


class TestPerformanceMetrics:
    """Test PerformanceMetrics class"""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics"""
        metrics = PerformanceMetrics(
            avg_processing_time_ms=125.5,
            median_processing_time_ms=120.0,
            p95_processing_time_ms=200.0,
            throughput_per_second=50.0,
            total_processed=1000,
            error_rate=0.05,
        )

        assert metrics.avg_processing_time_ms == 125.5
        assert metrics.median_processing_time_ms == 120.0
        assert metrics.p95_processing_time_ms == 200.0
        assert metrics.throughput_per_second == 50.0
        assert metrics.total_processed == 1000
        assert metrics.error_rate == 0.05

    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics to_dict method"""
        metrics = PerformanceMetrics(avg_processing_time_ms=125.5, throughput_per_second=50.0, total_processed=1000)

        result = metrics.to_dict()

        expected_keys = [
            "avg_processing_time_ms",
            "median_processing_time_ms",
            "p95_processing_time_ms",
            "p99_processing_time_ms",
            "throughput_per_second",
            "total_processed",
            "error_rate",
            "memory_usage_mb",
        ]

        for key in expected_keys:
            assert key in result

        assert result["avg_processing_time_ms"] == 125.5
        assert result["total_processed"] == 1000


class TestMetricsCollector:
    """Test MetricsCollector class"""

    def setup_method(self):
        """Setup for each test method"""
        self.collector = MetricsCollector(buffer_size=100, aggregation_interval_seconds=1, retention_days=1)

    def test_collector_initialization(self):
        """Test collector initialization"""
        assert self.collector is not None
        assert self.collector.buffer_size == 100
        assert self.collector.aggregation_interval == 1
        assert self.collector.retention_days == 1

    def test_record_event(self):
        """Test recording events"""
        event = MetricEvent(
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
            method=ProcessingMethod.PATTERN_PRIMARY,
            success=True,
            confidence=0.85,
        )

        # Record event
        self.collector.record_event(event)

        # Check event was recorded
        assert len(self.collector._events) == 1
        recorded_event = list(self.collector._events)[0]
        assert recorded_event.confidence == 0.85

    def test_record_multiple_events(self):
        """Test recording multiple events"""
        events = []
        for i in range(10):
            event = MetricEvent(
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
                method=ProcessingMethod.PATTERN_PRIMARY,
                confidence=0.8 + (i * 0.01),
            )
            events.append(event)
            self.collector.record_event(event)

        assert len(self.collector._events) == 10

    def test_buffer_overflow(self):
        """Test buffer overflow handling"""
        # Create collector with small buffer
        small_collector = MetricsCollector(buffer_size=5)

        # Add more events than buffer size
        for i in range(10):
            event = MetricEvent(
                timestamp=datetime.now(), metric_type=MetricType.PERFORMANCE, method=ProcessingMethod.PATTERN_PRIMARY
            )
            small_collector.record_event(event)

        # Buffer should not exceed max size
        assert len(small_collector._events) <= 5

    def test_get_performance_metrics(self):
        """Test getting performance metrics"""
        # Add some test events
        for i in range(5):
            event = MetricEvent(
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
                method=ProcessingMethod.PATTERN_PRIMARY,
                processing_time_ms=100 + i * 10,
                success=i % 2 == 0,  # Alternate success/failure
            )
            self.collector.record_event(event)

        metrics = self.collector.get_performance_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.avg_processing_time_ms > 0
        assert metrics.total_processed == 5
        assert 0.0 <= metrics.error_rate <= 1.0

    def test_get_system_metrics(self):
        """Test getting system metrics"""
        # Add events with different methods
        methods = [
            ProcessingMethod.PATTERN_PRIMARY,
            ProcessingMethod.ML_PRIMARY,
            ProcessingMethod.HYBRID,
            ProcessingMethod.FALLBACK,
        ]

        for method in methods:
            event = MetricEvent(timestamp=datetime.now(), metric_type=MetricType.PERFORMANCE, method=method)
            self.collector.record_event(event)

        metrics = self.collector.get_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert len(metrics.method_distribution) > 0

        # Check all methods are represented
        for method in methods:
            assert method.value in metrics.method_distribution

    def test_aggregation_start_stop(self):
        """Test starting and stopping aggregation"""
        # Start aggregation
        self.collector.start_aggregation()

        # Add some events
        for i in range(3):
            event = MetricEvent(
                timestamp=datetime.now(), metric_type=MetricType.PERFORMANCE, method=ProcessingMethod.PATTERN_PRIMARY
            )
            self.collector.record_event(event)

        # Wait briefly for aggregation
        time.sleep(0.1)

        # Stop aggregation
        self.collector.stop_aggregation()

        # Should not raise exceptions
        assert True

    def test_thread_safety(self):
        """Test thread safety of metrics collection"""

        def record_events(collector, start_id, count):
            """Record events from a thread"""
            for i in range(count):
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY,
                    confidence=0.8,
                    metadata={"thread_id": start_id + i},
                )
                collector.record_event(event)

        # Create multiple threads
        threads = []
        events_per_thread = 10
        thread_count = 3

        for i in range(thread_count):
            thread = threading.Thread(target=record_events, args=(self.collector, i * events_per_thread, events_per_thread))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all events were recorded
        expected_total = thread_count * events_per_thread
        assert len(self.collector._events) == expected_total

    def test_metrics_calculation_performance(self, benchmark):
        """Test performance of metrics calculation"""
        # Add many events
        for i in range(100):
            event = MetricEvent(
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
                method=ProcessingMethod.PATTERN_PRIMARY,
                processing_time_ms=100 + i,
                success=i % 3 != 0,
            )
            self.collector.record_event(event)

        # Benchmark metrics calculation
        result = benchmark(self.collector.get_performance_metrics)

        assert isinstance(result, PerformanceMetrics)
        assert result.total_processed == 100


class TestPatternMetrics:
    """Test PatternMetrics class"""

    def test_pattern_metrics_creation(self):
        """Test creating PatternMetrics"""
        metrics = PatternMetrics(
            pattern_id="pattern_123", usage_count=50, success_rate=0.85, avg_confidence=0.78, avg_processing_time_ms=125.5
        )

        assert metrics.pattern_id == "pattern_123"
        assert metrics.usage_count == 50
        assert metrics.success_rate == 0.85
        assert metrics.avg_confidence == 0.78
        assert metrics.avg_processing_time_ms == 125.5

    def test_pattern_metrics_to_dict(self):
        """Test PatternMetrics to_dict method"""
        timestamp = datetime.now()
        metrics = PatternMetrics(pattern_id="pattern_123", usage_count=50, success_rate=0.85, last_used=timestamp)

        result = metrics.to_dict()

        expected_keys = [
            "pattern_id",
            "usage_count",
            "success_rate",
            "avg_confidence",
            "avg_processing_time_ms",
            "common_failures",
            "last_used",
        ]

        for key in expected_keys:
            assert key in result

        assert result["pattern_id"] == "pattern_123"
        assert result["last_used"] == timestamp.isoformat()


@pytest.mark.integration
class TestMetricsCollectorIntegration:
    """Integration tests for metrics collector"""

    def test_full_metrics_pipeline(self):
        """Test full metrics collection pipeline"""
        collector = MetricsCollector(buffer_size=50)

        # Start aggregation
        collector.start_aggregation()

        try:
            # Simulate real usage pattern
            methods = [ProcessingMethod.PATTERN_PRIMARY, ProcessingMethod.ML_PRIMARY, ProcessingMethod.HYBRID]

            # Record events over time
            for i in range(30):
                method = methods[i % len(methods)]
                success = i % 4 != 0  # 75% success rate

                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=method,
                    pattern_id=f"pattern_{i % 5}",
                    success=success,
                    confidence=0.7 + (0.2 * success),
                    processing_time_ms=80 + (i % 50),
                    address_length=20 + (i % 30),
                )

                collector.record_event(event)

                # Small delay to simulate real timing
                time.sleep(0.001)

            # Wait for aggregation
            time.sleep(0.1)

            # Get comprehensive metrics
            perf_metrics = collector.get_performance_metrics()
            system_metrics = collector.get_system_metrics()
            pattern_metrics = collector.get_pattern_metrics()

            # Verify results
            assert isinstance(perf_metrics, PerformanceMetrics)
            assert perf_metrics.total_processed == 30
            assert perf_metrics.avg_processing_time_ms > 0

            assert isinstance(system_metrics, SystemMetrics)
            assert len(system_metrics.method_distribution) == 3

            assert isinstance(pattern_metrics, dict)
            assert len(pattern_metrics) <= 5  # We used 5 different patterns

        finally:
            collector.stop_aggregation()

    def test_real_world_simulation(self, performance_addresses):
        """Test with real-world address simulation"""
        collector = MetricsCollector()

        # Simulate processing real addresses
        import random

        for i, sample in enumerate(performance_addresses[:20]):
            # Simulate different processing outcomes
            if sample.category == "edge_case":
                success = False
                confidence = random.uniform(0.1, 0.4)
                method = ProcessingMethod.FALLBACK
            elif sample.difficulty_level == "easy":
                success = True
                confidence = random.uniform(0.8, 0.95)
                method = ProcessingMethod.PATTERN_PRIMARY
            else:
                success = random.random() > 0.2
                confidence = random.uniform(0.5, 0.8)
                method = random.choice(
                    [ProcessingMethod.PATTERN_PRIMARY, ProcessingMethod.ML_PRIMARY, ProcessingMethod.HYBRID]
                )

            event = MetricEvent(
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
                method=method,
                success=success,
                confidence=confidence,
                processing_time_ms=random.uniform(50, 200),
                address_length=len(sample.input_address),
            )

            collector.record_event(event)

        # Analyze results
        metrics = collector.get_performance_metrics()

        assert metrics.total_processed == 20
        assert 0.0 <= metrics.error_rate <= 1.0
        assert metrics.avg_processing_time_ms > 0
