"""
Performance benchmark tests for address normalization system
"""

import pytest
import time
import sys
import statistics
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src to path
test_root = Path(__file__).parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.output.enhanced_formatter import EnhancedFormatter
from addrnorm.monitoring.metrics_collector import MetricsCollector, MetricEvent, MetricType, ProcessingMethod
from addrnorm.scoring.confidence import ConfidenceCalculator
from addrnorm.scoring.quality import QualityAssessment


class BenchmarkMetrics:
    """Container for benchmark metrics"""

    def __init__(self):
        self.processing_times = []
        self.memory_usage = []
        self.success_rate = 0.0
        self.throughput = 0.0
        self.error_rate = 0.0
        self.confidence_scores = []
        self.quality_scores = []

    def add_measurement(self, processing_time, success, confidence=None, quality=None, memory_mb=None):
        """Add a measurement to the benchmark"""
        self.processing_times.append(processing_time)
        if confidence is not None:
            self.confidence_scores.append(confidence)
        if quality is not None:
            self.quality_scores.append(quality)
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)

    def calculate_stats(self):
        """Calculate benchmark statistics"""
        if not self.processing_times:
            return {}

        return {
            "processing_time": {
                "mean": statistics.mean(self.processing_times),
                "median": statistics.median(self.processing_times),
                "std": statistics.stdev(self.processing_times) if len(self.processing_times) > 1 else 0,
                "min": min(self.processing_times),
                "max": max(self.processing_times),
                "p95": sorted(self.processing_times)[int(len(self.processing_times) * 0.95)],
                "p99": sorted(self.processing_times)[int(len(self.processing_times) * 0.99)],
            },
            "confidence": (
                {
                    "mean": statistics.mean(self.confidence_scores) if self.confidence_scores else 0,
                    "std": statistics.stdev(self.confidence_scores) if len(self.confidence_scores) > 1 else 0,
                }
                if self.confidence_scores
                else {}
            ),
            "quality": (
                {
                    "mean": statistics.mean(self.quality_scores) if self.quality_scores else 0,
                    "std": statistics.stdev(self.quality_scores) if len(self.quality_scores) > 1 else 0,
                }
                if self.quality_scores
                else {}
            ),
            "throughput": self.throughput,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
        }


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def setup_method(self):
        """Setup for each test method"""
        self.formatter = EnhancedFormatter(
            include_confidence=True, include_quality=True, include_explanations=False  # Disable for performance
        )
        self.collector = MetricsCollector()
        self.benchmark_metrics = BenchmarkMetrics()

    def mock_normalize_realistic(self, address):
        """Realistic mock normalization with variable performance"""
        import random

        # Simulate realistic processing time based on address complexity
        base_time = 0.001  # 1ms base
        complexity_factor = len(address) / 100.0
        processing_time = base_time + (complexity_factor * 0.002)
        time.sleep(processing_time)

        # Simulate success rate based on address quality
        success_probability = 0.95 if len(address) > 20 else 0.85
        success = random.random() < success_probability

        if success:
            components = {
                "il": random.choice(["İstanbul", "Ankara", "İzmir"]),
                "ilce": random.choice(["Kadıköy", "Beşiktaş", "Şişli"]),
            }

            if len(address) > 40:
                components["mahalle"] = "Test Mahalle"
                components["yol"] = "Test Caddesi"

            return {
                "success": True,
                "components": components,
                "normalized": address.title(),
                "method": "pattern_primary" if len(address) > 30 else "ml_primary",
            }
        else:
            return {"success": False, "components": {}, "error": "Normalization failed"}

    @pytest.mark.timeout(300)  # 5 minute timeout
    def test_single_address_performance(self, performance_addresses):
        """Benchmark single address processing performance"""
        test_addresses = performance_addresses[:100]
        metrics = BenchmarkMetrics()

        # Warm up
        for i in range(5):
            with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                self.formatter.format_single(test_addresses[0].input_address)

        # Actual benchmark
        start_time = time.time()
        successful_count = 0

        for sample in test_addresses:
            addr_start = time.time()

            with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                result = self.formatter.format_single(sample.input_address)

            processing_time = (time.time() - addr_start) * 1000  # ms

            if result.success:
                successful_count += 1

            confidence = result.confidence.overall if result.confidence else 0
            quality = result.quality.overall_score if result.quality else 0

            metrics.add_measurement(
                processing_time=processing_time, success=result.success, confidence=confidence, quality=quality
            )

        total_time = time.time() - start_time
        metrics.throughput = len(test_addresses) / total_time
        metrics.success_rate = successful_count / len(test_addresses)

        stats = metrics.calculate_stats()

        # Performance assertions
        assert stats["processing_time"]["mean"] < 10.0  # Average under 10ms
        assert stats["processing_time"]["p95"] < 20.0  # 95th percentile under 20ms
        assert stats["processing_time"]["p99"] < 50.0  # 99th percentile under 50ms
        assert metrics.throughput > 50  # At least 50 addresses per second
        assert metrics.success_rate > 0.8  # At least 80% success rate

        print(f"\nSingle Address Performance Benchmark:")
        print(f"Average processing time: {stats['processing_time']['mean']:.2f}ms")
        print(f"P95 processing time: {stats['processing_time']['p95']:.2f}ms")
        print(f"Throughput: {metrics.throughput:.1f} addresses/second")
        print(f"Success rate: {metrics.success_rate:.1%}")

    @pytest.mark.timeout(300)
    def test_batch_processing_performance(self, performance_addresses):
        """Benchmark batch processing performance"""
        batch_sizes = [10, 50, 100, 500]
        results = {}

        for batch_size in batch_sizes:
            batch_addresses = [s.input_address for s in performance_addresses[:batch_size]]

            # Mock batch normalization
            def mock_batch_normalize(addresses):
                results = []
                for addr in addresses:
                    mock_result = self.mock_normalize_realistic(addr)
                    results.append(mock_result)
                return results

            # Warm up
            with patch.object(self.formatter, "_normalize_address_batch", side_effect=mock_batch_normalize):
                self.formatter.format_batch(batch_addresses[:5])

            # Benchmark
            start_time = time.time()

            with patch.object(self.formatter, "_normalize_address_batch", side_effect=mock_batch_normalize):
                batch_results = self.formatter.format_batch(batch_addresses)

            total_time = time.time() - start_time
            throughput = len(batch_addresses) / total_time
            avg_time_per_item = (total_time / len(batch_addresses)) * 1000  # ms

            successful = sum(1 for r in batch_results if r.success)
            success_rate = successful / len(batch_results)

            results[batch_size] = {
                "throughput": throughput,
                "avg_time_per_item": avg_time_per_item,
                "success_rate": success_rate,
                "total_time": total_time,
            }

            # Performance assertions for batch processing
            assert throughput > 100  # Should be faster than single processing
            assert avg_time_per_item < 5.0  # Should be under 5ms per item in batch
            assert success_rate > 0.8

        print(f"\nBatch Processing Performance Benchmark:")
        for batch_size, metrics in results.items():
            print(
                f"Batch size {batch_size}: {metrics['throughput']:.1f} addr/sec, " f"{metrics['avg_time_per_item']:.2f}ms/item"
            )

    @pytest.mark.timeout(300)
    def test_concurrent_processing_performance(self, performance_addresses):
        """Benchmark concurrent processing performance"""
        test_addresses = [s.input_address for s in performance_addresses[:200]]
        thread_counts = [1, 2, 4, 8]
        results = {}

        for thread_count in thread_counts:
            metrics = BenchmarkMetrics()

            def process_chunk(chunk):
                chunk_results = []
                chunk_times = []

                for addr in chunk:
                    start_time = time.time()

                    with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                        result = self.formatter.format_single(addr)

                    processing_time = (time.time() - start_time) * 1000
                    chunk_results.append(result)
                    chunk_times.append(processing_time)

                return chunk_results, chunk_times

            # Split addresses into chunks
            chunk_size = len(test_addresses) // thread_count
            chunks = [test_addresses[i : i + chunk_size] for i in range(0, len(test_addresses), chunk_size)]

            # Benchmark concurrent processing
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

                all_results = []
                all_times = []

                for future in futures:
                    chunk_results, chunk_times = future.result()
                    all_results.extend(chunk_results)
                    all_times.extend(chunk_times)

            total_time = time.time() - start_time
            throughput = len(all_results) / total_time
            successful = sum(1 for r in all_results if r.success)
            success_rate = successful / len(all_results)

            results[thread_count] = {
                "throughput": throughput,
                "total_time": total_time,
                "success_rate": success_rate,
                "avg_processing_time": statistics.mean(all_times),
            }

            # Performance should scale with thread count (up to a point)
            if thread_count == 1:
                baseline_throughput = throughput
            else:
                # Should see some improvement with more threads
                improvement_factor = throughput / baseline_throughput
                assert improvement_factor > 1.2  # At least 20% improvement

        print(f"\nConcurrent Processing Performance Benchmark:")
        for threads, metrics in results.items():
            print(f"{threads} threads: {metrics['throughput']:.1f} addr/sec, " f"{metrics['avg_processing_time']:.2f}ms avg")

    @pytest.mark.timeout(300)
    def test_monitoring_overhead_benchmark(self, performance_addresses):
        """Benchmark monitoring system overhead"""
        test_addresses = performance_addresses[:100]

        # Benchmark without monitoring
        start_time = time.time()
        no_monitoring_results = []

        for sample in test_addresses:
            with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                result = self.formatter.format_single(sample.input_address)
            no_monitoring_results.append(result)

        no_monitoring_time = time.time() - start_time

        # Benchmark with monitoring
        self.collector.start_aggregation()

        try:
            start_time = time.time()
            monitoring_results = []

            for sample in test_addresses:
                with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                    result = self.formatter.format_single(sample.input_address)

                monitoring_results.append(result)

                # Record monitoring metrics
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY,
                    success=result.success,
                    confidence=result.confidence.overall if result.confidence else 0,
                )
                self.collector.record_event(event)

            monitoring_time = time.time() - start_time

        finally:
            self.collector.stop_aggregation()

        # Calculate overhead
        overhead_percentage = ((monitoring_time - no_monitoring_time) / no_monitoring_time) * 100

        # Monitoring overhead should be minimal
        assert overhead_percentage < 20.0  # Less than 20% overhead

        print(f"\nMonitoring Overhead Benchmark:")
        print(f"Without monitoring: {no_monitoring_time:.3f}s")
        print(f"With monitoring: {monitoring_time:.3f}s")
        print(f"Overhead: {overhead_percentage:.1f}%")

    @pytest.mark.timeout(600)  # 10 minute timeout for stress test
    def test_stress_test_performance(self, performance_addresses):
        """Stress test with large volume of addresses"""
        # Use a large number of addresses for stress testing
        stress_addresses = performance_addresses[:1000]

        metrics = BenchmarkMetrics()
        self.collector.start_aggregation()

        try:
            start_time = time.time()
            successful_count = 0
            error_count = 0

            for i, sample in enumerate(stress_addresses):
                addr_start = time.time()

                try:
                    with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                        result = self.formatter.format_single(sample.input_address)

                    processing_time = (time.time() - addr_start) * 1000

                    if result.success:
                        successful_count += 1

                    confidence = result.confidence.overall if result.confidence else 0
                    metrics.add_measurement(processing_time=processing_time, success=result.success, confidence=confidence)

                    # Record monitoring
                    event = MetricEvent(
                        timestamp=datetime.now(),
                        metric_type=MetricType.PERFORMANCE,
                        method=ProcessingMethod.PATTERN_PRIMARY,
                        success=result.success,
                        confidence=confidence,
                        processing_time_ms=processing_time,
                    )
                    self.collector.record_event(event)

                except Exception as e:
                    error_count += 1
                    event = MetricEvent(
                        timestamp=datetime.now(),
                        metric_type=MetricType.ERROR,
                        method=ProcessingMethod.FALLBACK,
                        success=False,
                        error_type=type(e).__name__,
                    )
                    self.collector.record_event(event)

                # Progress reporting
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    current_throughput = (i + 1) / elapsed
                    print(f"Processed {i + 1}/1000 addresses, " f"throughput: {current_throughput:.1f} addr/sec")

            total_time = time.time() - start_time
            metrics.throughput = len(stress_addresses) / total_time
            metrics.success_rate = successful_count / len(stress_addresses)
            metrics.error_rate = error_count / len(stress_addresses)

            stats = metrics.calculate_stats()

            # Stress test assertions
            assert metrics.throughput > 30  # Should maintain at least 30 addr/sec under stress
            assert metrics.success_rate > 0.75  # Should maintain 75% success rate
            assert metrics.error_rate < 0.05  # Less than 5% errors
            assert stats["processing_time"]["p99"] < 100.0  # 99th percentile under 100ms

            print(f"\nStress Test Performance Results:")
            print(f"Total addresses processed: {len(stress_addresses)}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average throughput: {metrics.throughput:.1f} addr/sec")
            print(f"Success rate: {metrics.success_rate:.1%}")
            print(f"Error rate: {metrics.error_rate:.1%}")
            print(f"P99 processing time: {stats['processing_time']['p99']:.2f}ms")

        finally:
            self.collector.stop_aggregation()

    def test_memory_efficiency_benchmark(self, performance_addresses):
        """Benchmark memory efficiency during processing"""
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory benchmarking")

        # Baseline memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process addresses in batches to test memory efficiency
        batch_size = 100
        test_addresses = performance_addresses[:500]
        memory_readings = [initial_memory]

        for i in range(0, len(test_addresses), batch_size):
            batch = test_addresses[i : i + batch_size]

            # Process batch
            for sample in batch:
                with patch.object(self.formatter, "_normalize_address", side_effect=self.mock_normalize_realistic):
                    result = self.formatter.format_single(sample.input_address)

            # Record memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)

            # Force garbage collection
            import gc

            gc.collect()

        # Calculate memory statistics
        max_memory = max(memory_readings)
        avg_memory = statistics.mean(memory_readings)
        memory_growth = max_memory - initial_memory

        # Memory efficiency assertions
        assert memory_growth < 50  # Should not grow more than 50MB
        assert max_memory < initial_memory + 100  # Should not exceed 100MB increase

        print(f"\nMemory Efficiency Benchmark:")
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Maximum memory: {max_memory:.1f}MB")
        print(f"Average memory: {avg_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")


@pytest.mark.benchmark
class TestAccuracyBenchmarks:
    """Accuracy and quality benchmark tests"""

    def setup_method(self):
        """Setup for each test method"""
        self.confidence_calc = ConfidenceCalculator()
        self.quality_assessment = QualityAssessment()

    def test_confidence_scoring_accuracy(self, sample_addresses):
        """Benchmark confidence scoring accuracy"""
        # Test confidence scoring with known good/bad examples
        high_quality_addresses = [
            "Moda Mahallesi, Bahariye Caddesi No:15/3, Kadıköy, İstanbul",
            "Alsancak Mahallesi, Kıbrıs Şehitleri Caddesi No:140, Konak, İzmir",
            "Çankaya Mahallesi, Atatürk Bulvarı No:23, Çankaya, Ankara",
        ]

        low_quality_addresses = ["moda bhr cd 15", "xyz123 invalid", "test address"]

        # Test high quality addresses
        high_quality_scores = []
        for addr in high_quality_addresses:
            # Mock high-quality normalization result
            mock_result = {
                "components": {
                    "mahalle": "Test Mahalle",
                    "yol": "Test Caddesi",
                    "no": "15",
                    "ilce": "Kadıköy",
                    "il": "İstanbul",
                },
                "method": "pattern_primary",
                "pattern_matches": [0.95, 0.90, 0.88],
            }

            confidence = self.confidence_calc.calculate_overall_confidence(
                normalization_result=mock_result, input_address=addr
            )
            high_quality_scores.append(confidence.overall)

        # Test low quality addresses
        low_quality_scores = []
        for addr in low_quality_addresses:
            mock_result = {"components": {"il": "İstanbul"}, "method": "fallback", "pattern_matches": [0.3]}

            confidence = self.confidence_calc.calculate_overall_confidence(
                normalization_result=mock_result, input_address=addr
            )
            low_quality_scores.append(confidence.overall)

        # Accuracy assertions
        avg_high_quality = statistics.mean(high_quality_scores)
        avg_low_quality = statistics.mean(low_quality_scores)

        assert avg_high_quality > 0.8  # High quality should score above 0.8
        assert avg_low_quality < 0.5  # Low quality should score below 0.5
        assert avg_high_quality > avg_low_quality + 0.3  # Clear separation

        print(f"\nConfidence Scoring Accuracy:")
        print(f"High quality average: {avg_high_quality:.3f}")
        print(f"Low quality average: {avg_low_quality:.3f}")
        print(f"Separation: {avg_high_quality - avg_low_quality:.3f}")

    def test_quality_assessment_accuracy(self, sample_addresses):
        """Benchmark quality assessment accuracy"""
        # Test with addresses of known quality levels
        complete_addresses = sample_addresses[:10]  # Assume these are complete
        incomplete_addresses = [
            type("Sample", (), {"input_address": "Istanbul"})(),
            type("Sample", (), {"input_address": "Kadikoy"})(),
            type("Sample", (), {"input_address": "Test"})(),
        ]

        # Test complete addresses
        complete_scores = []
        for sample in complete_addresses:
            mock_result = {
                "components": {"il": "İstanbul", "ilce": "Kadıköy", "mahalle": "Moda", "yol": "Bahariye Caddesi", "no": "15"}
            }

            quality = self.quality_assessment.assess_quality(
                normalization_result=mock_result, input_address=sample.input_address
            )
            complete_scores.append(quality.overall_score)

        # Test incomplete addresses
        incomplete_scores = []
        for sample in incomplete_addresses:
            mock_result = {"components": {"il": "İstanbul"}}

            quality = self.quality_assessment.assess_quality(
                normalization_result=mock_result, input_address=sample.input_address
            )
            incomplete_scores.append(quality.overall_score)

        # Quality assessment accuracy
        avg_complete = statistics.mean(complete_scores)
        avg_incomplete = statistics.mean(incomplete_scores)

        assert avg_complete > 0.7  # Complete addresses should score high
        assert avg_incomplete < 0.4  # Incomplete addresses should score low
        assert avg_complete > avg_incomplete + 0.3  # Clear quality distinction

        print(f"\nQuality Assessment Accuracy:")
        print(f"Complete addresses average: {avg_complete:.3f}")
        print(f"Incomplete addresses average: {avg_incomplete:.3f}")
        print(f"Quality separation: {avg_complete - avg_incomplete:.3f}")


def save_benchmark_results(results, output_file="benchmark_results.json"):
    """Save benchmark results to file"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Benchmark results saved to {output_file}")


if __name__ == "__main__":
    # Run benchmarks independently
    pytest.main([__file__, "-v", "-m", "benchmark"])
