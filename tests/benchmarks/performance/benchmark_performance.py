"""
Performance Benchmark Suite for Address Normalization

Comprehensive performance testing including:
- Processing speed benchmarks
- Memory usage profiling
- Throughput analysis
- Scalability testing
"""

import time
import psutil
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import tracemalloc
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from addrnorm.preprocess.api import AddressNormalizer
from addrnorm.integration.hybrid import HybridProcessor
from addrnorm.monitoring.metrics_collector import MetricsCollector
from tests.fixtures.address_samples import get_performance_test_samples, get_all_samples


@dataclass
class BenchmarkResult:
    """Single benchmark test result"""

    test_name: str
    processing_time: float
    memory_usage_mb: float
    throughput_per_sec: float
    success_rate: float
    error_count: int
    sample_size: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""

    timestamp: str
    system_info: Dict[str, Any]
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark:
    """Performance benchmark runner"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.normalizer = None
        self.collector = None

    def setup(self):
        """Setup benchmark environment"""
        # Initialize normalizer
        self.normalizer = AddressNormalizer()

        # Initialize metrics collector
        self.collector = MetricsCollector()
        self.collector.start_aggregation()

        # Warm up JIT/caches
        self._warmup()

    def _warmup(self):
        """Warm up the system with a few test runs"""
        test_addresses = [
            "İstanbul Kadıköy Test Mahallesi Test Sokak No:1",
            "Ankara Çankaya Test Bulvarı No:123",
            "İzmir Karşıyaka Test Caddesi No:456",
        ]

        for addr in test_addresses:
            try:
                self.normalizer.process(addr)
            except:
                pass  # Ignore warmup errors

    def teardown(self):
        """Cleanup benchmark environment"""
        if self.collector:
            self.collector.stop_aggregation()

    def run_single_thread_benchmark(self, sample_size: int = 1000) -> BenchmarkResult:
        """Run single-threaded processing benchmark"""
        test_data = get_performance_test_samples(sample_size)

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run benchmark
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0

        for sample in test_data:
            try:
                result = self.normalizer.process(sample.address)
                if result.success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1

        end_time = time.perf_counter()

        # Memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = max(end_memory - start_memory, peak / 1024 / 1024)

        # Calculate metrics
        processing_time = end_time - start_time
        throughput = sample_size / processing_time if processing_time > 0 else 0
        success_rate = success_count / sample_size if sample_size > 0 else 0

        return BenchmarkResult(
            test_name="single_thread_processing",
            processing_time=processing_time,
            memory_usage_mb=memory_used,
            throughput_per_sec=throughput,
            success_rate=success_rate,
            error_count=error_count,
            sample_size=sample_size,
            details={
                "avg_time_per_address": processing_time / sample_size if sample_size > 0 else 0,
                "peak_memory_mb": peak / 1024 / 1024,
                "memory_efficiency": sample_size / memory_used if memory_used > 0 else 0,
            },
        )

    def run_concurrent_benchmark(self, sample_size: int = 1000, max_workers: int = 4) -> BenchmarkResult:
        """Run concurrent processing benchmark"""
        test_data = get_performance_test_samples(sample_size)

        # Split data into chunks
        chunk_size = sample_size // max_workers
        chunks = [test_data[i : i + chunk_size] for i in range(0, len(test_data), chunk_size)]

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024

        # Run concurrent benchmark
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0

        def process_chunk(chunk):
            """Process a chunk of addresses"""
            local_normalizer = AddressNormalizer()
            chunk_success = 0
            chunk_errors = 0

            for sample in chunk:
                try:
                    result = local_normalizer.process(sample.address)
                    if result.success:
                        chunk_success += 1
                    else:
                        chunk_errors += 1
                except Exception:
                    chunk_errors += 1

            return chunk_success, chunk_errors

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

            for future in futures:
                chunk_success, chunk_errors = future.result()
                success_count += chunk_success
                error_count += chunk_errors

        end_time = time.perf_counter()

        # Memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = max(end_memory - start_memory, peak / 1024 / 1024)

        # Calculate metrics
        processing_time = end_time - start_time
        throughput = sample_size / processing_time if processing_time > 0 else 0
        success_rate = success_count / sample_size if sample_size > 0 else 0

        return BenchmarkResult(
            test_name=f"concurrent_processing_{max_workers}_workers",
            processing_time=processing_time,
            memory_usage_mb=memory_used,
            throughput_per_sec=throughput,
            success_rate=success_rate,
            error_count=error_count,
            sample_size=sample_size,
            details={
                "workers": max_workers,
                "chunks": len(chunks),
                "avg_chunk_size": chunk_size,
                "speedup_factor": (
                    (self.last_single_thread_time / processing_time) if hasattr(self, "last_single_thread_time") else 1.0
                ),
                "parallel_efficiency": (
                    (self.last_single_thread_time / (processing_time * max_workers))
                    if hasattr(self, "last_single_thread_time")
                    else 1.0
                ),
            },
        )

    def run_memory_stress_test(self, max_sample_size: int = 10000) -> BenchmarkResult:
        """Run memory stress test with increasing data sizes"""
        sizes = [100, 500, 1000, 2500, 5000, max_sample_size]
        memory_usage = []
        processing_times = []

        for size in sizes:
            # Force garbage collection
            gc.collect()

            # Get test data
            test_data = get_performance_test_samples(size)

            # Memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024

            # Process addresses
            start_time = time.perf_counter()
            for sample in test_data[:size]:  # Limit to current size
                try:
                    self.normalizer.process(sample.address)
                except Exception:
                    pass

            end_time = time.perf_counter()

            # Memory after
            memory_after = process.memory_info().rss / 1024 / 1024

            memory_usage.append(memory_after - memory_before)
            processing_times.append(end_time - start_time)

        # Calculate memory growth rate
        memory_growth_rate = 0
        if len(memory_usage) > 1:
            # Linear regression for memory growth
            x = sizes
            y = memory_usage
            n = len(x)
            x_mean = sum(x) / n
            y_mean = sum(y) / n
            memory_growth_rate = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / sum(
                (x[i] - x_mean) ** 2 for i in range(n)
            )

        return BenchmarkResult(
            test_name="memory_stress_test",
            processing_time=sum(processing_times),
            memory_usage_mb=max(memory_usage),
            throughput_per_sec=max_sample_size / sum(processing_times) if sum(processing_times) > 0 else 0,
            success_rate=1.0,  # We don't track individual successes here
            error_count=0,
            sample_size=max_sample_size,
            details={
                "test_sizes": sizes,
                "memory_usage_per_size": memory_usage,
                "processing_times_per_size": processing_times,
                "memory_growth_rate_mb_per_1k": memory_growth_rate * 1000,
                "memory_efficiency": max_sample_size / max(memory_usage) if max(memory_usage) > 0 else 0,
            },
        )

    def run_scalability_test(self) -> BenchmarkResult:
        """Test scalability across different thread counts"""
        sample_size = 2000
        thread_counts = [1, 2, 4, 8, min(16, multiprocessing.cpu_count())]
        results = {}

        # Single thread baseline
        baseline_result = self.run_single_thread_benchmark(sample_size)
        self.last_single_thread_time = baseline_result.processing_time
        baseline_time = baseline_result.processing_time

        for thread_count in thread_counts[1:]:  # Skip 1 since we have baseline
            result = self.run_concurrent_benchmark(sample_size, thread_count)
            results[thread_count] = {
                "time": result.processing_time,
                "throughput": result.throughput_per_sec,
                "speedup": baseline_time / result.processing_time if result.processing_time > 0 else 1.0,
                "efficiency": (baseline_time / result.processing_time) / thread_count if result.processing_time > 0 else 1.0,
            }

        # Calculate scalability metrics
        max_speedup = max((res["speedup"] for res in results.values()), default=1.0)
        optimal_threads = max((k for k, v in results.items() if v["speedup"] == max_speedup), default=1)

        return BenchmarkResult(
            test_name="scalability_test",
            processing_time=baseline_time,
            memory_usage_mb=baseline_result.memory_usage_mb,
            throughput_per_sec=baseline_result.throughput_per_sec,
            success_rate=baseline_result.success_rate,
            error_count=baseline_result.error_count,
            sample_size=sample_size,
            details={
                "thread_results": results,
                "max_speedup": max_speedup,
                "optimal_thread_count": optimal_threads,
                "scalability_efficiency": max_speedup / optimal_threads if optimal_threads > 0 else 1.0,
                "baseline_throughput": baseline_result.throughput_per_sec,
            },
        )

    def run_all_benchmarks(self, sample_sizes: List[int] = None) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        if sample_sizes is None:
            sample_sizes = [1000, 5000, 10000]

        self.setup()

        suite = BenchmarkSuite(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), system_info=self._get_system_info())

        try:
            # Run benchmarks for different sample sizes
            for size in sample_sizes:
                print(f"Running benchmarks for sample size: {size}")

                # Single thread
                result = self.run_single_thread_benchmark(size)
                suite.results.append(result)

                # Concurrent
                result = self.run_concurrent_benchmark(size, max_workers=4)
                suite.results.append(result)

            # Memory stress test
            print("Running memory stress test...")
            result = self.run_memory_stress_test(max(sample_sizes))
            suite.results.append(result)

            # Scalability test
            print("Running scalability test...")
            result = self.run_scalability_test()
            suite.results.append(result)

            # Generate summary
            suite.summary = self._generate_summary(suite.results)

        finally:
            self.teardown()

        return suite

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": multiprocessing.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        }

    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results"""
        if not results:
            return {}

        # Aggregate metrics
        total_addresses_processed = sum(r.sample_size for r in results)
        avg_throughput = statistics.mean(r.throughput_per_sec for r in results)
        max_throughput = max(r.throughput_per_sec for r in results)
        avg_memory_usage = statistics.mean(r.memory_usage_mb for r in results)
        avg_success_rate = statistics.mean(r.success_rate for r in results)

        # Find best performers
        best_throughput_test = max(results, key=lambda r: r.throughput_per_sec)
        most_memory_efficient = min(
            results, key=lambda r: r.memory_usage_mb / r.sample_size if r.sample_size > 0 else float("inf")
        )

        return {
            "total_addresses_processed": total_addresses_processed,
            "avg_throughput_per_sec": avg_throughput,
            "max_throughput_per_sec": max_throughput,
            "avg_memory_usage_mb": avg_memory_usage,
            "avg_success_rate": avg_success_rate,
            "best_throughput_test": best_throughput_test.test_name,
            "most_memory_efficient_test": most_memory_efficient.test_name,
            "recommendations": self._generate_recommendations(results),
        }

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []

        # Analyze results for recommendations
        concurrent_results = [r for r in results if "concurrent" in r.test_name]
        single_results = [r for r in results if "single_thread" in r.test_name]

        if concurrent_results and single_results:
            best_concurrent = max(concurrent_results, key=lambda r: r.throughput_per_sec)
            best_single = max(single_results, key=lambda r: r.throughput_per_sec)

            if best_concurrent.throughput_per_sec > best_single.throughput_per_sec * 1.5:
                recommendations.append("Use concurrent processing for better performance")
            else:
                recommendations.append("Single-threaded processing may be sufficient for your workload")

        # Memory recommendations
        memory_results = [r for r in results if r.memory_usage_mb > 0]
        if memory_results:
            avg_memory_per_address = statistics.mean(
                r.memory_usage_mb / r.sample_size for r in memory_results if r.sample_size > 0
            )
            if avg_memory_per_address > 0.1:  # More than 0.1MB per address
                recommendations.append("Consider batch processing to optimize memory usage")

        # Success rate recommendations
        success_rates = [r.success_rate for r in results]
        if success_rates and statistics.mean(success_rates) < 0.95:
            recommendations.append("Review error handling and validation logic")

        return recommendations

    def save_results(self, suite: BenchmarkSuite, filename: str = None) -> Path:
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert to dict for JSON serialization
        suite_dict = {
            "timestamp": suite.timestamp,
            "system_info": suite.system_info,
            "results": [
                {
                    "test_name": r.test_name,
                    "processing_time": r.processing_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "throughput_per_sec": r.throughput_per_sec,
                    "success_rate": r.success_rate,
                    "error_count": r.error_count,
                    "sample_size": r.sample_size,
                    "details": r.details,
                }
                for r in suite.results
            ],
            "summary": suite.summary,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(suite_dict, f, indent=2, ensure_ascii=False)

        return output_path


def run_performance_benchmark(sample_sizes: List[int] = None, output_dir: str = "benchmark_results") -> Path:
    """Convenience function to run performance benchmark"""
    benchmark = PerformanceBenchmark(output_dir)
    suite = benchmark.run_all_benchmarks(sample_sizes)
    return benchmark.save_results(suite)


if __name__ == "__main__":
    # Example usage
    print("Running performance benchmark suite...")
    result_path = run_performance_benchmark([500, 1000, 2000])
    print(f"Results saved to: {result_path}")
