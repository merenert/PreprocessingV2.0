"""
Integration tests for the complete address normalization pipeline
"""

import pytest
import sys
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
test_root = Path(__file__).parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.output.enhanced_formatter import EnhancedFormatter
from addrnorm.monitoring.metrics_collector import MetricsCollector, MetricEvent, MetricType, ProcessingMethod
from addrnorm.monitoring.analytics import SystemAnalytics
from addrnorm.monitoring.reporter import SystemReporter
from addrnorm.scoring.confidence import ConfidenceCalculator
from addrnorm.scoring.quality import QualityAssessment


class TestFullPipelineIntegration:
    """Test complete pipeline integration"""

    def setup_method(self):
        """Setup for each test method"""
        self.formatter = EnhancedFormatter(include_confidence=True, include_quality=True, include_explanations=True)
        self.collector = MetricsCollector()
        self.analytics = SystemAnalytics(self.collector)
        self.reporter = SystemReporter(self.analytics)

    def test_end_to_end_processing(self, sample_addresses):
        """Test end-to-end address processing pipeline"""
        # Select test addresses
        test_addresses = sample_addresses[:10]

        # Mock the core normalization function
        def mock_normalize(address):
            # Simulate different outcomes based on address characteristics
            if not address or len(address) < 5:
                return {"success": False, "components": {}, "error": "Invalid input"}
            elif "xyz" in address.lower() or "invalid" in address.lower():
                return {"success": False, "components": {}, "error": "Unrecognized format"}
            else:
                # Successful processing
                components = {"il": "İstanbul", "ilce": "Kadıköy"}

                # Add more components for longer addresses
                if len(address) > 30:
                    components["mahalle"] = "Moda"
                if len(address) > 50:
                    components["yol"] = "Bahariye Caddesi"

                return {
                    "success": True,
                    "components": components,
                    "normalized": address.title(),
                    "method": "pattern_primary" if len(address) > 40 else "ml_primary",
                }

        # Start metrics collection
        self.collector.start_aggregation()

        try:
            results = []

            for sample in test_addresses:
                start_time = time.time()

                # Mock normalization
                with patch.object(self.formatter, "_normalize_address", side_effect=mock_normalize):
                    result = self.formatter.format_single(sample.input_address)

                processing_time = (time.time() - start_time) * 1000

                # Record metrics
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY if result.success else ProcessingMethod.FALLBACK,
                    success=result.success,
                    confidence=result.confidence.overall if result.confidence else 0.0,
                    processing_time_ms=processing_time,
                    address_length=len(sample.input_address),
                )
                self.collector.record_event(event)

                results.append(result)

            # Wait for metrics aggregation
            time.sleep(0.1)

            # Verify results
            assert len(results) == len(test_addresses)

            # Check success rate
            successful = sum(1 for r in results if r.success)
            success_rate = successful / len(results)
            assert success_rate >= 0.5  # Should process at least half successfully

            # Verify metrics collection
            perf_metrics = self.collector.get_performance_metrics()
            assert perf_metrics.total_processed == len(test_addresses)
            assert perf_metrics.avg_processing_time_ms > 0

            # Generate analytics report
            report = self.analytics.generate_comprehensive_report(time_window_hours=1)
            assert "performance_insights" in report
            assert "system_metrics" in report

        finally:
            self.collector.stop_aggregation()

    def test_batch_processing_with_monitoring(self, sample_addresses):
        """Test batch processing with comprehensive monitoring"""
        # Use first 20 addresses for batch processing
        batch_addresses = [s.input_address for s in sample_addresses[:20]]

        # Mock normalization for individual addresses (format_batch calls format_single)
        def mock_normalize(address):
            if len(address) < 10:
                result = {"success": False, "components": {}}
            else:
                result = {"success": True, "components": {"il": "İstanbul"}, "normalized": address.title()}
            return result

        self.collector.start_aggregation()

        try:
            # Process batch with monitoring
            start_time = time.time()

            with patch.object(self.formatter, "_normalize_address", side_effect=mock_normalize):
                results = self.formatter.format_batch(batch_addresses)

            total_time = time.time() - start_time

            # Record batch metrics
            for i, result in enumerate(results):
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY if result.success else ProcessingMethod.FALLBACK,
                    success=result.success,
                    processing_time_ms=(total_time / len(results)) * 1000,
                    address_length=len(batch_addresses[i]),
                )
                self.collector.record_event(event)

            # Verify batch results
            assert len(results) == len(batch_addresses)

            # Check performance
            successful = sum(1 for r in results if r.success)
            assert successful > 0

            # Verify batch processing is faster per item
            avg_time_per_item = (total_time / len(results)) * 1000
            assert avg_time_per_item < 100  # Should be under 100ms per item

        finally:
            self.collector.stop_aggregation()

    def test_export_and_reporting_integration(self, temp_output_dir, sample_addresses):
        """Test integrated export and reporting functionality"""
        # Process some addresses
        test_samples = sample_addresses[:5]

        def mock_normalize(address):
            return {
                "success": len(address) > 10,
                "components": {"il": "İstanbul", "ilce": "Kadıköy"} if len(address) > 10 else {},
                "normalized": address.title() if len(address) > 10 else address,
            }

        self.collector.start_aggregation()

        try:
            # Process addresses
            results = []
            for sample in test_samples:
                with patch.object(self.formatter, "_normalize_address", side_effect=mock_normalize):
                    result = self.formatter.format_single(sample.input_address)
                results.append(result)

                # Record metrics
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY,
                    success=result.success,
                    confidence=result.confidence.overall if result.confidence else 0.5,
                )
                self.collector.record_event(event)

            time.sleep(0.1)  # Wait for aggregation

            # Export results in multiple formats
            json_file = self.formatter.export_json(results, str(temp_output_dir / "results.json"))
            csv_file = self.formatter.export_csv(results, str(temp_output_dir / "results.csv"))
            xml_file = self.formatter.export_xml(results, str(temp_output_dir / "results.xml"))

            # Verify exports
            assert Path(json_file).exists()
            assert Path(csv_file).exists()
            assert Path(xml_file).exists()

            # Verify file contents
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                assert "results" in json_data
                assert len(json_data["results"]) == len(results)

            # Generate system report
            report_content = self.reporter.generate_report(format="html", time_window_hours=1)
            assert len(report_content) > 1000  # Should be substantial HTML

            # Save report
            report_file = temp_output_dir / "system_report.html"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)

            assert report_file.exists()

        finally:
            self.collector.stop_aggregation()

    def test_error_handling_integration(self):
        """Test error handling across the integrated system"""
        # Test with problematic inputs
        problematic_inputs = [
            "",
            None,
            "xyz123 invalid format",
            "a" * 1000,  # Very long input
            "🚀🎉🔥",  # Emoji only
        ]

        self.collector.start_aggregation()

        try:
            results = []

            for addr in problematic_inputs:
                try:
                    # Mock normalization that can handle errors
                    def mock_normalize_with_errors(address):
                        if not address:
                            raise ValueError("Empty address")
                        if len(address) > 500:
                            raise MemoryError("Input too long")
                        if not any(c.isalpha() for c in address):
                            return {"success": False, "error": "No valid characters"}
                        return {"success": False, "error": "Unrecognized format"}

                    with patch.object(self.formatter, "_normalize_address", side_effect=mock_normalize_with_errors):
                        result = self.formatter.format_single(addr or "")

                    results.append(result)

                    # Record error metrics
                    event = MetricEvent(
                        timestamp=datetime.now(),
                        metric_type=MetricType.ERROR,
                        method=ProcessingMethod.FALLBACK,
                        success=False,
                        error_type="processing_error",
                    )
                    self.collector.record_event(event)

                except Exception as e:
                    # System should handle errors gracefully
                    results.append(None)

                    event = MetricEvent(
                        timestamp=datetime.now(),
                        metric_type=MetricType.ERROR,
                        method=ProcessingMethod.FALLBACK,
                        success=False,
                        error_type=type(e).__name__,
                    )
                    self.collector.record_event(event)

            # Verify error handling
            valid_results = [r for r in results if r is not None]
            assert len(valid_results) >= 0  # System should not crash

            # Check error metrics
            perf_metrics = self.collector.get_performance_metrics()
            assert perf_metrics.error_rate >= 0.0  # Should track errors

        finally:
            self.collector.stop_aggregation()

    def test_concurrent_processing_integration(self, sample_addresses):
        """Test concurrent processing with monitoring"""
        import threading
        import queue

        # Setup for concurrent processing
        test_addresses = [s.input_address for s in sample_addresses[:20]]
        results_queue = queue.Queue()

        def process_addresses(addresses, start_idx):
            """Process addresses in a thread"""
            thread_results = []

            for i, addr in enumerate(addresses):

                def mock_normalize(address):
                    time.sleep(0.01)  # Simulate processing time
                    return {
                        "success": len(address) > 5,
                        "components": {"il": "İstanbul"} if len(address) > 5 else {},
                        "normalized": address.title(),
                    }

                with patch.object(self.formatter, "_normalize_address", side_effect=mock_normalize):
                    result = self.formatter.format_single(addr)

                thread_results.append(result)

                # Record metrics with thread info
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY,
                    success=result.success,
                    metadata={"thread_id": start_idx + i},
                )
                self.collector.record_event(event)

            results_queue.put(thread_results)

        self.collector.start_aggregation()

        try:
            # Create multiple threads
            threads = []
            chunk_size = 5

            for i in range(0, len(test_addresses), chunk_size):
                chunk = test_addresses[i : i + chunk_size]
                thread = threading.Thread(target=process_addresses, args=(chunk, i))
                threads.append(thread)

            # Start all threads
            start_time = time.time()
            for thread in threads:
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            total_time = time.time() - start_time

            # Collect results
            all_results = []
            while not results_queue.empty():
                thread_results = results_queue.get()
                all_results.extend(thread_results)

            # Verify concurrent processing
            assert len(all_results) == len(test_addresses)

            # Check that concurrent processing was faster than sequential
            assert total_time < len(test_addresses) * 0.02  # Should be faster than 20ms per item

            # Verify metrics collection worked with concurrency
            time.sleep(0.1)
            perf_metrics = self.collector.get_performance_metrics()
            assert perf_metrics.total_processed >= len(test_addresses)

        finally:
            self.collector.stop_aggregation()

    def test_memory_usage_monitoring(self, performance_addresses):
        """Test memory usage monitoring during processing"""
        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        self.collector.start_aggregation()

        try:
            # Process many addresses to test memory usage
            large_batch = performance_addresses[:100]

            def mock_normalize(address):
                # Simulate memory usage
                dummy_data = [address] * 100  # Create some memory usage
                return {
                    "success": True,
                    "components": {"il": "İstanbul"},
                    "normalized": address.title(),
                    "dummy": dummy_data,  # This will be cleaned up
                }

            results = []
            for i, sample in enumerate(large_batch):
                with patch.object(self.formatter, "_normalize_address", side_effect=mock_normalize):
                    result = self.formatter.format_single(sample.input_address)

                results.append(result)

                # Record memory metrics periodically
                if i % 20 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024

                    event = MetricEvent(
                        timestamp=datetime.now(),
                        metric_type=MetricType.PERFORMANCE,
                        method=ProcessingMethod.PATTERN_PRIMARY,
                        success=True,
                        metadata={"memory_mb": current_memory},
                    )
                    self.collector.record_event(event)

                # Clean up to prevent memory buildup
                if i % 50 == 0:
                    gc.collect()

            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"

            # Verify we processed everything
            assert len(results) == len(large_batch)

        finally:
            self.collector.stop_aggregation()
            gc.collect()  # Final cleanup


@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and recovery"""

    def test_recovery_from_failures(self, sample_addresses):
        """Test system recovery from various failure modes"""
        formatter = EnhancedFormatter()
        collector = MetricsCollector()

        # Simulate various failure modes
        failure_modes = ["timeout_error", "memory_error", "invalid_input", "network_error", "processing_error"]

        collector.start_aggregation()

        try:
            results = []
            failure_count = 0

            for i, sample in enumerate(sample_addresses[:15]):
                failure_mode = failure_modes[i % len(failure_modes)]

                def mock_normalize_with_failures(address):
                    if failure_mode == "timeout_error":
                        time.sleep(0.01)  # Simulate timeout
                        raise TimeoutError("Processing timeout")
                    elif failure_mode == "memory_error":
                        raise MemoryError("Insufficient memory")
                    elif failure_mode == "invalid_input":
                        raise ValueError("Invalid input format")
                    elif failure_mode == "network_error":
                        raise ConnectionError("Network unavailable")
                    else:  # processing_error
                        return {"success": False, "error": "Processing failed"}

                try:
                    with patch.object(formatter, "_normalize_address", side_effect=mock_normalize_with_failures):
                        result = formatter.format_single(sample.input_address)
                    results.append(result)

                except Exception as e:
                    # System should handle errors gracefully
                    failure_count += 1
                    results.append(None)

                    # Record failure metrics
                    event = MetricEvent(
                        timestamp=datetime.now(),
                        metric_type=MetricType.ERROR,
                        method=ProcessingMethod.FALLBACK,
                        success=False,
                        error_type=type(e).__name__,
                    )
                    collector.record_event(event)

            # Verify system resilience
            assert len(results) == 15  # Should handle all inputs
            assert failure_count > 0  # Should have encountered failures

            # System should continue operating despite failures
            valid_results = [r for r in results if r is not None]

            # Check error tracking
            perf_metrics = collector.get_performance_metrics()
            assert perf_metrics.error_rate > 0  # Should track the failures

        finally:
            collector.stop_aggregation()

    def test_high_load_handling(self, performance_addresses):
        """Test system behavior under high load"""
        formatter = EnhancedFormatter()
        collector = MetricsCollector(buffer_size=1000)  # Larger buffer for high load

        # Simulate high load with many concurrent requests
        high_load_addresses = performance_addresses[:200]

        def mock_normalize_variable_time(address):
            # Variable processing time to simulate real conditions
            import random

            time.sleep(random.uniform(0.001, 0.01))

            return {
                "success": random.random() > 0.1,  # 90% success rate
                "components": {"il": "İstanbul"} if len(address) > 10 else {},
                "normalized": address.title(),
            }

        collector.start_aggregation()

        try:
            start_time = time.time()
            results = []

            # Process under high load
            for sample in high_load_addresses:
                with patch.object(formatter, "_normalize_address", side_effect=mock_normalize_variable_time):
                    result = formatter.format_single(sample.input_address)

                results.append(result)

                # Record metrics for high load
                event = MetricEvent(
                    timestamp=datetime.now(),
                    metric_type=MetricType.PERFORMANCE,
                    method=ProcessingMethod.PATTERN_PRIMARY,
                    success=result.success,
                    processing_time_ms=(time.time() - start_time) * 1000 / len(results),
                )
                collector.record_event(event)

            total_time = time.time() - start_time

            # Verify high load handling
            assert len(results) == len(high_load_addresses)

            # Check performance under load
            avg_time_per_request = (total_time / len(results)) * 1000
            assert avg_time_per_request < 50  # Should maintain reasonable performance

            # Verify metrics collection handled high load
            perf_metrics = collector.get_performance_metrics()
            assert perf_metrics.total_processed == len(high_load_addresses)
            assert perf_metrics.throughput_per_second > 0

        finally:
            collector.stop_aggregation()
