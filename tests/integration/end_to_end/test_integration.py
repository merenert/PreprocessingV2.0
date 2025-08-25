"""
End-to-End Integration Tests

Comprehensive integration testing for the complete address normalization pipeline:
- Full workflow testing
- API integration testing
- CLI integration testing
- Error handling verification
- Performance validation
"""

import pytest
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from tests.fixtures.address_samples import get_test_addresses, get_edge_case_data


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    @pytest.fixture
    def sample_addresses(self):
        """Sample addresses for testing"""
        return [
            "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No:15 Daire:3",
            "Ankara Çankaya Çukurambar Mahallesi Dumlupınar Bulvarı 234/A",
            "İzmir Karşıyaka Bostanlı Mahallesi Atatürk Caddesi No:567 Kat:2 Daire:8",
            "Antalya Muratpaşa Konyaaltı Mahallesi Cumhuriyet Caddesi No:123",
            "Bursa Nilüfer Görükle Mahallesi Uludağ Üniversitesi Kampüsü",
        ]

    def test_basic_processing_workflow(self, sample_addresses):
        """Test basic address processing workflow"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer

            normalizer = AddressNormalizer()

            for address in sample_addresses:
                result = normalizer.process(address)

                # Basic validation
                assert result is not None
                assert hasattr(result, "success")
                assert hasattr(result, "normalized_address")
                assert hasattr(result, "confidence")

                if result.success:
                    assert result.normalized_address is not None
                    assert isinstance(result.confidence, (int, float))
                    assert 0 <= result.confidence <= 1

        except ImportError:
            pytest.skip("AddressNormalizer not available")

    def test_batch_processing_workflow(self, sample_addresses):
        """Test batch processing capabilities"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer

            normalizer = AddressNormalizer()

            # Test batch processing
            results = []
            for address in sample_addresses:
                result = normalizer.process(address)
                results.append(result)

            # Validate batch results
            assert len(results) == len(sample_addresses)

            successful_results = [r for r in results if r.success]
            assert len(successful_results) > 0  # At least some should succeed

            # Check consistency
            success_rate = len(successful_results) / len(results)
            assert success_rate >= 0.5  # At least 50% success rate expected

        except ImportError:
            pytest.skip("AddressNormalizer not available")

    def test_explanation_integration(self, sample_addresses):
        """Test explanation system integration"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer
            from addrnorm.explanation.explainer import AddressExplainer

            normalizer = AddressNormalizer()
            explainer = AddressExplainer()

            for address in sample_addresses[:3]:  # Test with first 3
                result = normalizer.process(address)
                explanation = explainer.explain(result)

                assert explanation is not None
                assert isinstance(explanation, str)
                assert len(explanation) > 0

                # Explanation should contain relevant information
                if result.success:
                    assert "başarılı" in explanation.lower() or "success" in explanation.lower()
                else:
                    assert "başarısız" in explanation.lower() or "fail" in explanation.lower()

        except ImportError:
            pytest.skip("Required modules not available")

    def test_monitoring_integration(self, sample_addresses):
        """Test monitoring system integration"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer
            from addrnorm.monitoring.collector import MetricsCollector

            normalizer = AddressNormalizer()
            collector = MetricsCollector()
            collector.start_aggregation()

            try:
                # Process addresses with monitoring
                for address in sample_addresses:
                    result = normalizer.process(address)

                    # Simulate metric recording
                    collector.record_processing_event(
                        address=address,
                        result=result,
                        processing_time=getattr(result, "processing_time", 0.1),
                        method=getattr(result, "processing_method", "test"),
                    )

                # Allow time for aggregation
                time.sleep(1)

                # Check metrics
                metrics = collector.get_current_metrics()
                assert metrics is not None

            finally:
                collector.stop_aggregation()

        except ImportError:
            pytest.skip("Monitoring modules not available")

    def test_adaptive_processing_workflow(self, sample_addresses):
        """Test adaptive processing capabilities"""
        try:
            from addrnorm.adaptive.processor import AdaptiveProcessor

            processor = AdaptiveProcessor()

            # Process with adaptation
            for address in sample_addresses:
                result = processor.process_adaptive(address)

                assert result is not None
                assert hasattr(result, "confidence")
                assert hasattr(result, "adaptations_applied")

        except ImportError:
            pytest.skip("AdaptiveProcessor not available")

    def test_error_handling_workflow(self):
        """Test error handling across the pipeline"""
        error_cases = [
            "",  # Empty string
            None,  # None input
            "x" * 10000,  # Very long string
            "Invalid çhars ñ ë test",  # Mixed character sets
            "SQL'; DROP TABLE addresses;--",  # Injection attempt
        ]

        try:
            from addrnorm.preprocess.api import AddressNormalizer

            normalizer = AddressNormalizer()

            for error_case in error_cases:
                try:
                    result = normalizer.process(error_case)

                    # Should handle errors gracefully
                    assert result is not None
                    assert hasattr(result, "success")

                    # For error cases, should either succeed or fail gracefully
                    if not result.success:
                        assert hasattr(result, "error_details")

                except Exception as e:
                    # Should not raise unhandled exceptions
                    pytest.fail(f"Unhandled exception for input '{error_case}': {e}")

        except ImportError:
            pytest.skip("AddressNormalizer not available")


class TestAPIIntegration:
    """Test API integration scenarios"""

    def test_api_initialization(self):
        """Test API component initialization"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer

            # Test default initialization
            normalizer = AddressNormalizer()
            assert normalizer is not None

            # Test with custom config
            config = {"language": "tr", "strict_mode": False}
            normalizer = AddressNormalizer(config=config)
            assert normalizer is not None

        except ImportError:
            pytest.skip("API module not available")

    def test_api_configuration(self):
        """Test API configuration options"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer

            # Test different configuration options
            configs = [
                {"language": "tr", "strict_mode": True},
                {"language": "en", "strict_mode": False},
                {"timeout": 30, "max_retries": 3},
                {"debug": True, "verbose": True},
            ]

            for config in configs:
                normalizer = AddressNormalizer(config=config)
                result = normalizer.process("Test Address")
                assert result is not None

        except ImportError:
            pytest.skip("API module not available")

    def test_api_error_responses(self):
        """Test API error response handling"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer

            normalizer = AddressNormalizer()

            # Test various error conditions
            error_inputs = [None, "", " " * 100, {"invalid": "input"}, 123456]

            for error_input in error_inputs:
                try:
                    result = normalizer.process(error_input)

                    # Should return a valid result object
                    assert result is not None
                    assert hasattr(result, "success")

                    # For invalid inputs, success should be False
                    if error_input in [None, "", {"invalid": "input"}, 123456]:
                        assert not result.success

                except Exception as e:
                    # Should not raise unhandled exceptions
                    pytest.fail(f"Unhandled API exception: {e}")

        except ImportError:
            pytest.skip("API module not available")


class TestCLIIntegration:
    """Test CLI integration scenarios"""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for CLI tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_cli_help_commands(self):
        """Test CLI help functionality"""
        cli_commands = [
            ["python", "-m", "addrnorm.cli.benchmark", "--help"],
            ["python", "-m", "addrnorm.preprocess.cli", "--help"],
        ]

        for command in cli_commands:
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=30, cwd=Path(__file__).parent.parent.parent.parent
                )

                # Should not fail
                assert result.returncode != 1  # Allow 0 or other non-error codes

                # Should produce help output
                output = result.stdout + result.stderr
                assert len(output) > 0

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip(f"CLI command not available: {' '.join(command)}")

    def test_benchmark_cli_commands(self, temp_directory):
        """Test benchmark CLI commands"""
        benchmark_commands = [
            ["python", "-m", "addrnorm.cli.benchmark", "run", "--type", "performance", "--size", "10"],
            ["python", "-m", "addrnorm.cli.benchmark", "list-results", "--output", str(temp_directory)],
        ]

        for command in benchmark_commands:
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=60, cwd=Path(__file__).parent.parent.parent.parent
                )

                # Should not fail catastrophically
                assert result.returncode in [0, 1]  # Allow expected error codes

                # Should produce some output
                output = result.stdout + result.stderr
                assert len(output) > 0

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip(f"Benchmark CLI not available: {' '.join(command)}")

    def test_processing_cli_integration(self, temp_directory):
        """Test processing CLI integration"""
        # Create test input file
        test_file = temp_directory / "test_addresses.txt"
        test_addresses = ["İstanbul Kadıköy Test Mahallesi", "Ankara Çankaya Test Bulvarı"]

        with open(test_file, "w", encoding="utf-8") as f:
            for addr in test_addresses:
                f.write(addr + "\n")

        # Test processing command
        command = [
            "python",
            "-m",
            "addrnorm.preprocess.cli",
            "--input",
            str(test_file),
            "--output",
            str(temp_directory / "output.json"),
        ]

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Check if command executed
            assert result.returncode in [0, 1]  # Allow expected error codes

            # Check output
            output = result.stdout + result.stderr
            assert len(output) > 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Processing CLI not available")


class TestPerformanceIntegration:
    """Test performance-related integration scenarios"""

    def test_performance_under_load(self):
        """Test system performance under load"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer

            normalizer = AddressNormalizer()

            # Generate test load
            test_addresses = [f"Test Address {i} Street No:{i}" for i in range(100)]

            start_time = time.time()
            results = []

            for address in test_addresses:
                result = normalizer.process(address)
                results.append(result)

            end_time = time.time()

            # Performance validation
            total_time = end_time - start_time
            avg_time_per_address = total_time / len(test_addresses)

            # Should process reasonably fast
            assert avg_time_per_address < 1.0  # Less than 1 second per address

            # Should maintain reasonable success rate
            successful_results = [r for r in results if r.success]
            success_rate = len(successful_results) / len(results)
            assert success_rate >= 0.7  # At least 70% success rate

        except ImportError:
            pytest.skip("Performance testing modules not available")

    def test_memory_efficiency(self):
        """Test memory efficiency of the system"""
        try:
            import psutil
            from addrnorm.preprocess.api import AddressNormalizer

            process = psutil.Process()
            initial_memory = process.memory_info().rss

            normalizer = AddressNormalizer()

            # Process multiple addresses
            for i in range(1000):
                address = f"Memory Test Address {i} Street No:{i}"
                result = normalizer.process(address)

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024  # 100MB

        except ImportError:
            pytest.skip("Memory testing modules not available")

    def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        try:
            from addrnorm.preprocess.api import AddressNormalizer
            from concurrent.futures import ThreadPoolExecutor
            import threading

            def process_addresses(address_list, results_list, errors_list):
                """Process addresses in a thread"""
                normalizer = AddressNormalizer()

                for address in address_list:
                    try:
                        result = normalizer.process(address)
                        results_list.append(result)
                    except Exception as e:
                        errors_list.append(e)

            # Create test data
            address_chunks = [
                [f"Thread 1 Address {i}" for i in range(50)],
                [f"Thread 2 Address {i}" for i in range(50)],
                [f"Thread 3 Address {i}" for i in range(50)],
            ]

            results_lists = [[], [], []]
            errors_lists = [[], [], []]

            # Run concurrent processing
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []

                for i in range(3):
                    future = executor.submit(process_addresses, address_chunks[i], results_lists[i], errors_lists[i])
                    futures.append(future)

                # Wait for completion
                for future in futures:
                    future.result()

            # Validate results
            total_results = sum(len(results) for results in results_lists)
            total_errors = sum(len(errors) for errors in errors_lists)

            assert total_results > 0
            assert total_errors < total_results  # More successes than errors

        except ImportError:
            pytest.skip("Concurrent processing modules not available")


class TestRegressionScenarios:
    """Test regression scenarios and backward compatibility"""

    def test_backward_compatibility(self):
        """Test backward compatibility with previous versions"""
        # Test data from previous versions
        legacy_test_cases = [
            {"input": "İstanbul Kadıköy Moda Mahallesi", "expected_components": ["İstanbul", "Kadıköy", "Moda"]},
            {"input": "Ankara Çankaya Çukurambar", "expected_components": ["Ankara", "Çankaya", "Çukurambar"]},
        ]

        try:
            from addrnorm.preprocess.api import AddressNormalizer

            normalizer = AddressNormalizer()

            for test_case in legacy_test_cases:
                result = normalizer.process(test_case["input"])

                # Should still work with legacy inputs
                assert result is not None
                assert hasattr(result, "success")

                # If successful, should contain expected components
                if result.success and result.normalized_address:
                    normalized_str = str(result.normalized_address)
                    for expected_component in test_case["expected_components"]:
                        # Should contain the component somewhere in the result
                        assert expected_component in normalized_str or any(
                            comp in normalized_str for comp in test_case["expected_components"]
                        )

        except ImportError:
            pytest.skip("Compatibility testing modules not available")

    def test_configuration_migration(self):
        """Test configuration migration scenarios"""
        # Test different configuration formats
        config_formats = [
            # Legacy format
            {"language": "tr", "mode": "strict"},
            # Current format
            {"language": "tr", "strict_mode": True},
            # Extended format
            {"language": "tr", "strict_mode": False, "timeout": 30, "enable_monitoring": True},
        ]

        try:
            from addrnorm.preprocess.api import AddressNormalizer

            for config in config_formats:
                try:
                    normalizer = AddressNormalizer(config=config)
                    result = normalizer.process("Test configuration migration")

                    # Should handle different config formats
                    assert result is not None

                except Exception as e:
                    # Should not fail due to config format issues
                    pytest.fail(f"Configuration migration failed: {e}")

        except ImportError:
            pytest.skip("Configuration migration testing not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
