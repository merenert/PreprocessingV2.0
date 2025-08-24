"""
Test automation and reporting utilities
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
test_root = Path(__file__).parent.parent
src_path = test_root / "src"
sys.path.insert(0, str(src_path))


class TestRunner:
    """Automated test runner with reporting"""

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = datetime.now()

    def run_unit_tests(self, coverage: bool = True, parallel: bool = True) -> Dict[str, Any]:
        """Run unit tests with optional coverage and parallel execution"""
        print("🧪 Running unit tests...")

        cmd = ["python", "-m", "pytest", "tests/unit/", "-v"]

        if coverage:
            cmd.extend(["--cov=src/addrnorm", "--cov-report=html", "--cov-report=term-missing", "--cov-report=json"])

        if parallel:
            cmd.extend(["-n", "auto"])  # Use pytest-xdist for parallel execution

        # Add output file
        junit_file = self.output_dir / "unit_tests.xml"
        cmd.extend(["--junitxml", str(junit_file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            unit_results = {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": self._get_test_duration(result.stdout),
                "junit_file": str(junit_file),
            }

            if coverage:
                unit_results["coverage"] = self._parse_coverage_report()

            self.results["unit_tests"] = unit_results
            return unit_results

        except subprocess.TimeoutExpired:
            self.results["unit_tests"] = {"status": "timeout", "error": "Unit tests timed out after 5 minutes"}
            return self.results["unit_tests"]

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running integration tests...")

        cmd = ["python", "-m", "pytest", "tests/integration/", "-v", "--timeout=300"]

        junit_file = self.output_dir / "integration_tests.xml"
        cmd.extend(["--junitxml", str(junit_file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            integration_results = {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": self._get_test_duration(result.stdout),
                "junit_file": str(junit_file),
            }

            self.results["integration_tests"] = integration_results
            return integration_results

        except subprocess.TimeoutExpired:
            self.results["integration_tests"] = {"status": "timeout", "error": "Integration tests timed out after 10 minutes"}
            return self.results["integration_tests"]

    def run_benchmarks(self, save_results: bool = True) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("🚀 Running performance benchmarks...")

        cmd = ["python", "-m", "pytest", "tests/benchmarks/", "-v", "-m", "benchmark", "--timeout=600"]

        if save_results:
            benchmark_file = self.output_dir / "benchmark_results.txt"
            cmd.extend([f"--result-log={benchmark_file}"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

            benchmark_results = {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": self._get_test_duration(result.stdout),
            }

            if save_results:
                # Save detailed benchmark results
                with open(self.output_dir / "benchmark_output.txt", "w") as f:
                    f.write(result.stdout)
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)

                benchmark_results["output_file"] = str(self.output_dir / "benchmark_output.txt")

            self.results["benchmarks"] = benchmark_results
            return benchmark_results

        except subprocess.TimeoutExpired:
            self.results["benchmarks"] = {"status": "timeout", "error": "Benchmarks timed out after 20 minutes"}
            return self.results["benchmarks"]

    def run_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks"""
        print("✨ Running code quality checks...")

        quality_results = {}

        # Black formatting check
        try:
            result = subprocess.run(
                ["python", "-m", "black", "--check", "--diff", "src/", "tests/"], capture_output=True, text=True, timeout=60
            )
            quality_results["black"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout + result.stderr,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            quality_results["black"] = {"status": "skipped", "reason": "Black not available"}

        # isort import sorting check
        try:
            result = subprocess.run(
                ["python", "-m", "isort", "--check-only", "--diff", "src/", "tests/"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            quality_results["isort"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout + result.stderr,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            quality_results["isort"] = {"status": "skipped", "reason": "isort not available"}

        # Flake8 linting
        try:
            result = subprocess.run(
                ["python", "-m", "flake8", "src/", "tests/", "--max-line-length=127"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            quality_results["flake8"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout + result.stderr,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            quality_results["flake8"] = {"status": "skipped", "reason": "Flake8 not available"}

        self.results["quality_checks"] = quality_results
        return quality_results

    def run_all_tests(
        self, include_benchmarks: bool = True, include_quality: bool = True, coverage: bool = True
    ) -> Dict[str, Any]:
        """Run all tests and checks"""
        print(f"🎯 Starting comprehensive test suite at {self.start_time}")
        print("=" * 60)

        # Run tests in order
        self.run_unit_tests(coverage=coverage)
        self.run_integration_tests()

        if include_benchmarks:
            self.run_benchmarks()

        if include_quality:
            self.run_quality_checks()

        # Generate summary
        self.results["summary"] = self._generate_summary()

        # Save results
        self._save_results()

        print("=" * 60)
        print(f"✅ Test suite completed at {datetime.now()}")
        print(f"📊 Results saved to: {self.output_dir}")

        return self.results

    def _get_test_duration(self, output: str) -> Optional[float]:
        """Extract test duration from pytest output"""
        try:
            for line in output.split("\n"):
                if "seconds" in line and "===" in line:
                    # Look for pattern like "=== 5 passed in 2.34s ==="
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "seconds" in part or part.endswith("s"):
                            return float(part.replace("s", ""))
        except:
            pass
        return None

    def _parse_coverage_report(self) -> Optional[Dict[str, Any]]:
        """Parse coverage report if available"""
        try:
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    return {
                        "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                        "files": len(coverage_data.get("files", {})),
                        "covered_lines": coverage_data.get("totals", {}).get("covered_lines", 0),
                        "total_lines": coverage_data.get("totals", {}).get("num_statements", 0),
                    }
        except:
            pass
        return None

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test suite summary"""
        total_duration = (datetime.now() - self.start_time).total_seconds()

        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "test_results": {},
        }

        # Summarize each test category
        for test_type, results in self.results.items():
            if test_type != "summary" and isinstance(results, dict):
                summary["test_results"][test_type] = {
                    "status": results.get("status", "unknown"),
                    "duration": results.get("duration"),
                    "returncode": results.get("returncode"),
                }

        # Overall status
        all_passed = all(result.get("status") == "passed" for result in summary["test_results"].values())
        summary["overall_status"] = "passed" if all_passed else "failed"

        return summary

    def _save_results(self):
        """Save test results to JSON file"""
        results_file = self.output_dir / "test_results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        # Generate HTML report
        self._generate_html_report()

    def _generate_html_report(self):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Results Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .failed {{ background: #ffe8e8; }}
        .test-section {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .test-header {{ background: #f8f9fa; padding: 10px; font-weight: bold; }}
        .test-content {{ padding: 15px; }}
        .status-passed {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .status-timeout {{ color: orange; font-weight: bold; }}
        .status-skipped {{ color: gray; font-weight: bold; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Test Results Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {self.results.get('summary', {}).get('total_duration_seconds', 0):.1f} seconds</p>
    </div>

    <div class="summary {'failed' if self.results.get('summary', {}).get('overall_status') != 'passed' else ''}">
        <h2>📊 Summary</h2>
        <p><strong>Overall Status:</strong>
           <span class="status-{self.results.get('summary', {}).get('overall_status', 'unknown')}">
               {self.results.get('summary', {}).get('overall_status', 'unknown').upper()}
           </span>
        </p>
    </div>
"""

        # Add each test section
        for test_type, results in self.results.items():
            if test_type == "summary":
                continue

            if isinstance(results, dict):
                status = results.get("status", "unknown")
                html_content += f"""
    <div class="test-section">
        <div class="test-header">
            {test_type.replace('_', ' ').title()} -
            <span class="status-{status}">{status.upper()}</span>
        </div>
        <div class="test-content">
"""

                if "duration" in results and results["duration"]:
                    html_content += f"<p><strong>Duration:</strong> {results['duration']:.2f} seconds</p>"

                if "coverage" in results:
                    cov = results["coverage"]
                    html_content += f"<p><strong>Coverage:</strong> {cov.get('total_coverage', 0):.1f}%</p>"

                if "stdout" in results and results["stdout"]:
                    html_content += f"<h4>Output:</h4><pre>{results['stdout'][:2000]}{'...' if len(results['stdout']) > 2000 else ''}</pre>"

                if "stderr" in results and results["stderr"]:
                    html_content += f"<h4>Errors:</h4><pre>{results['stderr'][:1000]}{'...' if len(results['stderr']) > 1000 else ''}</pre>"

                html_content += "</div></div>"

        html_content += """
</body>
</html>
"""

        # Save HTML report
        html_file = self.output_dir / "test_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📋 HTML report generated: {html_file}")


def main():
    """Command line interface for test automation"""
    parser = argparse.ArgumentParser(description="Run automated test suite")
    parser.add_argument("--no-benchmarks", action="store_true", help="Skip performance benchmarks")
    parser.add_argument("--no-quality", action="store_true", help="Skip code quality checks")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for test results")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--benchmarks-only", action="store_true", help="Run only benchmarks")

    args = parser.parse_args()

    runner = TestRunner(output_dir=args.output_dir)

    try:
        if args.unit_only:
            runner.run_unit_tests(coverage=not args.no_coverage)
        elif args.integration_only:
            runner.run_integration_tests()
        elif args.benchmarks_only:
            runner.run_benchmarks()
        else:
            runner.run_all_tests(
                include_benchmarks=not args.no_benchmarks, include_quality=not args.no_quality, coverage=not args.no_coverage
            )

        # Print summary
        summary = runner.results.get("summary", {})
        print(f"\n{'='*60}")
        print(f"🎯 TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {summary.get('overall_status', 'unknown').upper()}")
        print(f"Total Duration: {summary.get('total_duration_seconds', 0):.1f} seconds")

        for test_type, result in summary.get("test_results", {}).items():
            status = result.get("status", "unknown")
            duration = result.get("duration", 0) or 0
            print(f"{test_type.replace('_', ' ').title()}: {status.upper()} ({duration:.1f}s)")

        print(f"\n📁 Results saved to: {runner.output_dir}")

        # Exit with appropriate code
        if summary.get("overall_status") == "passed":
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n❌ Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
