#!/usr/bin/env python3
"""
Quick test runner script for development
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description, timeout=300):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return 124
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return 1


def main():
    """Main test runner"""
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("🧪 Address Normalization Test Suite")
    print(f"📁 Working directory: {project_root}")

    # Check if virtual environment is active
    if not os.environ.get("VIRTUAL_ENV"):
        print("⚠️  Warning: No virtual environment detected")
        print("   Consider activating a virtual environment first")

    # Install test dependencies if needed
    print("\n📦 Installing test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"], check=False)

    # Run tests
    exit_codes = []

    # 1. Unit tests
    exit_codes.append(
        run_command([sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short", "--maxfail=3"], "Unit Tests")
    )

    # 2. Integration tests
    exit_codes.append(
        run_command(
            [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short", "--timeout=300"], "Integration Tests"
        )
    )

    # 3. Code quality checks
    exit_codes.append(
        run_command([sys.executable, "-m", "black", "--check", "--diff", "src/", "tests/"], "Code Formatting Check")
    )

    exit_codes.append(
        run_command([sys.executable, "-m", "isort", "--check-only", "--diff", "src/", "tests/"], "Import Sorting Check")
    )

    exit_codes.append(run_command([sys.executable, "-m", "flake8", "src/", "tests/"], "Linting Check"))

    # 4. Type checking (optional)
    exit_codes.append(run_command([sys.executable, "-m", "mypy", "src/"], "Type Checking", timeout=120))

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    test_names = ["Unit Tests", "Integration Tests", "Code Formatting", "Import Sorting", "Linting", "Type Checking"]

    passed = 0
    failed = 0

    for i, (name, code) in enumerate(zip(test_names, exit_codes)):
        status = "✅ PASSED" if code == 0 else "❌ FAILED"
        print(f"{name:<20} {status}")
        if code == 0:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Test run interrupted by user")
        sys.exit(130)
