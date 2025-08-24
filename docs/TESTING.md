# Test Suite & Benchmark System

Bu dokümantasyon, Address Normalization sisteminin kapsamlı test suite ve benchmark sistemini açıklar.

## 📋 Test Suite Genel Bakış

Test suite şu bileşenlerden oluşur:

### 🧪 Test Kategorileri

1. **Unit Tests** (`tests/unit/`)
   - Confidence scoring sistemi testleri
   - Quality assessment testleri
   - Enhanced formatter testleri
   - Monitoring system testleri

2. **Integration Tests** (`tests/integration/`)
   - Full pipeline integration testleri
   - End-to-end işlem testleri
   - Sistem resilience testleri
   - Concurrent processing testleri

3. **Benchmark Tests** (`tests/benchmarks/`)
   - Performance benchmark testleri
   - Memory efficiency testleri
   - Stress testing
   - Accuracy benchmarks

### 📊 Test Fixtures & Data

- **Address Samples** (`tests/fixtures/address_samples.py`)
  - 1000+ Turkish address sample
  - Kategorize edilmiş test verileri (residential, commercial, landmark, edge cases)
  - Performance test verileri

## 🚀 Test Çalıştırma

### Hızlı Test

```bash
# Tüm testleri çalıştır
python run_tests.py

# Sadece unit testler
python -m pytest tests/unit/ -v

# Sadece integration testler
python -m pytest tests/integration/ -v

# Benchmark testleri
python -m pytest tests/benchmarks/ -v -m benchmark
```

### Kapsamlı Test Automation

```bash
# Test automation script kullanarak
python tests/test_automation.py

# Sadece unit testler
python tests/test_automation.py --unit-only

# Benchmark olmadan
python tests/test_automation.py --no-benchmarks

# Özel output directory
python tests/test_automation.py --output-dir custom_results
```

### Coverage Raporu

```bash
# Coverage ile test çalıştır
pytest tests/unit/ --cov=src/addrnorm --cov-report=html

# HTML coverage raporu görüntüle
# htmlcov/index.html dosyasını browser'da aç
```

## 📈 Performance Benchmarks

### Benchmark Kategorileri

1. **Single Address Performance**
   - Average processing time: <10ms
   - P95 processing time: <20ms
   - Throughput: >50 addresses/second

2. **Batch Processing**
   - Optimized batch processing
   - Improved throughput with larger batches
   - Memory efficient processing

3. **Concurrent Processing**
   - Multi-threaded performance testing
   - Scalability analysis
   - Resource utilization benchmarks

4. **Memory Efficiency**
   - Memory usage monitoring
   - Memory leak detection
   - Resource optimization validation

### Benchmark Çalıştırma

```bash
# Tüm benchmark testler
pytest tests/benchmarks/ -v -m benchmark --timeout=600

# Specific benchmark testler
pytest tests/benchmarks/test_performance_benchmarks.py::TestPerformanceBenchmarks::test_single_address_performance -v

# Memory benchmarks (psutil gerekli)
pip install psutil
pytest tests/benchmarks/ -v -m memory
```

## 🔍 Code Quality Checks

### Otomatik Quality Checks

```bash
# Code formatting
black --check src/ tests/

# Import sorting
isort --check-only src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Quality Tools Configuration

- **Black**: Code formatting (127 char line length)
- **isort**: Import sorting
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking
- **Bandit**: Security scanning

## 🤖 CI/CD Integration

### GitHub Actions

Test suite GitHub Actions ile tamamen otomatize edilmiştir:

1. **Test Workflow** (`.github/workflows/test-and-benchmark.yml`)
   - Python 3.8-3.11 matrix testing
   - Unit ve integration testler
   - Coverage reporting
   - Quality checks

2. **Benchmark Workflow**
   - Scheduled daily benchmarks
   - Performance regression detection
   - Benchmark result artifacts

3. **Security Scanning**
   - Dependency vulnerability scanning
   - Security linting

### Local CI Simulation

```bash
# Tüm CI checks simulate et
python tests/test_automation.py

# Quality checks
python tests/test_automation.py --no-benchmarks --no-integration
```

## 📊 Test Results & Reporting

### Test Output Formats

1. **Console Output**: Real-time test results
2. **HTML Report**: Detailed test report (`test_results/test_report.html`)
3. **JUnit XML**: CI/CD integration için
4. **JSON Report**: Programmatic access için

### Coverage Reports

- **Terminal**: Real-time coverage feedback
- **HTML**: Interactive coverage report
- **XML**: CI/CD integration için

### Benchmark Results

- **JSON**: Machine-readable benchmark data
- **Text**: Human-readable benchmark summary
- **Artifacts**: CI/CD system artifacts

## 🔧 Test Configuration

### PyTest Configuration

Test configuration `pyproject.toml` dosyasında:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "benchmark: Performance benchmark tests"
]
timeout = 300
addopts = [
    "--strict-markers",
    "--cov=src/addrnorm",
    "--cov-fail-under=80"
]
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["src/addrnorm"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "def __repr__"]
fail_under = 80
```

## 🎯 Test Development Guidelines

### Test Yazma Kuralları

1. **Unit Tests**
   - Her public method için test
   - Edge cases ve error conditions
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests**
   - End-to-end scenarios
   - Real system interactions
   - Data flow validation
   - Performance within limits

3. **Benchmark Tests**
   - Consistent test environment
   - Statistical significance
   - Performance regression detection
   - Resource usage monitoring

### Test Data Management

```python
# Address samples kullanımı
from tests.fixtures.address_samples import (
    RESIDENTIAL_SAMPLES,
    COMMERCIAL_SAMPLES,
    get_performance_addresses
)

def test_function(sample_addresses):
    for sample in sample_addresses[:10]:
        # Test logic
        pass
```

### Mock Usage

```python
# Service mocking
with patch.object(formatter, '_normalize_address') as mock_normalize:
    mock_normalize.return_value = {'success': True}
    result = formatter.format_single(address)
```

## 📝 Test Maintenance

### Test Data Updates

1. Address samples güncelleme
2. New test categories ekleme
3. Performance baseline adjustments

### Benchmark Baseline Updates

```bash
# Yeni baseline oluştur
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Baseline ile karşılaştır
pytest tests/benchmarks/ --benchmark-compare=baseline
```

### Test Environment Setup

```bash
# Test dependencies install
pip install -r requirements-test.txt

# Test environment validation
python -c "import pytest, coverage, black, flake8; print('Test environment ready')"
```

## 🏃‍♂️ Performance Targets

### Minimum Performance Standards

- **Unit Tests**: <5 seconds total execution
- **Integration Tests**: <5 minutes total execution
- **Single Address Processing**: <10ms average
- **Batch Processing**: >100 addresses/second
- **Memory Usage**: <100MB increase during processing
- **Test Coverage**: >80% line coverage

### Benchmark Thresholds

```python
# Performance assertions in tests
assert avg_processing_time < 10.0  # ms
assert throughput > 50  # addresses/second
assert memory_growth < 50  # MB
assert success_rate > 0.8  # 80%
```

Bu comprehensive test suite, Address Normalization sisteminin quality, performance ve reliability'sini garanti eder.
