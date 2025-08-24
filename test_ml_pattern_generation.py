"""
ML Pattern Generation Tests

ML tabanlı pattern generation sistemi için unit test'ler.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from addrnorm.pattern_generation import (
    create_ml_pattern_system,
    MLPatternConfig,
    ClusteringAlgorithm,
    PatternType,
    ValidationStatus,
)


class TestMLPatternGeneration(unittest.TestCase):
    """
    ML Pattern Generation System test sınıfı
    """

    def setUp(self):
        """Test setup"""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

        # Test config
        self.config = MLPatternConfig(
            clustering_algorithm=ClusteringAlgorithm.KMEANS,
            n_clusters=3,
            min_cluster_size=2,
            confidence_threshold=0.5,
            quality_threshold=0.6,
        )

        # Create system
        self.system = create_ml_pattern_system(config=self.config, output_dir=self.test_dir)

        # Sample addresses
        self.sample_addresses = [
            "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
            "İstanbul Kadıköy Fenerbahçe Mahallesi Bağdat Caddesi No: 123",
            "Ankara Çankaya Kızılay Mahallesi Atatürk Bulvarı No: 89",
            "Ankara Çankaya Kavaklidere Mahallesi Tunalı Caddesi No: 156",
            "İzmir Konak Alsancak Mahallesi Cumhuriyet Bulvarı No: 45",
            "İzmir Bornova Erzene Mahallesi İzmir Yolu Caddesi No: 178",
        ]

        # Sample existing patterns
        self.existing_patterns = [
            r"^(\w+)\s+(\w+)\s+(\w+)\s+Mahallesi\s+(\w+)\s+Caddesi\s+No:\s*(\d+)$",
            r"^(\w+)\s+(\w+)\s+(\w+)\s+Mahallesi\s+(\w+)\s+Bulvarı\s+No:\s*(\d+)$",
        ]

    def test_system_creation(self):
        """Test system creation"""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.ml_suggester)
        self.assertIsNotNone(self.system.validator)
        self.assertIsNotNone(self.system.conflict_detector)
        self.assertIsNotNone(self.system.review_interface)

    def test_pattern_generation(self):
        """Test pattern generation"""
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=5
        )

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

        # Check pattern structure
        for pattern in patterns:
            self.assertIsNotNone(pattern.pattern_id)
            self.assertIsNotNone(pattern.regex_pattern)
            self.assertIsNotNone(pattern.template)
            self.assertIsInstance(pattern.confidence, float)
            self.assertGreaterEqual(pattern.confidence, 0.0)
            self.assertLessEqual(pattern.confidence, 1.0)

    def test_pattern_validation(self):
        """Test pattern validation"""
        # Generate patterns first
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=3
        )

        # Validate patterns
        validation_results = self.system.validate_patterns(patterns=patterns, existing_patterns=self.existing_patterns)

        self.assertEqual(len(validation_results), len(patterns))

        # Check validation structure
        for result in validation_results:
            self.assertIsNotNone(result.pattern_id)
            self.assertIsInstance(result.is_valid, bool)
            self.assertIsInstance(result.quality_score, float)
            self.assertGreaterEqual(result.quality_score, 0.0)
            self.assertLessEqual(result.quality_score, 1.0)

    def test_conflict_detection(self):
        """Test conflict detection"""
        # Generate patterns first
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=3
        )

        # Detect conflicts
        conflict_reports = self.system.detect_conflicts(patterns=patterns, existing_patterns=self.existing_patterns)

        self.assertIsInstance(conflict_reports, list)

        # Check conflict structure
        for conflict in conflict_reports:
            self.assertIsNotNone(conflict.pattern_id)
            self.assertIsNotNone(conflict.severity)
            self.assertIsInstance(conflict.overlap_score, float)
            self.assertGreaterEqual(conflict.overlap_score, 0.0)
            self.assertLessEqual(conflict.overlap_score, 1.0)

    def test_batch_review(self):
        """Test batch review"""
        # Generate patterns first
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=3
        )

        # Validate patterns
        validation_results = self.system.validate_patterns(patterns=patterns)

        # Batch review
        review_decisions = self.system.review_patterns(
            patterns=patterns, validation_results=validation_results, reviewer_name="test_reviewer", batch_mode=True
        )

        self.assertEqual(len(review_decisions), len(patterns))

        # Check review structure
        for decision in review_decisions:
            self.assertIsNotNone(decision.pattern_id)
            self.assertIsNotNone(decision.decision)
            self.assertEqual(decision.reviewer, "batch_auto_reviewer")
            self.assertIsInstance(decision.priority, int)

    def test_approved_patterns(self):
        """Test getting approved patterns"""
        # Generate and review patterns
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=3
        )

        validation_results = self.system.validate_patterns(patterns=patterns)

        review_decisions = self.system.review_patterns(
            patterns=patterns, validation_results=validation_results, batch_mode=True
        )

        # Get approved patterns
        approved_patterns = self.system.get_approved_patterns()

        self.assertIsInstance(approved_patterns, list)

        # Check that approved patterns are subset of generated patterns
        approved_ids = {p.pattern_id for p in approved_patterns}
        generated_ids = {p.pattern_id for p in patterns}
        self.assertTrue(approved_ids.issubset(generated_ids))

    def test_full_pipeline(self):
        """Test full pipeline"""
        results = self.system.full_pipeline(
            address_samples=self.sample_addresses,
            existing_patterns=self.existing_patterns,
            reviewer_name="test_reviewer",
            batch_review=True,
            auto_export=True,
        )

        # Check results structure
        self.assertIn("pipeline_duration", results)
        self.assertIn("total_patterns_generated", results)
        self.assertIn("approved_patterns", results)
        self.assertIn("system_metrics", results)
        self.assertIn("report_file", results)

        # Check that some patterns were generated
        self.assertGreater(results["total_patterns_generated"], 0)

        # Check that report file was created
        if results["report_file"]:
            self.assertTrue(Path(results["report_file"]).exists())

    def test_system_metrics(self):
        """Test system metrics tracking"""
        # Generate patterns to update metrics
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=3
        )

        validation_results = self.system.validate_patterns(patterns=patterns)

        review_decisions = self.system.review_patterns(
            patterns=patterns, validation_results=validation_results, batch_mode=True
        )

        # Check metrics
        metrics = self.system.system_metrics

        self.assertGreater(metrics["total_addresses_processed"], 0)
        self.assertGreater(metrics["patterns_generated"], 0)
        self.assertGreaterEqual(metrics["avg_generation_time"], 0.0)
        self.assertGreaterEqual(metrics["avg_validation_time"], 0.0)

    def test_system_reset(self):
        """Test system state reset"""
        # Generate some data
        patterns = self.system.generate_patterns_from_addresses(
            address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=3
        )

        # Check that system has data
        self.assertGreater(len(self.system.generated_patterns), 0)

        # Reset system
        self.system.reset_system_state()

        # Check that system is clean
        self.assertEqual(len(self.system.generated_patterns), 0)
        self.assertEqual(len(self.system.validation_results), 0)
        self.assertEqual(len(self.system.conflict_reports), 0)
        self.assertEqual(len(self.system.review_decisions), 0)


class TestMLPatternComponents(unittest.TestCase):
    """
    Individual component tests
    """

    def setUp(self):
        """Test setup"""
        self.config = MLPatternConfig(clustering_algorithm=ClusteringAlgorithm.KMEANS, n_clusters=2, min_cluster_size=2)

        self.sample_addresses = [
            "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
            "İstanbul Kadıköy Fenerbahçe Mahallesi Bağdat Caddesi No: 123",
            "Ankara Çankaya Kızılay Mahallesi Atatürk Bulvarı No: 89",
        ]

    def test_ml_suggester(self):
        """Test ML suggester component"""
        from addrnorm.pattern_generation.ml_suggester import MLPatternSuggester

        suggester = MLPatternSuggester(self.config)
        patterns = suggester.generate_patterns(address_samples=self.sample_addresses, min_cluster_size=2, max_patterns=5)

        self.assertIsInstance(patterns, list)

    def test_validator(self):
        """Test validator component"""
        from addrnorm.pattern_generation.validator import PatternValidator
        from addrnorm.pattern_generation.models import PatternSuggestion, PatternTemplate, ClusterResult

        validator = PatternValidator()

        # Create a mock pattern
        pattern = PatternSuggestion(
            pattern_id="test_pattern",
            pattern_type=PatternType.STREET_LEVEL,
            regex_pattern=r"test_pattern",
            template=PatternTemplate(
                template="test template", components=["city", "district"], complexity_score=0.5, generalizability=0.7
            ),
            confidence=0.8,
            coverage=0.6,
            source_cluster=ClusterResult(cluster_id=0, size=3, addresses=self.sample_addresses),
            examples=self.sample_addresses[:2],
            quality_score=0.7,
        )

        result = validator.validate_pattern(pattern)

        self.assertIsNotNone(result)
        self.assertIsInstance(result.is_valid, bool)
        self.assertIsInstance(result.quality_score, float)

    def test_conflict_detector(self):
        """Test conflict detector component"""
        from addrnorm.pattern_generation.conflict_detector import ConflictDetector

        detector = ConflictDetector()

        # Mock patterns
        patterns = [
            {"pattern_id": "pattern_1", "regex_pattern": r"^(\w+)\s+(\w+)\s+(\w+)\s+Mahallesi", "is_existing": False},
            {"pattern_id": "pattern_2", "regex_pattern": r"^(\w+)\s+(\w+)\s+(\w+)\s+Mah\.", "is_existing": False},
        ]

        conflicts = detector.detect_conflicts(patterns)

        self.assertIsInstance(conflicts, list)


def run_tests():
    """
    Test'leri çalıştır
    """
    print("🧪 Running ML Pattern Generation Tests...")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMLPatternGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPatternComponents))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)

    print(f"\n📊 Test Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {total_tests - failures - errors}")
    print(f"  Failed: {failures}")
    print(f"  Errors: {errors}")

    success = failures == 0 and errors == 0
    print(f"  Result: {'✅ PASSED' if success else '❌ FAILED'}")

    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
