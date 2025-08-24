"""
ML Pattern Generation System Integration

Tüm ML pattern generation bileşenlerini entegre eden ana sistem sınıfı.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

from .models import (
    PatternSuggestion,
    ValidationResult,
    ConflictReport,
    ReviewDecision,
    ValidationStatus,
    MLPatternConfig,
    PatternTemplate,
)
from .ml_suggester import MLPatternSuggester
from .validator import PatternValidator
from .conflict_detector import ConflictDetector
from .review_interface import PatternReviewInterface


class MLPatternGenerationSystem:
    """
    ML tabanlı pattern generation sistemi ana sınıfı

    Bu sınıf tüm bileşenleri entegre eder:
    1. ML Pattern Suggestion
    2. Pattern Validation
    3. Conflict Detection
    4. Human Review Interface
    5. Adaptive Learning Integration
    """

    def __init__(self, config: Optional[MLPatternConfig] = None, output_dir: str = "ml_pattern_outputs"):
        self.config = config or MLPatternConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.ml_suggester = MLPatternSuggester(self.config)
        self.validator = PatternValidator(self.config)
        self.conflict_detector = ConflictDetector(self.config)
        self.review_interface = PatternReviewInterface()

        # System state
        self.generated_patterns: List[PatternSuggestion] = []
        self.validation_results: List[ValidationResult] = []
        self.conflict_reports: List[ConflictReport] = []
        self.review_decisions: List[ReviewDecision] = []

        # Performance metrics
        self.system_metrics = {
            "total_addresses_processed": 0,
            "patterns_generated": 0,
            "patterns_approved": 0,
            "patterns_rejected": 0,
            "avg_generation_time": 0.0,
            "avg_validation_time": 0.0,
        }

    def generate_patterns_from_addresses(
        self,
        address_samples: List[str],
        pattern_types: Optional[List[str]] = None,
        min_cluster_size: int = 5,
        max_patterns: int = 50,
    ) -> List[PatternSuggestion]:
        """
        Adres örneklerinden pattern'ler üret

        Args:
            address_samples: Adres örnekleri
            pattern_types: Pattern türleri
            min_cluster_size: Minimum cluster boyutu
            max_patterns: Maksimum pattern sayısı

        Returns:
            List[PatternSuggestion]: Üretilen pattern'ler
        """
        start_time = datetime.now()
        self.logger.info(f"Pattern generation started with {len(address_samples)} addresses")

        try:
            # ML suggester ile pattern'leri üret
            suggestions = self.ml_suggester.generate_patterns(addresses=address_samples)

            self.generated_patterns.extend(suggestions)

            # Metrics güncelle
            generation_time = (datetime.now() - start_time).total_seconds()
            self._update_generation_metrics(len(address_samples), len(suggestions), generation_time)

            self.logger.info(f"Generated {len(suggestions)} patterns in {generation_time:.2f}s")

            return suggestions

        except Exception as e:
            self.logger.error(f"Pattern generation failed: {e}")
            raise

    def validate_patterns(
        self, patterns: Optional[List[PatternSuggestion]] = None, existing_patterns: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Pattern'leri validate et

        Args:
            patterns: Validate edilecek pattern'ler
            existing_patterns: Mevcut pattern'ler

        Returns:
            List[ValidationResult]: Validation sonuçları
        """
        patterns = patterns or self.generated_patterns
        start_time = datetime.now()

        self.logger.info(f"Validating {len(patterns)} patterns")

        try:
            validation_results = []

            for pattern in patterns:
                result = self.validator.validate_pattern(suggestion=pattern, test_addresses=None)
                validation_results.append(result)

            self.validation_results.extend(validation_results)

            # Metrics güncelle
            validation_time = (datetime.now() - start_time).total_seconds()
            self._update_validation_metrics(validation_time)

            # Summary
            valid_count = sum(1 for r in validation_results if r.is_valid)
            self.logger.info(f"Validation complete: {valid_count}/{len(patterns)} valid")

            return validation_results

        except Exception as e:
            self.logger.error(f"Pattern validation failed: {e}")
            raise

    def detect_conflicts(
        self, patterns: Optional[List[PatternSuggestion]] = None, existing_patterns: Optional[List[str]] = None
    ) -> List[ConflictReport]:
        """
        Pattern conflict'lerini tespit et

        Args:
            patterns: Conflict detection için pattern'ler
            existing_patterns: Mevcut pattern'ler

        Returns:
            List[ConflictReport]: Conflict raporları
        """
        patterns = patterns or self.generated_patterns

        self.logger.info(f"Detecting conflicts for {len(patterns)} patterns")

        try:
            # Conflict detection (sadece yeni pattern'ler için)
            conflict_reports = self.conflict_detector.detect_conflicts(patterns)
            self.conflict_reports.extend(conflict_reports)

            # Summary
            critical_conflicts = sum(1 for c in conflict_reports if c.severity.name == "CRITICAL")

            self.logger.info(
                f"Conflict detection complete: {len(conflict_reports)} conflicts, " f"{critical_conflicts} critical"
            )

            return conflict_reports

        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}")
            raise

    def review_patterns(
        self,
        patterns: Optional[List[PatternSuggestion]] = None,
        validation_results: Optional[List[ValidationResult]] = None,
        conflict_reports: Optional[List[ConflictReport]] = None,
        reviewer_name: str = "human_reviewer",
        batch_mode: bool = False,
    ) -> List[ReviewDecision]:
        """
        Pattern'leri human review'a gönder

        Args:
            patterns: Review edilecek pattern'ler
            validation_results: Validation sonuçları
            conflict_reports: Conflict raporları
            reviewer_name: Reviewer adı
            batch_mode: Batch mode aktif mi

        Returns:
            List[ReviewDecision]: Review kararları
        """
        patterns = patterns or self.generated_patterns
        validation_results = validation_results or self.validation_results
        conflict_reports = conflict_reports or self.conflict_reports

        self.logger.info(f"Starting review session for {len(patterns)} patterns")

        try:
            decisions = self.review_interface.start_review_session(
                suggestions=patterns,
                validation_results=validation_results,
                conflict_reports=conflict_reports,
                reviewer_name=reviewer_name,
                batch_mode=batch_mode,
            )

            self.review_decisions.extend(decisions)

            # Metrics güncelle
            self._update_review_metrics(decisions)

            return decisions

        except Exception as e:
            self.logger.error(f"Pattern review failed: {e}")
            raise

    def get_approved_patterns(self) -> List[PatternSuggestion]:
        """
        Onaylanmış pattern'leri getir

        Returns:
            List[PatternSuggestion]: Onaylanmış pattern'ler
        """
        approved_ids = {
            decision.pattern_id for decision in self.review_decisions if decision.decision == ValidationStatus.APPROVED
        }

        approved_patterns = [pattern for pattern in self.generated_patterns if pattern.pattern_id in approved_ids]

        return approved_patterns

    def get_patterns_needing_modification(self) -> List[Tuple[PatternSuggestion, ReviewDecision]]:
        """
        Modifikasyon gereken pattern'leri getir

        Returns:
            List[Tuple[PatternSuggestion, ReviewDecision]]: Pattern ve review decision
        """
        modification_decisions = {
            decision.pattern_id: decision
            for decision in self.review_decisions
            if decision.decision == ValidationStatus.NEEDS_MODIFICATION
        }

        results = []
        for pattern in self.generated_patterns:
            if pattern.pattern_id in modification_decisions:
                results.append((pattern, modification_decisions[pattern.pattern_id]))

        return results

    def apply_pattern_modifications(
        self, modifications: List[Tuple[PatternSuggestion, ReviewDecision]]
    ) -> List[PatternSuggestion]:
        """
        Pattern modifikasyonlarını uygula

        Args:
            modifications: Modifikasyon listesi

        Returns:
            List[PatternSuggestion]: Modifiye edilmiş pattern'ler
        """
        modified_patterns = []

        for pattern, decision in modifications:
            if not decision.modifications:
                continue

            # Yeni pattern oluştur
            modified_pattern = PatternSuggestion(
                pattern_id=f"{pattern.pattern_id}_modified",
                pattern_type=pattern.pattern_type,
                regex_pattern=decision.modifications.get("regex_pattern", pattern.regex_pattern),
                template=PatternTemplate(
                    template=decision.modifications.get("template", pattern.template.template),
                    components=pattern.template.components,
                    complexity_score=pattern.template.complexity_score,
                    generalizability=pattern.template.generalizability,
                ),
                confidence=pattern.confidence,
                coverage=pattern.coverage,
                source_cluster=pattern.source_cluster,
                examples=pattern.examples,
                quality_score=pattern.quality_score,
            )

            modified_patterns.append(modified_pattern)

        self.logger.info(f"Applied modifications to {len(modified_patterns)} patterns")

        return modified_patterns

    def export_system_report(self, output_file: Optional[str] = None) -> str:
        """
        Sistem raporunu export et

        Args:
            output_file: Çıktı dosyası

        Returns:
            str: Rapor dosya yolu
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"ml_pattern_system_report_{timestamp}.json"

        # System state özeti
        approved_patterns = self.get_approved_patterns()
        patterns_needing_modification = self.get_patterns_needing_modification()

        report_data = {
            "system_report": {
                "generated_at": datetime.now().isoformat(),
                "system_metrics": self.system_metrics,
                "summary": {
                    "total_patterns_generated": len(self.generated_patterns),
                    "total_validated": len(self.validation_results),
                    "total_conflicts_detected": len(self.conflict_reports),
                    "total_reviewed": len(self.review_decisions),
                    "approved_patterns": len(approved_patterns),
                    "patterns_needing_modification": len(patterns_needing_modification),
                    "rejected_patterns": sum(1 for d in self.review_decisions if d.decision == ValidationStatus.REJECTED),
                },
                "patterns": [pattern.dict() for pattern in self.generated_patterns],
                "validation_results": [result.dict() for result in self.validation_results],
                "conflict_reports": [report.dict() for report in self.conflict_reports],
                "review_decisions": [decision.dict() for decision in self.review_decisions],
            }
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"System report exported: {output_file}")

        return str(output_file)

    def full_pipeline(
        self,
        address_samples: List[str],
        existing_patterns: Optional[List[str]] = None,
        reviewer_name: str = "human_reviewer",
        batch_review: bool = False,
        auto_export: bool = True,
    ) -> Dict[str, Any]:
        """
        Full ML pattern generation pipeline'ı çalıştır

        Args:
            address_samples: Adres örnekleri
            existing_patterns: Mevcut pattern'ler
            reviewer_name: Reviewer adı
            batch_review: Batch review modu
            auto_export: Otomatik rapor export

        Returns:
            Dict[str, Any]: Pipeline sonuçları
        """
        pipeline_start = datetime.now()
        self.logger.info("Starting full ML pattern generation pipeline")

        try:
            # 1. Pattern Generation
            self.logger.info("Step 1: Generating patterns...")
            patterns = self.generate_patterns_from_addresses(address_samples)

            # 2. Pattern Validation
            self.logger.info("Step 2: Validating patterns...")
            validation_results = self.validate_patterns(patterns, existing_patterns)

            # 3. Conflict Detection
            self.logger.info("Step 3: Detecting conflicts...")
            conflict_reports = self.detect_conflicts(patterns, existing_patterns)

            # 4. Human Review
            self.logger.info("Step 4: Human review...")
            review_decisions = self.review_patterns(
                patterns, validation_results, conflict_reports, reviewer_name, batch_review
            )

            # 5. Get final results
            approved_patterns = self.get_approved_patterns()
            patterns_needing_modification = self.get_patterns_needing_modification()

            # 6. Auto export
            report_file = None
            if auto_export:
                self.logger.info("Step 5: Exporting system report...")
                report_file = self.export_system_report()

            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()

            # Pipeline results
            results = {
                "pipeline_duration": pipeline_duration,
                "total_patterns_generated": len(patterns),
                "approved_patterns": len(approved_patterns),
                "patterns_needing_modification": len(patterns_needing_modification),
                "rejected_patterns": sum(1 for d in review_decisions if d.decision == ValidationStatus.REJECTED),
                "total_conflicts": len(conflict_reports),
                "critical_conflicts": sum(1 for c in conflict_reports if c.severity.name == "CRITICAL"),
                "system_metrics": self.system_metrics,
                "report_file": report_file,
            }

            self.logger.info(f"Pipeline completed in {pipeline_duration:.2f}s")
            self.logger.info(
                f"Results: {len(approved_patterns)} approved, " f"{len(patterns_needing_modification)} need modification"
            )

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def _update_generation_metrics(self, addresses_count: int, patterns_count: int, duration: float):
        """Generation metrics güncelle"""
        self.system_metrics["total_addresses_processed"] += addresses_count
        self.system_metrics["patterns_generated"] += patterns_count

        # Moving average for generation time
        if self.system_metrics["avg_generation_time"] == 0:
            self.system_metrics["avg_generation_time"] = duration
        else:
            self.system_metrics["avg_generation_time"] = self.system_metrics["avg_generation_time"] * 0.7 + duration * 0.3

    def _update_validation_metrics(self, duration: float):
        """Validation metrics güncelle"""
        if self.system_metrics["avg_validation_time"] == 0:
            self.system_metrics["avg_validation_time"] = duration
        else:
            self.system_metrics["avg_validation_time"] = self.system_metrics["avg_validation_time"] * 0.7 + duration * 0.3

    def _update_review_metrics(self, decisions: List[ReviewDecision]):
        """Review metrics güncelle"""
        approved = sum(1 for d in decisions if d.decision == ValidationStatus.APPROVED)
        rejected = sum(1 for d in decisions if d.decision == ValidationStatus.REJECTED)

        self.system_metrics["patterns_approved"] += approved
        self.system_metrics["patterns_rejected"] += rejected

    def reset_system_state(self):
        """Sistem state'ini reset et"""
        self.generated_patterns = []
        self.validation_results = []
        self.conflict_reports = []
        self.review_decisions = []

        self.logger.info("System state reset")


def create_ml_pattern_system(
    config: Optional[MLPatternConfig] = None, output_dir: str = "ml_pattern_outputs"
) -> MLPatternGenerationSystem:
    """
    ML Pattern Generation System oluştur

    Args:
        config: Sistem konfigürasyonu
        output_dir: Çıktı dizini

    Returns:
        MLPatternGenerationSystem: Sistem instance
    """
    return MLPatternGenerationSystem(config, output_dir)
