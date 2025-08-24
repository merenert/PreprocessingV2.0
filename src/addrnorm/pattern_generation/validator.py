"""
Pattern Validator

Pattern'lerin kalitesini, genelleştirilebilirliğini ve çakışma riskini değerlendiren sınıf.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import unicodedata

from .models import PatternSuggestion, ValidationResult, PatternType, MLPatternConfig, AddressComponent


class PatternQualityMetrics:
    """
    Pattern kalite metrikleri hesaplama sınıfı
    """

    @staticmethod
    def calculate_complexity_score(pattern: str) -> float:
        """
        Pattern karmaşıklık skorunu hesapla

        Args:
            pattern: Regex pattern

        Returns:
            float: Karmaşıklık skoru (0.0-1.0)
        """
        if not pattern:
            return 1.0

        complexity_factors = {
            "length": len(pattern) / 200,  # Uzunluk faktörü
            "special_chars": len(re.findall(r"[+*?{}[\]()\\|.]", pattern)) / 20,
            "groups": pattern.count("(") / 10,
            "alternatives": pattern.count("|") / 5,
            "quantifiers": len(re.findall(r"[+*?]", pattern)) / 10,
            "lookarounds": len(re.findall(r"\?[=!<]", pattern)) / 5,
        }

        # Weighted complexity score
        weights = {
            "length": 0.2,
            "special_chars": 0.3,
            "groups": 0.2,
            "alternatives": 0.1,
            "quantifiers": 0.15,
            "lookarounds": 0.05,
        }

        complexity = sum(min(complexity_factors[factor], 1.0) * weights[factor] for factor in complexity_factors)

        return min(complexity, 1.0)

    @staticmethod
    def calculate_generalizability_score(pattern: str, examples: List[str], test_addresses: List[str] = None) -> float:
        """
        Pattern genelleştirilebilirlik skorunu hesapla

        Args:
            pattern: Regex pattern
            examples: Pattern'in çıkarıldığı örnekler
            test_addresses: Test için kullanılacak adresler

        Returns:
            float: Genelleştirilebilirlik skoru (0.0-1.0)
        """
        if not pattern or not examples:
            return 0.0

        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return 0.0

        # Training accuracy
        training_matches = sum(1 for example in examples if compiled_pattern.search(example))
        training_accuracy = training_matches / len(examples) if examples else 0

        # Diversity of matches
        diversity_score = PatternQualityMetrics._calculate_diversity(examples)

        # Test accuracy (if test set available)
        test_accuracy = 1.0
        if test_addresses:
            test_matches = sum(1 for addr in test_addresses if compiled_pattern.search(addr))
            test_accuracy = test_matches / len(test_addresses)

        # Component coverage analysis
        component_coverage = PatternQualityMetrics._analyze_component_coverage(pattern, examples)

        # Weighted generalizability
        generalizability = training_accuracy * 0.3 + test_accuracy * 0.3 + diversity_score * 0.2 + component_coverage * 0.2

        return min(generalizability, 1.0)

    @staticmethod
    def _calculate_diversity(examples: List[str]) -> float:
        """Örnek çeşitliliğini hesapla"""
        if not examples:
            return 0.0

        # Word diversity
        all_words = []
        for example in examples:
            words = re.findall(r"\w+", example.lower())
            all_words.extend(words)

        unique_words = len(set(all_words))
        total_words = len(all_words)

        word_diversity = unique_words / total_words if total_words > 0 else 0

        # Length diversity
        lengths = [len(example) for example in examples]
        length_std = np.std(lengths) if len(lengths) > 1 else 0
        length_diversity = min(length_std / 50, 1.0)  # Normalize to 0-1

        # Structure diversity
        structures = set()
        for example in examples:
            # Basit yapı analizi
            structure = re.sub(r"\d+", "NUM", example)
            structure = re.sub(r"\w+", "WORD", structure)
            structures.add(structure)

        structure_diversity = len(structures) / len(examples)

        return (word_diversity + length_diversity + structure_diversity) / 3

    @staticmethod
    def _analyze_component_coverage(pattern: str, examples: List[str]) -> float:
        """Component coverage analizi"""

        # Türkçe adres component'leri
        components = ["il", "ilce", "mahalle", "sokak", "cadde", "bulvar", "no", "daire"]

        component_coverage = 0

        for component in components:
            # Pattern'de bu component var mı?
            if component in pattern.lower():
                component_coverage += 0.2

            # Örneklerde bu component var mı?
            component_count = sum(1 for example in examples if component in example.lower())

            if component_count > 0:
                component_coverage += 0.1

        return min(component_coverage, 1.0)

    @staticmethod
    def calculate_collision_risk(pattern: str, existing_patterns: List[str], test_addresses: List[str] = None) -> float:
        """
        Pattern çakışma riskini hesapla

        Args:
            pattern: Yeni pattern
            existing_patterns: Mevcut pattern'ler
            test_addresses: Test adresleri

        Returns:
            float: Çakışma riski (0.0-1.0)
        """
        if not pattern or not existing_patterns:
            return 0.0

        try:
            new_compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return 1.0  # Invalid pattern = high risk

        collision_risks = []

        for existing_pattern in existing_patterns:
            try:
                existing_compiled = re.compile(existing_pattern, re.IGNORECASE)

                # Pattern similarity
                similarity = PatternQualityMetrics._calculate_pattern_similarity(pattern, existing_pattern)

                # Overlap in test addresses
                overlap = 0.0
                if test_addresses:
                    overlap = PatternQualityMetrics._calculate_address_overlap(new_compiled, existing_compiled, test_addresses)

                # Combined risk
                risk = similarity * 0.6 + overlap * 0.4
                collision_risks.append(risk)

            except re.error:
                continue

        # Maximum risk
        return max(collision_risks) if collision_risks else 0.0

    @staticmethod
    def _calculate_pattern_similarity(pattern1: str, pattern2: str) -> float:
        """İki pattern arasındaki benzerlik"""

        # Exact match
        if pattern1 == pattern2:
            return 1.0

        # Character-level similarity
        set1 = set(pattern1.lower())
        set2 = set(pattern2.lower())

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        char_similarity = intersection / union if union > 0 else 0

        # Subsequence similarity
        subseq_similarity = PatternQualityMetrics._longest_common_subsequence(pattern1, pattern2) / max(
            len(pattern1), len(pattern2)
        )

        return (char_similarity + subseq_similarity) / 2

    @staticmethod
    def _longest_common_subsequence(s1: str, s2: str) -> int:
        """En uzun ortak alt dizi"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def _calculate_address_overlap(pattern1: re.Pattern, pattern2: re.Pattern, addresses: List[str]) -> float:
        """Address overlap hesapla"""

        matches1 = set()
        matches2 = set()

        for i, address in enumerate(addresses):
            if pattern1.search(address):
                matches1.add(i)
            if pattern2.search(address):
                matches2.add(i)

        if not matches1 and not matches2:
            return 0.0

        intersection = len(matches1.intersection(matches2))
        union = len(matches1.union(matches2))

        return intersection / union if union > 0 else 0.0


class PatternValidator:
    """
    Pattern validation ana sınıfı
    """

    def __init__(self, config: MLPatternConfig, existing_patterns: List[str] = None):
        self.config = config
        self.existing_patterns = existing_patterns or []
        self.logger = logging.getLogger(__name__)

        # Validation cache
        self._validation_cache = {}

    def validate_pattern(self, suggestion: PatternSuggestion, test_addresses: List[str] = None) -> ValidationResult:
        """
        Pattern'i kapsamlı şekilde validate et

        Args:
            suggestion: Pattern önerisi
            test_addresses: Test adresleri

        Returns:
            ValidationResult: Validation sonucu
        """
        pattern_id = suggestion.pattern_id

        # Cache kontrolü
        cache_key = f"{pattern_id}_{hash(str(test_addresses))}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        self.logger.info(f"Validating pattern {pattern_id}")

        # 1. Complexity analysis
        complexity_analysis = self._analyze_complexity(suggestion)

        # 2. Generalizability analysis
        generalizability_score = self._analyze_generalizability(suggestion, test_addresses)

        # 3. Collision risk analysis
        collision_risk = self._analyze_collision_risk(suggestion, test_addresses)

        # 4. Coverage analysis
        coverage_analysis = self._analyze_coverage(suggestion, test_addresses)

        # 5. Quality score calculation
        quality_score = self._calculate_overall_quality(
            suggestion, complexity_analysis, generalizability_score, collision_risk, coverage_analysis
        )

        # 6. Issue detection
        issues = self._detect_issues(
            suggestion, complexity_analysis, generalizability_score, collision_risk, coverage_analysis
        )

        # 7. Recommendations
        recommendations = self._generate_recommendations(suggestion, issues, complexity_analysis)

        # 8. Final validation decision
        is_valid = self._make_validation_decision(quality_score, collision_risk, generalizability_score, issues)

        result = ValidationResult(
            pattern_id=pattern_id,
            is_valid=is_valid,
            quality_score=quality_score,
            complexity_analysis=complexity_analysis,
            generalizability_score=generalizability_score,
            collision_risk=collision_risk,
            coverage_analysis=coverage_analysis,
            issues=issues,
            recommendations=recommendations,
        )

        # Cache'le
        self._validation_cache[cache_key] = result

        return result

    def validate_batch(self, suggestions: List[PatternSuggestion], test_addresses: List[str] = None) -> List[ValidationResult]:
        """
        Batch pattern validation

        Args:
            suggestions: Pattern önerileri
            test_addresses: Test adresleri

        Returns:
            List[ValidationResult]: Validation sonuçları
        """
        results = []

        for suggestion in suggestions:
            try:
                result = self.validate_pattern(suggestion, test_addresses)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error validating pattern {suggestion.pattern_id}: {e}")

                # Error durumunda default result
                error_result = ValidationResult(
                    pattern_id=suggestion.pattern_id,
                    is_valid=False,
                    quality_score=0.0,
                    generalizability_score=0.0,
                    collision_risk=1.0,
                    issues=[f"Validation error: {str(e)}"],
                    recommendations=["Pattern needs manual review"],
                )
                results.append(error_result)

        return results

    def _analyze_complexity(self, suggestion: PatternSuggestion) -> Dict[str, float]:
        """Complexity analizi"""

        pattern = suggestion.regex_pattern

        complexity_metrics = {
            "overall_complexity": PatternQualityMetrics.calculate_complexity_score(pattern),
            "pattern_length": len(pattern) / 200,
            "special_char_ratio": len(re.findall(r"[+*?{}[\]()\\|.]", pattern)) / len(pattern),
            "group_count": pattern.count("(") / 10,
            "alternative_count": pattern.count("|") / 5,
            "readability_score": self._calculate_readability(pattern),
        }

        # Normalize scores
        for key in complexity_metrics:
            complexity_metrics[key] = min(complexity_metrics[key], 1.0)

        return complexity_metrics

    def _calculate_readability(self, pattern: str) -> float:
        """Pattern okunabilirlik skoru"""

        readability_factors = {
            "descriptive_groups": len(re.findall(r"\(\?P<\w+>", pattern)) / 5,
            "whitespace_usage": pattern.count(r"\s") / len(pattern) * 10,
            "escape_complexity": pattern.count("\\") / len(pattern) * 5,
            "nested_complexity": self._count_nested_groups(pattern) / 3,
        }

        # Lower is more readable
        readability = 1.0 - sum(min(factor, 1.0) for factor in readability_factors.values()) / len(readability_factors)

        return max(readability, 0.0)

    def _count_nested_groups(self, pattern: str) -> int:
        """Nested group sayısını say"""
        max_depth = 0
        current_depth = 0

        for char in pattern:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _analyze_generalizability(self, suggestion: PatternSuggestion, test_addresses: List[str]) -> float:
        """Generalizability analizi"""

        return PatternQualityMetrics.calculate_generalizability_score(
            suggestion.regex_pattern, suggestion.examples, test_addresses
        )

    def _analyze_collision_risk(self, suggestion: PatternSuggestion, test_addresses: List[str]) -> float:
        """Collision risk analizi"""

        return PatternQualityMetrics.calculate_collision_risk(suggestion.regex_pattern, self.existing_patterns, test_addresses)

    def _analyze_coverage(self, suggestion: PatternSuggestion, test_addresses: List[str]) -> Dict[str, Any]:
        """Coverage analizi"""

        coverage_analysis = {
            "training_coverage": suggestion.coverage,
            "test_coverage": 0.0,
            "component_coverage": {},
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        }

        if test_addresses:
            try:
                compiled_pattern = re.compile(suggestion.regex_pattern, re.IGNORECASE)

                matches = sum(1 for addr in test_addresses if compiled_pattern.search(addr))

                coverage_analysis["test_coverage"] = matches / len(test_addresses)

                # Component coverage analizi
                components = ["il", "ilce", "mahalle", "sokak", "no"]
                for component in components:
                    component_matches = sum(
                        1 for addr in test_addresses if component in addr.lower() and compiled_pattern.search(addr)
                    )

                    component_total = sum(1 for addr in test_addresses if component in addr.lower())

                    if component_total > 0:
                        coverage_analysis["component_coverage"][component] = component_matches / component_total

            except re.error:
                coverage_analysis["test_coverage"] = 0.0

        return coverage_analysis

    def _calculate_overall_quality(
        self,
        suggestion: PatternSuggestion,
        complexity_analysis: Dict[str, float],
        generalizability_score: float,
        collision_risk: float,
        coverage_analysis: Dict[str, Any],
    ) -> float:
        """Genel kalite skoru hesapla"""

        # Quality factors
        factors = {
            "pattern_confidence": suggestion.confidence,
            "generalizability": generalizability_score,
            "complexity_penalty": 1.0 - complexity_analysis["overall_complexity"],
            "collision_penalty": 1.0 - collision_risk,
            "coverage_quality": coverage_analysis.get("test_coverage", suggestion.coverage),
        }

        # Weights
        weights = {
            "pattern_confidence": 0.25,
            "generalizability": 0.25,
            "complexity_penalty": 0.2,
            "collision_penalty": 0.15,
            "coverage_quality": 0.15,
        }

        quality_score = sum(factors[factor] * weights[factor] for factor in factors)

        return min(quality_score, 1.0)

    def _detect_issues(
        self,
        suggestion: PatternSuggestion,
        complexity_analysis: Dict[str, float],
        generalizability_score: float,
        collision_risk: float,
        coverage_analysis: Dict[str, Any],
    ) -> List[str]:
        """Pattern sorunlarını tespit et"""

        issues = []

        # Complexity issues
        if complexity_analysis["overall_complexity"] > self.config.max_pattern_complexity:
            issues.append(f"Pattern çok karmaşık (complexity: {complexity_analysis['overall_complexity']:.2f})")

        if complexity_analysis["readability_score"] < 0.3:
            issues.append("Pattern okunabilirliği düşük")

        # Generalizability issues
        if generalizability_score < self.config.min_generalizability:
            issues.append(f"Düşük genelleştirilebilirlik (score: {generalizability_score:.2f})")

        # Collision issues
        if collision_risk > self.config.max_collision_risk:
            issues.append(f"Yüksek çakışma riski (risk: {collision_risk:.2f})")

        # Coverage issues
        if suggestion.coverage < self.config.min_coverage:
            issues.append(f"Düşük coverage (coverage: {suggestion.coverage:.2f})")

        test_coverage = coverage_analysis.get("test_coverage", 0)
        if test_coverage < suggestion.coverage * 0.8:
            issues.append("Test coverage training coverage'dan çok düşük")

        # Regex validity
        try:
            re.compile(suggestion.regex_pattern)
        except re.error as e:
            issues.append(f"Invalid regex pattern: {str(e)}")

        return issues

    def _generate_recommendations(
        self, suggestion: PatternSuggestion, issues: List[str], complexity_analysis: Dict[str, float]
    ) -> List[str]:
        """İyileştirme önerileri oluştur"""

        recommendations = []

        # Complexity-based recommendations
        if complexity_analysis["overall_complexity"] > 0.8:
            recommendations.append("Pattern'i basitleştirin")

            if complexity_analysis["group_count"] > 0.5:
                recommendations.append("Gereksiz grupları kaldırın")

            if complexity_analysis["alternative_count"] > 0.3:
                recommendations.append("Alternatifleri azaltın veya ayrı pattern'lere bölün")

        # Coverage-based recommendations
        if suggestion.coverage < 0.8:
            recommendations.append("Pattern'in coverage'ını artırmak için daha fazla örnek toplayın")

        # Generalizability recommendations
        if len(suggestion.examples) < 10:
            recommendations.append("Daha fazla örnek adres ekleyin")

        # Issue-specific recommendations
        for issue in issues:
            if "çakışma" in issue.lower():
                recommendations.append("Mevcut pattern'lerle çakışmayı önlemek için daha spesifik hale getirin")
            elif "karmaşık" in issue.lower():
                recommendations.append("Pattern'i daha basit alt-pattern'lere bölün")
            elif "coverage" in issue.lower():
                recommendations.append("Pattern'in kapsam alanını genişletin")

        # Default recommendation
        if not recommendations:
            recommendations.append("Pattern kaliteli görünüyor, human review ile onaylayın")

        return recommendations

    def _make_validation_decision(
        self, quality_score: float, collision_risk: float, generalizability_score: float, issues: List[str]
    ) -> bool:
        """Final validation kararı"""

        # Critical issues check
        critical_issues = [
            issue for issue in issues if any(keyword in issue.lower() for keyword in ["invalid", "error", "critical"])
        ]

        if critical_issues:
            return False

        # Quality thresholds
        if quality_score < 0.5:
            return False

        if collision_risk > self.config.max_collision_risk:
            return False

        if generalizability_score < self.config.min_generalizability:
            return False

        # Issue count threshold
        if len(issues) > 3:
            return False

        return True
