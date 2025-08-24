"""
Pattern Analyzer - Advanced analysis for address patterns

Analyzes address patterns for quality, coverage, and performance metrics.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .models import PatternSuggestion, PatternPerformance, PatternQuality, PatternGenerationConfig, AddressCluster

logger = logging.getLogger(__name__)


@dataclass
class PatternAnalysisResult:
    """Pattern analysis comprehensive result"""

    pattern_id: str
    quality_score: float
    quality_level: PatternQuality
    coverage_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    complexity_metrics: Dict[str, float]
    recommendations: List[str]
    analysis_timestamp: datetime


class PatternAnalyzer:
    """
    Advanced pattern analysis and quality assessment
    """

    def __init__(self, config: Optional[PatternGenerationConfig] = None):
        self.config = config or PatternGenerationConfig()
        self.analysis_cache = {}
        self.performance_history = defaultdict(list)

    def analyze_pattern_quality(
        self, pattern: str, test_addresses: List[str], expected_extractions: Optional[List[Dict[str, str]]] = None
    ) -> PatternAnalysisResult:
        """
        Comprehensive pattern quality analysis

        Args:
            pattern: Regex pattern to analyze
            test_addresses: Test address list
            expected_extractions: Expected extraction results for validation

        Returns:
            PatternAnalysisResult: Comprehensive analysis result
        """

        pattern_id = self._generate_pattern_id(pattern)

        # Test pattern compilation
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Pattern compilation failed: {e}")
            return self._create_failed_analysis(pattern_id, f"Pattern compilation error: {e}")

        # Coverage analysis
        coverage_metrics = self._analyze_coverage(compiled_pattern, test_addresses)

        # Performance analysis
        performance_metrics = self._analyze_performance(compiled_pattern, test_addresses, expected_extractions)

        # Complexity analysis
        complexity_metrics = self._analyze_complexity(pattern)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(coverage_metrics, performance_metrics, complexity_metrics)
        quality_level = self._determine_quality_level(quality_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(coverage_metrics, performance_metrics, complexity_metrics)

        result = PatternAnalysisResult(
            pattern_id=pattern_id,
            quality_score=quality_score,
            quality_level=quality_level,
            coverage_metrics=coverage_metrics,
            performance_metrics=performance_metrics,
            complexity_metrics=complexity_metrics,
            recommendations=recommendations,
            analysis_timestamp=datetime.now(),
        )

        # Cache result
        self.analysis_cache[pattern_id] = result

        return result

    def _analyze_coverage(self, pattern: re.Pattern, addresses: List[str]) -> Dict[str, float]:
        """Analyze pattern coverage metrics"""

        total_addresses = len(addresses)
        if total_addresses == 0:
            return {"match_rate": 0.0, "field_coverage": 0.0, "completeness": 0.0}

        matches = 0
        total_groups = 0
        successful_groups = 0
        field_counts = defaultdict(int)

        for address in addresses:
            match = pattern.search(address)
            if match:
                matches += 1
                groups = match.groups()
                total_groups += len(groups)

                # Count non-empty groups
                for i, group in enumerate(groups):
                    if group and group.strip():
                        successful_groups += 1
                        field_counts[f"field_{i}"] += 1

        match_rate = matches / total_addresses
        field_coverage = successful_groups / total_groups if total_groups > 0 else 0.0

        # Calculate completeness (how well fields are populated)
        avg_field_population = np.mean(list(field_counts.values())) / matches if matches > 0 else 0.0
        completeness = min(1.0, avg_field_population)

        return {
            "match_rate": round(match_rate, 3),
            "field_coverage": round(field_coverage, 3),
            "completeness": round(completeness, 3),
            "total_matches": matches,
            "avg_groups_per_match": round(total_groups / matches, 2) if matches > 0 else 0.0,
        }

    def _analyze_performance(
        self, pattern: re.Pattern, addresses: List[str], expected_extractions: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, float]:
        """Analyze pattern performance metrics"""

        import time

        # Performance timing
        start_time = time.time()

        matches = []
        for address in addresses:
            match = pattern.search(address)
            matches.append(match)

        execution_time = time.time() - start_time
        avg_time_per_address = execution_time / len(addresses) if addresses else 0.0

        # Accuracy metrics (if expected extractions provided)
        accuracy_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        if expected_extractions and len(expected_extractions) == len(addresses):
            correct_extractions = 0
            total_predictions = 0
            total_expected = 0

            for i, (address, expected) in enumerate(zip(addresses, expected_extractions)):
                match = matches[i]
                if match:
                    extracted = match.groupdict() if match.groupdict() else {}
                    total_predictions += len(extracted)

                    # Check if extraction matches expected
                    if extracted == expected:
                        correct_extractions += 1

                total_expected += len(expected)

            accuracy = correct_extractions / len(addresses) if addresses else 0.0
            precision = correct_extractions / total_predictions if total_predictions > 0 else 0.0
            recall = correct_extractions / total_expected if total_expected > 0 else 0.0

            accuracy_metrics = {"accuracy": round(accuracy, 3), "precision": round(precision, 3), "recall": round(recall, 3)}

        return {
            "execution_time_ms": round(execution_time * 1000, 2),
            "avg_time_per_address_ms": round(avg_time_per_address * 1000, 3),
            "throughput_addresses_per_sec": round(len(addresses) / execution_time, 1) if execution_time > 0 else 0.0,
            **accuracy_metrics,
        }

    def _analyze_complexity(self, pattern: str) -> Dict[str, float]:
        """Analyze pattern complexity metrics"""

        # Basic complexity metrics
        pattern_length = len(pattern)

        # Count special regex characters
        special_chars = set(r".*+?^${}[]|()\/")
        special_char_count = sum(1 for char in pattern if char in special_chars)

        # Count groups
        group_count = pattern.count("(")
        named_group_count = pattern.count("(?P<")

        # Count quantifiers
        quantifier_patterns = [r"\*", r"\+", r"\?", r"\{.*?\}"]
        quantifier_count = sum(len(re.findall(q, pattern)) for q in quantifier_patterns)

        # Count character classes
        char_class_count = pattern.count("[")

        # Calculate complexity score (0-1, lower is simpler)
        complexity_factors = [
            pattern_length / 100,  # Normalize by typical pattern length
            special_char_count / 20,
            group_count / 10,
            quantifier_count / 10,
            char_class_count / 5,
        ]

        complexity_score = min(1.0, sum(complexity_factors))

        # Readability score (inverse of complexity)
        readability_score = 1.0 - complexity_score

        return {
            "pattern_length": pattern_length,
            "special_char_count": special_char_count,
            "group_count": group_count,
            "named_group_count": named_group_count,
            "quantifier_count": quantifier_count,
            "char_class_count": char_class_count,
            "complexity_score": round(complexity_score, 3),
            "readability_score": round(readability_score, 3),
        }

    def _calculate_quality_score(
        self, coverage: Dict[str, float], performance: Dict[str, float], complexity: Dict[str, float]
    ) -> float:
        """Calculate overall quality score"""

        weights = self.config.quality_weights

        # Coverage component (30%)
        coverage_score = coverage["match_rate"] * 0.4 + coverage["field_coverage"] * 0.3 + coverage["completeness"] * 0.3

        # Performance component (20%)
        performance_score = 1.0  # Base score
        if performance["avg_time_per_address_ms"] > 10:  # Penalty for slow patterns
            performance_score *= 0.8
        if performance.get("accuracy", 0) > 0:
            performance_score = (performance_score + performance["accuracy"]) / 2

        # Complexity component (30% - readability)
        complexity_score = complexity["readability_score"]

        # Specificity component (20% - how specific vs generic the pattern is)
        specificity_score = min(1.0, complexity["group_count"] / 5)  # More groups = more specific

        # Weighted combination
        overall_score = (
            coverage_score * weights["completeness"]
            + performance_score * weights["performance"]
            + complexity_score * weights["readability"]
            + specificity_score * weights["specificity"]
        )

        return round(min(1.0, overall_score), 3)

    def _determine_quality_level(self, score: float) -> PatternQuality:
        """Determine quality level from score"""

        if score >= 0.8:
            return PatternQuality.EXCELLENT
        elif score >= 0.6:
            return PatternQuality.GOOD
        elif score >= 0.4:
            return PatternQuality.FAIR
        else:
            return PatternQuality.POOR

    def _generate_recommendations(
        self, coverage: Dict[str, float], performance: Dict[str, float], complexity: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations"""

        recommendations = []

        # Coverage recommendations
        if coverage["match_rate"] < 0.7:
            recommendations.append("Pattern match rate is low. Consider making the pattern more flexible.")

        if coverage["field_coverage"] < 0.6:
            recommendations.append("Field coverage is low. Review capture groups for better extraction.")

        if coverage["completeness"] < 0.5:
            recommendations.append("Field completeness is low. Make optional fields truly optional with '?'.")

        # Performance recommendations
        if performance["avg_time_per_address_ms"] > 5:
            recommendations.append("Pattern execution is slow. Consider optimizing regex complexity.")

        if performance.get("accuracy", 1) < 0.8:
            recommendations.append("Pattern accuracy is low. Review extraction logic and test cases.")

        # Complexity recommendations
        if complexity["complexity_score"] > 0.7:
            recommendations.append("Pattern is complex. Consider breaking into simpler sub-patterns.")

        if complexity["group_count"] > 8:
            recommendations.append("Too many capture groups. Consider using non-capturing groups (?:...) where appropriate.")

        if complexity["pattern_length"] > 150:
            recommendations.append("Pattern is very long. Consider splitting into multiple patterns.")

        # Specificity recommendations
        if complexity["named_group_count"] == 0 and complexity["group_count"] > 0:
            recommendations.append("Use named capture groups (?P<name>...) for better maintainability.")

        if not recommendations:
            recommendations.append("Pattern quality is good. No major improvements needed.")

        return recommendations

    def _generate_pattern_id(self, pattern: str) -> str:
        """Generate unique pattern ID"""
        import hashlib

        return f"pattern_{hashlib.md5(pattern.encode()).hexdigest()[:8]}"

    def _create_failed_analysis(self, pattern_id: str, error: str) -> PatternAnalysisResult:
        """Create failed analysis result"""
        return PatternAnalysisResult(
            pattern_id=pattern_id,
            quality_score=0.0,
            quality_level=PatternQuality.POOR,
            coverage_metrics={"error": error},
            performance_metrics={"error": error},
            complexity_metrics={"error": error},
            recommendations=[f"Fix pattern error: {error}"],
            analysis_timestamp=datetime.now(),
        )

    def compare_patterns(self, patterns: List[str], test_addresses: List[str]) -> Dict[str, Any]:
        """Compare multiple patterns on same test data"""

        results = {}

        for i, pattern in enumerate(patterns):
            pattern_id = f"pattern_{i+1}"
            try:
                analysis = self.analyze_pattern_quality(pattern, test_addresses)
                results[pattern_id] = {
                    "pattern": pattern,
                    "analysis": analysis,
                    "quality_score": analysis.quality_score,
                    "quality_level": analysis.quality_level.value,
                }
            except Exception as e:
                logger.error(f"Error analyzing pattern {i+1}: {e}")
                results[pattern_id] = {"pattern": pattern, "error": str(e), "quality_score": 0.0, "quality_level": "poor"}

        # Rank patterns by quality
        ranked = sorted(results.items(), key=lambda x: x[1].get("quality_score", 0), reverse=True)

        return {
            "comparison_results": results,
            "ranked_patterns": [{"rank": i + 1, "pattern_id": pid, **data} for i, (pid, data) in enumerate(ranked)],
            "best_pattern": ranked[0] if ranked else None,
            "analysis_summary": {
                "total_patterns": len(patterns),
                "avg_quality_score": np.mean([r.get("quality_score", 0) for r in results.values()]),
                "best_score": max([r.get("quality_score", 0) for r in results.values()]) if results else 0,
            },
        }

    def get_analysis_history(self, pattern_id: str) -> List[PatternAnalysisResult]:
        """Get analysis history for a pattern"""
        return self.performance_history.get(pattern_id, [])

    def update_performance_metrics(self, pattern_id: str, metrics: PatternPerformance):
        """Update performance tracking for a pattern"""
        self.performance_history[pattern_id].append(metrics)

        # Keep only last 100 entries
        if len(self.performance_history[pattern_id]) > 100:
            self.performance_history[pattern_id] = self.performance_history[pattern_id][-100:]
