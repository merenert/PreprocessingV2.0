"""
Threshold Optimization Engine

Advanced threshold management and optimization system for pattern performance.
Provides data-driven recommendations for threshold adjustments based on
performance metrics, success rates, and processing efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import logging
from collections import defaultdict
import statistics

from .metrics_collector import MetricsCollector, MetricEvent
from .analytics import SystemAnalytics, AlertLevel

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Threshold optimization strategies"""

    PERFORMANCE_FOCUSED = "performance"  # Prioritize speed
    ACCURACY_FOCUSED = "accuracy"  # Prioritize precision
    BALANCED = "balanced"  # Balance speed and accuracy
    ADAPTIVE = "adaptive"  # Dynamic based on usage patterns


class OptimizationMetric(Enum):
    """Metrics to optimize for"""

    SUCCESS_RATE = "success_rate"
    PROCESSING_TIME = "processing_time"
    CONFIDENCE_SCORE = "confidence_score"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


@dataclass
class ThresholdRecommendation:
    """Threshold adjustment recommendation"""

    pattern_id: str
    current_threshold: float
    recommended_threshold: float
    expected_improvement: float
    confidence_level: float
    strategy: OptimizationStrategy
    rationale: str
    supporting_metrics: Dict[str, float]
    risk_assessment: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "current_threshold": round(self.current_threshold, 3),
            "recommended_threshold": round(self.recommended_threshold, 3),
            "expected_improvement": round(self.expected_improvement, 2),
            "confidence_level": round(self.confidence_level, 3),
            "strategy": self.strategy.value,
            "rationale": self.rationale,
            "supporting_metrics": {k: round(v, 3) for k, v in self.supporting_metrics.items()},
            "risk_assessment": self.risk_assessment,
        }


@dataclass
class ThresholdConfig:
    """Pattern threshold configuration"""

    pattern_id: str
    current_threshold: float
    min_threshold: float = 0.3
    max_threshold: float = 0.95
    adjustment_step: float = 0.02
    stability_period_hours: int = 24
    min_samples_required: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "current_threshold": self.current_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "adjustment_step": self.adjustment_step,
            "stability_period_hours": self.stability_period_hours,
            "min_samples_required": self.min_samples_required,
        }


@dataclass
class OptimizationResult:
    """Results of threshold optimization analysis"""

    total_patterns_analyzed: int
    recommendations_generated: int
    expected_overall_improvement: float
    high_impact_recommendations: List[ThresholdRecommendation]
    medium_impact_recommendations: List[ThresholdRecommendation]
    low_impact_recommendations: List[ThresholdRecommendation]
    risk_warnings: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_patterns_analyzed": self.total_patterns_analyzed,
            "recommendations_generated": self.recommendations_generated,
            "expected_overall_improvement": round(self.expected_overall_improvement, 2),
            "high_impact_recommendations": [r.to_dict() for r in self.high_impact_recommendations],
            "medium_impact_recommendations": [r.to_dict() for r in self.medium_impact_recommendations],
            "low_impact_recommendations": [r.to_dict() for r in self.low_impact_recommendations],
            "risk_warnings": self.risk_warnings,
            "timestamp": self.timestamp.isoformat(),
        }


class ThresholdOptimizer:
    """
    Advanced threshold optimization engine

    Features:
    - Performance-driven threshold recommendations
    - Multi-strategy optimization approaches
    - Risk assessment and safety checks
    - Adaptive learning from historical data
    - A/B testing framework for validation
    """

    def __init__(self, metrics_collector: MetricsCollector, analytics: SystemAnalytics):
        self.metrics_collector = metrics_collector
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

        # Default configurations
        self.default_configs = self._load_default_configs()

        # Optimization history
        self.optimization_history: List[OptimizationResult] = []

        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}

    def optimize_thresholds(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        time_window_hours: int = 24,
        min_confidence: float = 0.7,
    ) -> OptimizationResult:
        """
        Generate threshold optimization recommendations

        Args:
            strategy: Optimization strategy to use
            time_window_hours: Time window for analysis
            min_confidence: Minimum confidence level for recommendations

        Returns:
            OptimizationResult with recommendations and analysis
        """
        self.logger.info(f"Starting threshold optimization with strategy: {strategy.value}")

        # Get recent events for analysis
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_events = [e for e in self.metrics_collector._events if e.timestamp >= cutoff_time and e.pattern_id]

        if not recent_events:
            self.logger.warning("No recent events found for optimization")
            return self._create_empty_result()

        # Group events by pattern
        pattern_events = defaultdict(list)
        for event in recent_events:
            pattern_events[event.pattern_id].append(event)

        # Analyze each pattern and generate recommendations
        all_recommendations = []
        risk_warnings = []

        for pattern_id, events in pattern_events.items():
            if len(events) < self.default_configs[pattern_id].min_samples_required:
                risk_warnings.append(f"Insufficient data for {pattern_id} ({len(events)} samples)")
                continue

            recommendations = self._analyze_pattern_threshold(pattern_id, events, strategy, min_confidence)
            all_recommendations.extend(recommendations)

        # Categorize recommendations by impact
        high_impact = [r for r in all_recommendations if r.expected_improvement >= 5.0]
        medium_impact = [r for r in all_recommendations if 2.0 <= r.expected_improvement < 5.0]
        low_impact = [r for r in all_recommendations if r.expected_improvement < 2.0]

        # Calculate overall expected improvement
        overall_improvement = self._calculate_overall_improvement(all_recommendations)

        # Create result
        result = OptimizationResult(
            total_patterns_analyzed=len(pattern_events),
            recommendations_generated=len(all_recommendations),
            expected_overall_improvement=overall_improvement,
            high_impact_recommendations=high_impact,
            medium_impact_recommendations=medium_impact,
            low_impact_recommendations=low_impact,
            risk_warnings=risk_warnings,
            timestamp=datetime.now(),
        )

        # Store in history
        self.optimization_history.append(result)

        self.logger.info(f"Optimization complete: {len(all_recommendations)} recommendations generated")
        return result

    def _analyze_pattern_threshold(
        self, pattern_id: str, events: List[MetricEvent], strategy: OptimizationStrategy, min_confidence: float
    ) -> List[ThresholdRecommendation]:
        """Analyze a specific pattern and generate threshold recommendations"""

        config = self.default_configs.get(pattern_id, ThresholdConfig(pattern_id, 0.7))
        current_threshold = config.current_threshold

        # Calculate current performance metrics
        current_metrics = self._calculate_performance_metrics(events)

        # Test different threshold values
        threshold_candidates = self._generate_threshold_candidates(config)
        best_thresholds = []

        for test_threshold in threshold_candidates:
            # Simulate performance with new threshold
            simulated_metrics = self._simulate_threshold_performance(events, test_threshold, current_threshold)

            # Evaluate improvement based on strategy
            improvement = self._evaluate_improvement(current_metrics, simulated_metrics, strategy)

            if improvement > 0:
                best_thresholds.append((test_threshold, improvement, simulated_metrics))

        # Generate recommendations
        recommendations = []

        for threshold, improvement, metrics in best_thresholds:
            if improvement >= 1.0:  # Minimum 1% improvement
                confidence_level = self._calculate_recommendation_confidence(events, current_metrics, metrics)

                if confidence_level >= min_confidence:
                    recommendation = self._create_recommendation(
                        pattern_id, config, threshold, improvement, confidence_level, strategy, current_metrics, metrics
                    )
                    recommendations.append(recommendation)

        return recommendations

    def _calculate_performance_metrics(self, events: List[MetricEvent]) -> Dict[str, float]:
        """Calculate performance metrics for a set of events"""
        if not events:
            return {}

        # Success rate
        successful_events = sum(1 for e in events if e.success)
        success_rate = successful_events / len(events)

        # Processing time metrics
        processing_times = [e.processing_time_ms for e in events if e.processing_time_ms > 0]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0

        # Confidence metrics
        confidences = [e.confidence for e in events if e.confidence > 0]
        avg_confidence = statistics.mean(confidences) if confidences else 0

        # Error rate
        error_rate = 1 - success_rate

        # Throughput (events per minute)
        time_span = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds()
        throughput = len(events) / max(time_span / 60, 1)  # per minute

        return {
            "success_rate": success_rate,
            "avg_processing_time": avg_processing_time,
            "avg_confidence": avg_confidence,
            "error_rate": error_rate,
            "throughput": throughput,
            "total_events": len(events),
        }

    def _generate_threshold_candidates(self, config: ThresholdConfig) -> List[float]:
        """Generate candidate threshold values for testing"""
        candidates = []

        # Test values around current threshold
        current = config.current_threshold
        step = config.adjustment_step

        # Test lower thresholds (for performance improvement)
        for i in range(1, 6):  # Test 5 steps down
            candidate = max(config.min_threshold, current - (step * i))
            if candidate not in candidates:
                candidates.append(candidate)

        # Test higher thresholds (for accuracy improvement)
        for i in range(1, 6):  # Test 5 steps up
            candidate = min(config.max_threshold, current + (step * i))
            if candidate not in candidates:
                candidates.append(candidate)

        return sorted(candidates)

    def _simulate_threshold_performance(
        self, events: List[MetricEvent], new_threshold: float, current_threshold: float
    ) -> Dict[str, float]:
        """Simulate performance metrics with a new threshold"""

        # Simple simulation model based on threshold effects
        threshold_ratio = new_threshold / current_threshold

        # Get baseline metrics
        baseline_metrics = self._calculate_performance_metrics(events)

        # Simulate effects of threshold change
        simulated_metrics = baseline_metrics.copy()

        if new_threshold < current_threshold:
            # Lower threshold: better performance, potentially lower accuracy
            simulated_metrics["avg_processing_time"] *= 0.9 + 0.1 * threshold_ratio
            simulated_metrics["throughput"] *= 1.1 - 0.1 * threshold_ratio
            simulated_metrics["success_rate"] *= 0.95 + 0.05 * threshold_ratio
            simulated_metrics["avg_confidence"] *= 0.95 + 0.05 * threshold_ratio

        else:
            # Higher threshold: potentially better accuracy, slower performance
            simulated_metrics["avg_processing_time"] *= 1.1 - 0.1 / threshold_ratio
            simulated_metrics["throughput"] *= 0.9 + 0.1 / threshold_ratio
            simulated_metrics["success_rate"] *= 1.02 - 0.02 / threshold_ratio
            simulated_metrics["avg_confidence"] *= 1.02 - 0.02 / threshold_ratio

        # Ensure realistic bounds
        simulated_metrics["success_rate"] = min(1.0, max(0.0, simulated_metrics["success_rate"]))
        simulated_metrics["avg_confidence"] = min(1.0, max(0.0, simulated_metrics["avg_confidence"]))
        simulated_metrics["error_rate"] = 1 - simulated_metrics["success_rate"]

        return simulated_metrics

    def _evaluate_improvement(
        self, current_metrics: Dict[str, float], simulated_metrics: Dict[str, float], strategy: OptimizationStrategy
    ) -> float:
        """Evaluate improvement percentage based on strategy"""

        if strategy == OptimizationStrategy.PERFORMANCE_FOCUSED:
            # Focus on processing time and throughput
            time_improvement = (
                current_metrics["avg_processing_time"] - simulated_metrics["avg_processing_time"]
            ) / current_metrics["avg_processing_time"]
            throughput_improvement = (simulated_metrics["throughput"] - current_metrics["throughput"]) / current_metrics[
                "throughput"
            ]
            return (time_improvement * 0.6 + throughput_improvement * 0.4) * 100

        elif strategy == OptimizationStrategy.ACCURACY_FOCUSED:
            # Focus on success rate and confidence
            success_improvement = (simulated_metrics["success_rate"] - current_metrics["success_rate"]) / current_metrics[
                "success_rate"
            ]
            confidence_improvement = (
                simulated_metrics["avg_confidence"] - current_metrics["avg_confidence"]
            ) / current_metrics["avg_confidence"]
            return (success_improvement * 0.6 + confidence_improvement * 0.4) * 100

        elif strategy == OptimizationStrategy.BALANCED:
            # Balanced approach
            time_improvement = (
                current_metrics["avg_processing_time"] - simulated_metrics["avg_processing_time"]
            ) / current_metrics["avg_processing_time"]
            success_improvement = (simulated_metrics["success_rate"] - current_metrics["success_rate"]) / current_metrics[
                "success_rate"
            ]
            throughput_improvement = (simulated_metrics["throughput"] - current_metrics["throughput"]) / current_metrics[
                "throughput"
            ]
            confidence_improvement = (
                simulated_metrics["avg_confidence"] - current_metrics["avg_confidence"]
            ) / current_metrics["avg_confidence"]

            return (
                time_improvement * 0.3
                + success_improvement * 0.3
                + throughput_improvement * 0.2
                + confidence_improvement * 0.2
            ) * 100

        else:  # ADAPTIVE
            # Adaptive strategy based on current performance
            if current_metrics["avg_processing_time"] > 100:  # Focus on speed if slow
                return self._evaluate_improvement(current_metrics, simulated_metrics, OptimizationStrategy.PERFORMANCE_FOCUSED)
            elif current_metrics["success_rate"] < 0.85:  # Focus on accuracy if low
                return self._evaluate_improvement(current_metrics, simulated_metrics, OptimizationStrategy.ACCURACY_FOCUSED)
            else:  # Balanced approach if performing well
                return self._evaluate_improvement(current_metrics, simulated_metrics, OptimizationStrategy.BALANCED)

    def _calculate_recommendation_confidence(
        self, events: List[MetricEvent], current_metrics: Dict[str, float], simulated_metrics: Dict[str, float]
    ) -> float:
        """Calculate confidence level for a recommendation"""

        confidence_factors = []

        # Sample size factor
        sample_size_confidence = min(1.0, len(events) / 1000)  # Full confidence at 1000+ samples
        confidence_factors.append(sample_size_confidence)

        # Stability factor (consistent performance)
        processing_times = [e.processing_time_ms for e in events if e.processing_time_ms > 0]
        if processing_times:
            cv = statistics.stdev(processing_times) / statistics.mean(processing_times)  # Coefficient of variation
            stability_confidence = max(0.3, 1 - cv)  # Lower CV = higher confidence
            confidence_factors.append(stability_confidence)

        # Improvement magnitude factor
        improvement_magnitude = abs(simulated_metrics.get("success_rate", 0) - current_metrics.get("success_rate", 0))
        magnitude_confidence = min(1.0, improvement_magnitude * 10)  # Higher for larger improvements
        confidence_factors.append(magnitude_confidence)

        # Historical success factor (if available)
        historical_confidence = 0.8  # Default moderate confidence
        confidence_factors.append(historical_confidence)

        # Overall confidence is the weighted average
        return sum(confidence_factors) / len(confidence_factors)

    def _create_recommendation(
        self,
        pattern_id: str,
        config: ThresholdConfig,
        recommended_threshold: float,
        expected_improvement: float,
        confidence_level: float,
        strategy: OptimizationStrategy,
        current_metrics: Dict[str, float],
        simulated_metrics: Dict[str, float],
    ) -> ThresholdRecommendation:
        """Create a threshold recommendation"""

        # Generate rationale
        if recommended_threshold < config.current_threshold:
            rationale = (
                f"Lower threshold from {config.current_threshold:.3f} to "
                f"{recommended_threshold:.3f} to improve processing speed by "
                f"{expected_improvement:.1f}%"
            )
        else:
            rationale = (
                f"Raise threshold from {config.current_threshold:.3f} to "
                f"{recommended_threshold:.3f} to improve accuracy by "
                f"{expected_improvement:.1f}%"
            )

        # Risk assessment
        threshold_change = abs(recommended_threshold - config.current_threshold)
        if threshold_change > 0.1:
            risk_assessment = "HIGH - Large threshold change, recommend gradual implementation"
        elif threshold_change > 0.05:
            risk_assessment = "MEDIUM - Moderate change, monitor closely after implementation"
        else:
            risk_assessment = "LOW - Small adjustment, safe to implement"

        # Supporting metrics
        supporting_metrics = {
            "current_success_rate": current_metrics.get("success_rate", 0),
            "projected_success_rate": simulated_metrics.get("success_rate", 0),
            "current_avg_time": current_metrics.get("avg_processing_time", 0),
            "projected_avg_time": simulated_metrics.get("avg_processing_time", 0),
            "sample_size": current_metrics.get("total_events", 0),
        }

        return ThresholdRecommendation(
            pattern_id=pattern_id,
            current_threshold=config.current_threshold,
            recommended_threshold=recommended_threshold,
            expected_improvement=expected_improvement,
            confidence_level=confidence_level,
            strategy=strategy,
            rationale=rationale,
            supporting_metrics=supporting_metrics,
            risk_assessment=risk_assessment,
        )

    def _calculate_overall_improvement(self, recommendations: List[ThresholdRecommendation]) -> float:
        """Calculate expected overall system improvement"""
        if not recommendations:
            return 0.0

        # Weight improvements by confidence and impact
        weighted_improvements = []
        total_weight = 0

        for rec in recommendations:
            weight = rec.confidence_level * (1 + rec.expected_improvement / 100)
            weighted_improvements.append(rec.expected_improvement * weight)
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return sum(weighted_improvements) / total_weight

    def _create_empty_result(self) -> OptimizationResult:
        """Create empty optimization result"""
        return OptimizationResult(
            total_patterns_analyzed=0,
            recommendations_generated=0,
            expected_overall_improvement=0.0,
            high_impact_recommendations=[],
            medium_impact_recommendations=[],
            low_impact_recommendations=[],
            risk_warnings=["No recent data available for analysis"],
            timestamp=datetime.now(),
        )

    def _load_default_configs(self) -> Dict[str, ThresholdConfig]:
        """Load default threshold configurations"""
        return {
            "street_pattern": ThresholdConfig("street_pattern", 0.75),
            "building_pattern": ThresholdConfig("building_pattern", 0.82),
            "district_pattern": ThresholdConfig("district_pattern", 0.68),
            "apartment_pattern": ThresholdConfig("apartment_pattern", 0.70),
            "general_pattern": ThresholdConfig("general_pattern", 0.65),
        }

    def apply_recommendation(self, recommendation: ThresholdRecommendation) -> bool:
        """Apply a threshold recommendation (placeholder for actual implementation)"""
        # This would integrate with the actual pattern configuration system
        self.logger.info(
            f"Applied threshold recommendation for {recommendation.pattern_id}: "
            f"{recommendation.current_threshold} → {recommendation.recommended_threshold}"
        )

        # Update local configuration
        if recommendation.pattern_id in self.default_configs:
            self.default_configs[recommendation.pattern_id].current_threshold = recommendation.recommended_threshold

        return True

    def get_optimization_history(self, days: int = 30) -> List[OptimizationResult]:
        """Get optimization history for the specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [result for result in self.optimization_history if result.timestamp >= cutoff_date]
