"""
Adaptive Learning Engine - Core intelligence for automatic threshold optimization.

Analyzes pattern performance history and provides intelligent threshold adjustments
based on success rates, confidence levels, and processing volumes.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import asdict

from .models import (
    PatternPerformance,
    PerformanceMetrics,
    ThresholdUpdate,
    LearningConfig,
    OptimizationResult,
    OptimizationStrategy,
    PerformanceTrend,
    LearningState,
)

logger = logging.getLogger(__name__)


class AdaptiveLearningEngine:
    """
    Core adaptive learning engine that analyzes pattern performance
    and provides intelligent threshold optimization recommendations.
    """

    def __init__(self, config: LearningConfig, storage_path: str = "data/adaptive_learning"):
        """
        Initialize adaptive learning engine.

        Args:
            config: Learning configuration
            storage_path: Path to store performance data
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Pattern performance tracking
        self.pattern_performances: Dict[str, PatternPerformance] = {}
        self.threshold_history: List[ThresholdUpdate] = []
        self.learning_state = LearningState()

        # Load existing data
        self._load_data()

        logger.info(f"AdaptiveLearningEngine initialized with {len(self.pattern_performances)} patterns")

    def record_pattern_usage(
        self,
        pattern_id: str,
        pattern_type: str,
        success: bool,
        confidence: float,
        processing_time: float,
        threshold_used: float,
    ) -> None:
        """
        Record pattern usage for learning.

        Args:
            pattern_id: Unique pattern identifier
            pattern_type: Type of pattern (address, street, etc.)
            success: Whether pattern matching was successful
            confidence: Confidence score of the match
            processing_time: Time taken to process
            threshold_used: Threshold value used
        """
        # Get or create pattern performance
        if pattern_id not in self.pattern_performances:
            metrics = PerformanceMetrics(
                pattern_id=pattern_id,
                success_rate=0.0,
                total_processed=0,
                successful_matches=0,
                failed_matches=0,
                average_confidence=0.0,
                confidence_variance=0.0,
                processing_time_avg=0.0,
            )

            self.pattern_performances[pattern_id] = PatternPerformance(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                current_threshold=threshold_used,
                optimal_threshold=threshold_used,
                metrics=metrics,
            )

        # Update metrics
        pattern_perf = self.pattern_performances[pattern_id]
        pattern_perf.metrics.update_metrics(success, confidence, processing_time)
        pattern_perf.current_threshold = threshold_used

        # Add historical data point every 10 usages
        if pattern_perf.metrics.total_processed % 10 == 0:
            pattern_perf.add_historical_point(
                success_rate=pattern_perf.metrics.success_rate,
                threshold=threshold_used,
                volume=pattern_perf.metrics.total_processed,
            )

        # Calculate trend
        pattern_perf.calculate_trend()

        logger.debug(f"Recorded usage for pattern {pattern_id}: success={success}, confidence={confidence:.3f}")

    def analyze_pattern_performance(self, pattern_id: str) -> Dict:
        """
        Analyze performance of a specific pattern.

        Args:
            pattern_id: Pattern to analyze

        Returns:
            Analysis results
        """
        if pattern_id not in self.pattern_performances:
            return {"error": f"Pattern {pattern_id} not found"}

        pattern_perf = self.pattern_performances[pattern_id]
        metrics = pattern_perf.metrics

        # Calculate performance score
        performance_score = self._calculate_performance_score(pattern_perf)

        # Determine if optimization is needed
        needs_optimization = self._needs_optimization(pattern_perf)

        # Calculate recommended threshold
        recommended_threshold = self._calculate_optimal_threshold(pattern_perf)

        analysis = {
            "pattern_id": pattern_id,
            "current_performance": {
                "success_rate": metrics.success_rate,
                "total_processed": metrics.total_processed,
                "average_confidence": metrics.average_confidence,
                "processing_time_avg": metrics.processing_time_avg,
                "trend": pattern_perf.trend.value,
            },
            "threshold_analysis": {
                "current_threshold": pattern_perf.current_threshold,
                "optimal_threshold": pattern_perf.optimal_threshold,
                "recommended_threshold": recommended_threshold,
                "needs_optimization": needs_optimization,
            },
            "performance_score": performance_score,
            "optimization_potential": abs(recommended_threshold - pattern_perf.current_threshold),
            "last_optimization": pattern_perf.last_optimization.isoformat() if pattern_perf.last_optimization else None,
        }

        return analysis

    def optimize_threshold(self, pattern_id: str) -> OptimizationResult:
        """
        Optimize threshold for a specific pattern.

        Args:
            pattern_id: Pattern to optimize

        Returns:
            Optimization result
        """
        if pattern_id not in self.pattern_performances:
            return OptimizationResult(
                pattern_id=pattern_id,
                old_threshold=0.0,
                new_threshold=0.0,
                expected_improvement=0.0,
                confidence_score=0.0,
                optimization_applied=False,
                reason="Pattern not found",
            )

        pattern_perf = self.pattern_performances[pattern_id]
        old_threshold = pattern_perf.current_threshold

        # Check if optimization is needed
        if not self._needs_optimization(pattern_perf):
            return OptimizationResult(
                pattern_id=pattern_id,
                old_threshold=old_threshold,
                new_threshold=old_threshold,
                expected_improvement=0.0,
                confidence_score=0.0,
                optimization_applied=False,
                reason="Optimization not needed - performance satisfactory",
            )

        # Calculate optimal threshold
        new_threshold = self._calculate_optimal_threshold(pattern_perf)

        # Validate threshold change
        threshold_change = abs(new_threshold - old_threshold)
        if threshold_change > self.config.max_threshold_change:
            # Limit the change
            if new_threshold > old_threshold:
                new_threshold = old_threshold + self.config.max_threshold_change
            else:
                new_threshold = old_threshold - self.config.max_threshold_change

        # Ensure threshold bounds
        new_threshold = max(self.config.min_threshold, min(self.config.max_threshold, new_threshold))

        # Calculate expected improvement
        expected_improvement = self._estimate_improvement(pattern_perf, new_threshold)
        confidence_score = self._calculate_optimization_confidence(pattern_perf)

        # Apply optimization if improvement is significant
        optimization_applied = False
        reason = "No significant improvement expected"

        if expected_improvement > 0.01:  # At least 1% improvement
            pattern_perf.optimal_threshold = new_threshold
            pattern_perf.last_optimization = datetime.now()
            pattern_perf.optimization_count += 1
            optimization_applied = True
            reason = f"Expected {expected_improvement:.1%} improvement"

            # Record threshold update
            update = ThresholdUpdate(
                pattern_id=pattern_id,
                old_threshold=old_threshold,
                new_threshold=new_threshold,
                reason=reason,
                performance_before=pattern_perf.metrics.success_rate,
                expected_improvement=expected_improvement,
                strategy_used=self.config.optimization_strategy,
            )
            self.threshold_history.append(update)

            logger.info(f"Optimized threshold for {pattern_id}: {old_threshold:.3f} → {new_threshold:.3f}")

        return OptimizationResult(
            pattern_id=pattern_id,
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            expected_improvement=expected_improvement,
            confidence_score=confidence_score,
            optimization_applied=optimization_applied,
            reason=reason,
        )

    def batch_optimize_all_patterns(self) -> List[OptimizationResult]:
        """
        Optimize thresholds for all eligible patterns.

        Returns:
            List of optimization results
        """
        results = []

        for pattern_id in self.pattern_performances:
            result = self.optimize_threshold(pattern_id)
            results.append(result)

        # Update learning state
        self.learning_state.update_stats(results)

        # Save data
        self._save_data()

        optimized_count = sum(1 for r in results if r.optimization_applied)
        logger.info(f"Batch optimization completed: {optimized_count}/{len(results)} patterns optimized")

        return results

    def get_system_performance_report(self) -> Dict:
        """
        Get comprehensive system performance report.

        Returns:
            Performance report
        """
        if not self.pattern_performances:
            return {"error": "No patterns tracked yet"}

        # Calculate aggregate metrics
        total_processed = sum(p.metrics.total_processed for p in self.pattern_performances.values())
        total_successful = sum(p.metrics.successful_matches for p in self.pattern_performances.values())
        overall_success_rate = total_successful / total_processed if total_processed > 0 else 0.0

        avg_confidence = np.mean([p.metrics.average_confidence for p in self.pattern_performances.values()])
        avg_processing_time = np.mean([p.metrics.processing_time_avg for p in self.pattern_performances.values()])

        # Pattern statistics
        pattern_trends = {}
        for trend in PerformanceTrend:
            count = sum(1 for p in self.pattern_performances.values() if p.trend == trend)
            pattern_trends[trend.value] = count

        # Performance by pattern type
        type_performance = {}
        for pattern in self.pattern_performances.values():
            ptype = pattern.pattern_type
            if ptype not in type_performance:
                type_performance[ptype] = {"count": 0, "success_rate": 0.0, "confidence": 0.0}

            type_performance[ptype]["count"] += 1
            type_performance[ptype]["success_rate"] += pattern.metrics.success_rate
            type_performance[ptype]["confidence"] += pattern.metrics.average_confidence

        # Calculate averages
        for ptype in type_performance:
            count = type_performance[ptype]["count"]
            type_performance[ptype]["success_rate"] /= count
            type_performance[ptype]["confidence"] /= count

        # Recent optimizations
        recent_optimizations = len(
            [update for update in self.threshold_history if update.timestamp > datetime.now() - timedelta(days=7)]
        )

        report = {
            "system_overview": {
                "total_patterns_tracked": len(self.pattern_performances),
                "total_processed": total_processed,
                "overall_success_rate": overall_success_rate,
                "average_confidence": avg_confidence,
                "average_processing_time": avg_processing_time,
            },
            "pattern_trends": pattern_trends,
            "performance_by_type": type_performance,
            "optimization_stats": {
                "total_optimizations": len(self.threshold_history),
                "recent_optimizations": recent_optimizations,
                "average_improvement": self.learning_state.average_improvement,
                "last_optimization": (
                    self.learning_state.last_optimization_run.isoformat()
                    if self.learning_state.last_optimization_run
                    else None
                ),
            },
            "learning_state": asdict(self.learning_state),
        }

        return report

    def _calculate_performance_score(self, pattern_perf: PatternPerformance) -> float:
        """Calculate overall performance score for a pattern"""
        metrics = pattern_perf.metrics

        # Base score from success rate
        success_score = metrics.success_rate

        # Confidence bonus/penalty
        confidence_score = min(1.0, metrics.average_confidence)

        # Volume reliability factor
        volume_factor = min(1.0, metrics.total_processed / 100)  # More reliable with more data

        # Trend factor
        trend_factor = {
            PerformanceTrend.IMPROVING: 1.1,
            PerformanceTrend.STABLE: 1.0,
            PerformanceTrend.DECLINING: 0.9,
            PerformanceTrend.VOLATILE: 0.8,
        }.get(pattern_perf.trend, 1.0)

        # Weighted score
        score = (success_score * 0.5 + confidence_score * 0.3 + volume_factor * 0.2) * trend_factor

        return min(1.0, score)

    def _needs_optimization(self, pattern_perf: PatternPerformance) -> bool:
        """Determine if pattern needs threshold optimization"""
        metrics = pattern_perf.metrics

        # Need minimum samples
        if metrics.total_processed < self.config.min_samples_for_optimization:
            return False

        # Check if recently optimized
        if pattern_perf.last_optimization:
            hours_since_optimization = (datetime.now() - pattern_perf.last_optimization).total_seconds() / 3600
            if hours_since_optimization < self.config.optimization_frequency_hours:
                return False

        # Performance-based criteria
        success_rate = metrics.success_rate
        target_rate = self.config.success_rate_target

        # Need optimization if performance is below target or declining
        if success_rate < target_rate:
            return True

        if pattern_perf.trend == PerformanceTrend.DECLINING:
            return True

        # Check if threshold is far from optimal
        optimal_threshold = self._calculate_optimal_threshold(pattern_perf)
        threshold_diff = abs(optimal_threshold - pattern_perf.current_threshold)

        return threshold_diff > 0.05  # 5% difference threshold

    def _calculate_optimal_threshold(self, pattern_perf: PatternPerformance) -> float:
        """Calculate optimal threshold based on performance data"""
        metrics = pattern_perf.metrics

        # Strategy-specific calculation
        if self.config.optimization_strategy == OptimizationStrategy.CONSERVATIVE:
            return self._conservative_threshold(pattern_perf)
        elif self.config.optimization_strategy == OptimizationStrategy.AGGRESSIVE:
            return self._aggressive_threshold(pattern_perf)
        elif self.config.optimization_strategy == OptimizationStrategy.VOLUME_WEIGHTED:
            return self._volume_weighted_threshold(pattern_perf)
        else:  # BALANCED
            return self._balanced_threshold(pattern_perf)

    def _conservative_threshold(self, pattern_perf: PatternPerformance) -> float:
        """Conservative threshold calculation - prioritizes precision"""
        metrics = pattern_perf.metrics
        current_threshold = pattern_perf.current_threshold

        # If success rate is high, slightly lower threshold
        if metrics.success_rate >= 0.9:
            return max(self.config.min_threshold, current_threshold - 0.05)
        # If success rate is low, increase threshold
        elif metrics.success_rate < 0.7:
            return min(self.config.max_threshold, current_threshold + 0.1)

        return current_threshold

    def _aggressive_threshold(self, pattern_perf: PatternPerformance) -> float:
        """Aggressive threshold calculation - quick adjustments"""
        metrics = pattern_perf.metrics
        current_threshold = pattern_perf.current_threshold
        target = self.config.success_rate_target

        # More dramatic adjustments
        if metrics.success_rate > target + 0.1:
            return max(self.config.min_threshold, current_threshold - 0.1)
        elif metrics.success_rate < target - 0.1:
            return min(self.config.max_threshold, current_threshold + 0.15)

        return current_threshold

    def _volume_weighted_threshold(self, pattern_perf: PatternPerformance) -> float:
        """Volume-weighted threshold calculation"""
        metrics = pattern_perf.metrics
        current_threshold = pattern_perf.current_threshold

        # Weight adjustments by volume reliability
        volume_factor = min(1.0, metrics.total_processed / 1000)
        adjustment_strength = 0.05 + (0.1 * volume_factor)

        if metrics.success_rate < self.config.success_rate_target:
            return min(self.config.max_threshold, current_threshold + adjustment_strength)
        elif metrics.success_rate > self.config.success_rate_target + 0.1:
            return max(self.config.min_threshold, current_threshold - adjustment_strength)

        return current_threshold

    def _balanced_threshold(self, pattern_perf: PatternPerformance) -> float:
        """Balanced threshold calculation - default strategy"""
        metrics = pattern_perf.metrics
        current_threshold = pattern_perf.current_threshold
        target = self.config.success_rate_target

        # Multi-factor calculation
        success_gap = target - metrics.success_rate
        confidence_factor = metrics.average_confidence
        trend_factor = {
            PerformanceTrend.IMPROVING: -0.02,  # Slightly lower threshold
            PerformanceTrend.STABLE: 0.0,
            PerformanceTrend.DECLINING: 0.05,  # Higher threshold
            PerformanceTrend.VOLATILE: 0.03,  # Slightly higher for stability
        }.get(pattern_perf.trend, 0.0)

        # Calculate adjustment
        base_adjustment = success_gap * 0.5  # 50% of gap
        confidence_adjustment = (0.8 - confidence_factor) * 0.1  # Confidence penalty

        total_adjustment = base_adjustment + confidence_adjustment + trend_factor

        new_threshold = current_threshold + total_adjustment

        return max(self.config.min_threshold, min(self.config.max_threshold, new_threshold))

    def _estimate_improvement(self, pattern_perf: PatternPerformance, new_threshold: float) -> float:
        """Estimate performance improvement from threshold change"""
        current_rate = pattern_perf.metrics.success_rate
        threshold_change = new_threshold - pattern_perf.current_threshold

        # Simple linear model - in reality this would be more sophisticated
        # Based on historical data analysis
        if threshold_change > 0:  # Increasing threshold
            # Generally improves precision but may reduce recall
            estimated_improvement = min(0.1, threshold_change * 0.5)
        else:  # Decreasing threshold
            # May improve recall but could reduce precision
            estimated_improvement = min(0.05, abs(threshold_change) * 0.3)

        # Adjust based on current performance
        if current_rate > 0.9:
            estimated_improvement *= 0.5  # Less room for improvement
        elif current_rate < 0.6:
            estimated_improvement *= 1.5  # More room for improvement

        return estimated_improvement

    def _calculate_optimization_confidence(self, pattern_perf: PatternPerformance) -> float:
        """Calculate confidence in optimization recommendation"""
        metrics = pattern_perf.metrics

        # Volume confidence
        volume_conf = min(1.0, metrics.total_processed / 500)

        # Trend confidence
        trend_conf = {
            PerformanceTrend.STABLE: 0.9,
            PerformanceTrend.IMPROVING: 0.8,
            PerformanceTrend.DECLINING: 0.7,
            PerformanceTrend.VOLATILE: 0.5,
        }.get(pattern_perf.trend, 0.7)

        # Historical data confidence
        hist_conf = min(1.0, len(pattern_perf.historical_data) / 50)

        # Combined confidence
        return volume_conf * 0.4 + trend_conf * 0.4 + hist_conf * 0.2

    def _load_data(self) -> None:
        """Load performance data from storage"""
        try:
            # Load pattern performances
            perf_file = self.storage_path / "pattern_performances.json"
            if perf_file.exists():
                with open(perf_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert back to objects (simplified for now)
                    logger.info(f"Loaded performance data for {len(data)} patterns")

            # Load threshold history
            hist_file = self.storage_path / "threshold_history.json"
            if hist_file.exists():
                with open(hist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.threshold_history = [ThresholdUpdate.from_dict(item) for item in data]
                    logger.info(f"Loaded {len(self.threshold_history)} threshold updates")

        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")

    def _save_data(self) -> None:
        """Save performance data to storage"""
        try:
            # Save pattern performances (simplified)
            perf_file = self.storage_path / "pattern_performances.json"
            perf_data = {
                pattern_id: {
                    "pattern_id": perf.pattern_id,
                    "pattern_type": perf.pattern_type,
                    "current_threshold": perf.current_threshold,
                    "optimal_threshold": perf.optimal_threshold,
                    "success_rate": perf.metrics.success_rate,
                    "total_processed": perf.metrics.total_processed,
                    "last_updated": perf.metrics.last_updated.isoformat(),
                }
                for pattern_id, perf in self.pattern_performances.items()
            }

            with open(perf_file, "w", encoding="utf-8") as f:
                json.dump(perf_data, f, indent=2, ensure_ascii=False)

            # Save threshold history
            hist_file = self.storage_path / "threshold_history.json"
            hist_data = [update.to_dict() for update in self.threshold_history]

            with open(hist_file, "w", encoding="utf-8") as f:
                json.dump(hist_data, f, indent=2, ensure_ascii=False)

            logger.debug("Learning data saved successfully")

        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
