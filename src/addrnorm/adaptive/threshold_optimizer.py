"""
Threshold Optimizer - Advanced algorithms for dynamic threshold calculation.

Implements multiple optimization strategies with statistical models and
machine learning approaches for threshold optimization.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import optimize
from datetime import datetime, timedelta
import json
from pathlib import Path

from .models import PatternPerformance, LearningConfig, OptimizationStrategy, PerformanceTrend, OptimizationResult

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Advanced threshold optimizer using statistical models and
    optimization algorithms for dynamic threshold calculation.
    """

    def __init__(self, config: LearningConfig):
        """
        Initialize threshold optimizer.

        Args:
            config: Learning configuration
        """
        self.config = config
        self.optimization_cache: Dict[str, Dict] = {}

        logger.info("ThresholdOptimizer initialized")

    def optimize_single_pattern(self, pattern_perf: PatternPerformance) -> OptimizationResult:
        """
        Optimize threshold for a single pattern using advanced algorithms.

        Args:
            pattern_perf: Pattern performance data

        Returns:
            Optimization result with recommended threshold
        """
        pattern_id = pattern_perf.pattern_id
        current_threshold = pattern_perf.current_threshold

        # Check cache for recent optimization
        if self._has_recent_optimization(pattern_id):
            cached_result = self.optimization_cache[pattern_id]
            return OptimizationResult(
                pattern_id=pattern_id,
                old_threshold=current_threshold,
                new_threshold=cached_result["threshold"],
                expected_improvement=cached_result["improvement"],
                confidence_score=cached_result["confidence"],
                optimization_applied=False,
                reason="Using cached optimization result",
            )

        # Determine optimization strategy
        strategy = self._select_optimization_strategy(pattern_perf)

        # Apply strategy-specific optimization
        if strategy == OptimizationStrategy.CONSERVATIVE:
            result = self._conservative_optimization(pattern_perf)
        elif strategy == OptimizationStrategy.AGGRESSIVE:
            result = self._aggressive_optimization(pattern_perf)
        elif strategy == OptimizationStrategy.VOLUME_WEIGHTED:
            result = self._volume_weighted_optimization(pattern_perf)
        else:  # BALANCED
            result = self._balanced_optimization(pattern_perf)

        # Cache result
        self.optimization_cache[pattern_id] = {
            "threshold": result.new_threshold,
            "improvement": result.expected_improvement,
            "confidence": result.confidence_score,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def batch_optimize_with_constraints(
        self, patterns: List[PatternPerformance], global_constraints: Dict = None
    ) -> List[OptimizationResult]:
        """
        Optimize multiple patterns with global constraints.

        Args:
            patterns: List of pattern performances
            global_constraints: Global optimization constraints

        Returns:
            List of optimization results
        """
        if global_constraints is None:
            global_constraints = {}

        results = []

        # Group patterns by type for coordinated optimization
        pattern_groups = self._group_patterns_by_type(patterns)

        for pattern_type, type_patterns in pattern_groups.items():
            logger.info(f"Optimizing {len(type_patterns)} patterns of type {pattern_type}")

            # Apply type-specific optimization
            type_results = self._optimize_pattern_group(type_patterns, global_constraints)
            results.extend(type_results)

        # Post-process results to ensure global constraints
        results = self._apply_global_constraints(results, global_constraints)

        return results

    def find_optimal_threshold_distribution(self, patterns: List[PatternPerformance]) -> Dict:
        """
        Find optimal threshold distribution across all patterns.

        Args:
            patterns: List of pattern performances

        Returns:
            Optimal threshold distribution analysis
        """
        if not patterns:
            return {"error": "No patterns provided"}

        # Extract performance data
        thresholds = [p.current_threshold for p in patterns]
        success_rates = [p.metrics.success_rate for p in patterns]
        volumes = [p.metrics.total_processed for p in patterns]

        # Calculate statistics
        threshold_stats = {
            "mean": np.mean(thresholds),
            "median": np.median(thresholds),
            "std": np.std(thresholds),
            "min": np.min(thresholds),
            "max": np.max(thresholds),
        }

        # Find optimal distribution using optimization
        optimal_distribution = self._calculate_optimal_distribution(patterns)

        # Performance correlation analysis
        correlation_analysis = self._analyze_threshold_performance_correlation(patterns)

        return {
            "current_distribution": threshold_stats,
            "optimal_distribution": optimal_distribution,
            "correlation_analysis": correlation_analysis,
            "recommendations": self._generate_distribution_recommendations(patterns),
        }

    def adaptive_threshold_learning(self, pattern_perf: PatternPerformance, learning_rate: float = 0.1) -> float:
        """
        Adaptive threshold learning using gradient-based optimization.

        Args:
            pattern_perf: Pattern performance data
            learning_rate: Learning rate for gradient descent

        Returns:
            Optimized threshold
        """
        current_threshold = pattern_perf.current_threshold

        # Calculate performance gradient
        gradient = self._calculate_performance_gradient(pattern_perf)

        # Apply gradient descent step
        new_threshold = current_threshold - learning_rate * gradient

        # Apply constraints
        new_threshold = max(self.config.min_threshold, min(self.config.max_threshold, new_threshold))

        # Validate improvement
        if self._validate_threshold_improvement(pattern_perf, new_threshold):
            return new_threshold
        else:
            return current_threshold

    def multi_objective_optimization(self, pattern_perf: PatternPerformance) -> Dict:
        """
        Multi-objective optimization balancing multiple performance metrics.

        Args:
            pattern_perf: Pattern performance data

        Returns:
            Pareto-optimal solutions
        """

        # Define objectives: success_rate, confidence, processing_time
        def objective_function(threshold):
            # Simulate performance at different thresholds
            success_rate = self._estimate_success_rate(pattern_perf, threshold)
            confidence = self._estimate_confidence(pattern_perf, threshold)
            processing_time = self._estimate_processing_time(pattern_perf, threshold)

            # Convert to minimization problem (negative for maximization)
            return [-success_rate, -confidence, processing_time]

        # Define threshold search space
        threshold_bounds = [(self.config.min_threshold, self.config.max_threshold)]

        # Find Pareto-optimal solutions
        pareto_solutions = []
        threshold_candidates = np.linspace(self.config.min_threshold, self.config.max_threshold, 20)

        for threshold in threshold_candidates:
            objectives = objective_function(threshold)
            pareto_solutions.append(
                {
                    "threshold": threshold,
                    "success_rate": -objectives[0],
                    "confidence": -objectives[1],
                    "processing_time": objectives[2],
                }
            )

        # Filter Pareto-optimal solutions
        pareto_optimal = self._filter_pareto_optimal(pareto_solutions)

        # Select best solution based on weights
        best_solution = self._select_best_pareto_solution(pareto_optimal, pattern_perf)

        return {"all_solutions": pareto_solutions, "pareto_optimal": pareto_optimal, "recommended_solution": best_solution}

    def _select_optimization_strategy(self, pattern_perf: PatternPerformance) -> OptimizationStrategy:
        """Select appropriate optimization strategy based on pattern characteristics"""
        metrics = pattern_perf.metrics

        # High volume patterns -> volume weighted
        if metrics.total_processed > 1000:
            return OptimizationStrategy.VOLUME_WEIGHTED

        # Volatile patterns -> conservative
        if pattern_perf.trend == PerformanceTrend.VOLATILE:
            return OptimizationStrategy.CONSERVATIVE

        # Poor performance -> aggressive
        if metrics.success_rate < 0.6:
            return OptimizationStrategy.AGGRESSIVE

        # Default
        return OptimizationStrategy.BALANCED

    def _conservative_optimization(self, pattern_perf: PatternPerformance) -> OptimizationResult:
        """Conservative optimization strategy"""
        current_threshold = pattern_perf.current_threshold
        success_rate = pattern_perf.metrics.success_rate

        # Small, safe adjustments
        if success_rate < self.config.success_rate_target:
            adjustment = min(0.05, (self.config.success_rate_target - success_rate) * 0.5)
            new_threshold = min(self.config.max_threshold, current_threshold + adjustment)
        elif success_rate > self.config.success_rate_target + 0.1:
            adjustment = min(0.03, (success_rate - self.config.success_rate_target) * 0.3)
            new_threshold = max(self.config.min_threshold, current_threshold - adjustment)
        else:
            new_threshold = current_threshold

        expected_improvement = self._estimate_improvement(pattern_perf, new_threshold)
        confidence = 0.8  # Conservative strategy has high confidence

        return OptimizationResult(
            pattern_id=pattern_perf.pattern_id,
            old_threshold=current_threshold,
            new_threshold=new_threshold,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            optimization_applied=abs(new_threshold - current_threshold) > 0.01,
            reason="Conservative optimization applied",
        )

    def _aggressive_optimization(self, pattern_perf: PatternPerformance) -> OptimizationResult:
        """Aggressive optimization strategy"""
        current_threshold = pattern_perf.current_threshold
        success_rate = pattern_perf.metrics.success_rate

        # Larger adjustments for faster convergence
        if success_rate < self.config.success_rate_target:
            adjustment = min(0.15, (self.config.success_rate_target - success_rate) * 1.0)
            new_threshold = min(self.config.max_threshold, current_threshold + adjustment)
        elif success_rate > self.config.success_rate_target + 0.05:
            adjustment = min(0.1, (success_rate - self.config.success_rate_target) * 0.7)
            new_threshold = max(self.config.min_threshold, current_threshold - adjustment)
        else:
            new_threshold = current_threshold

        expected_improvement = self._estimate_improvement(pattern_perf, new_threshold) * 1.2
        confidence = 0.6  # Aggressive strategy has lower confidence

        return OptimizationResult(
            pattern_id=pattern_perf.pattern_id,
            old_threshold=current_threshold,
            new_threshold=new_threshold,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            optimization_applied=abs(new_threshold - current_threshold) > 0.01,
            reason="Aggressive optimization applied",
        )

    def _volume_weighted_optimization(self, pattern_perf: PatternPerformance) -> OptimizationResult:
        """Volume-weighted optimization strategy"""
        current_threshold = pattern_perf.current_threshold
        volume = pattern_perf.metrics.total_processed
        success_rate = pattern_perf.metrics.success_rate

        # Weight adjustments by data volume (more data = more confident adjustments)
        volume_factor = min(1.0, volume / 1000)
        base_adjustment = (self.config.success_rate_target - success_rate) * 0.8
        weighted_adjustment = base_adjustment * volume_factor

        new_threshold = current_threshold + weighted_adjustment
        new_threshold = max(self.config.min_threshold, min(self.config.max_threshold, new_threshold))

        expected_improvement = self._estimate_improvement(pattern_perf, new_threshold)
        confidence = 0.5 + 0.4 * volume_factor  # Higher confidence with more data

        return OptimizationResult(
            pattern_id=pattern_perf.pattern_id,
            old_threshold=current_threshold,
            new_threshold=new_threshold,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            optimization_applied=abs(new_threshold - current_threshold) > 0.01,
            reason=f"Volume-weighted optimization (volume={volume})",
        )

    def _balanced_optimization(self, pattern_perf: PatternPerformance) -> OptimizationResult:
        """Balanced optimization strategy"""
        # Use multi-objective optimization for balanced approach
        mo_result = self.multi_objective_optimization(pattern_perf)
        best_solution = mo_result["recommended_solution"]

        return OptimizationResult(
            pattern_id=pattern_perf.pattern_id,
            old_threshold=pattern_perf.current_threshold,
            new_threshold=best_solution["threshold"],
            expected_improvement=self._estimate_improvement(pattern_perf, best_solution["threshold"]),
            confidence_score=0.7,  # Balanced confidence
            optimization_applied=abs(best_solution["threshold"] - pattern_perf.current_threshold) > 0.01,
            reason="Balanced multi-objective optimization",
        )

    def _group_patterns_by_type(self, patterns: List[PatternPerformance]) -> Dict[str, List[PatternPerformance]]:
        """Group patterns by type for coordinated optimization"""
        groups = {}
        for pattern in patterns:
            ptype = pattern.pattern_type
            if ptype not in groups:
                groups[ptype] = []
            groups[ptype].append(pattern)
        return groups

    def _optimize_pattern_group(self, patterns: List[PatternPerformance], constraints: Dict) -> List[OptimizationResult]:
        """Optimize a group of patterns with coordinated strategy"""
        results = []

        # Calculate group statistics
        group_success_rate = np.mean([p.metrics.success_rate for p in patterns])
        group_volume = sum(p.metrics.total_processed for p in patterns)

        # Determine group strategy
        if group_success_rate < 0.7:
            # Group needs improvement - apply aggressive optimization
            for pattern in patterns:
                result = self._aggressive_optimization(pattern)
                results.append(result)
        else:
            # Group performing well - apply conservative optimization
            for pattern in patterns:
                result = self._conservative_optimization(pattern)
                results.append(result)

        return results

    def _apply_global_constraints(self, results: List[OptimizationResult], constraints: Dict) -> List[OptimizationResult]:
        """Apply global constraints to optimization results"""
        # Example constraints: max change per batch, total system impact, etc.
        max_changes = constraints.get("max_changes_per_batch", len(results))

        # Sort by expected improvement and take top changes
        sorted_results = sorted(results, key=lambda r: r.expected_improvement, reverse=True)

        # Apply only top improvements
        for i, result in enumerate(sorted_results):
            if i >= max_changes:
                result.optimization_applied = False
                result.reason += " (Global constraint: max changes exceeded)"

        return results

    def _calculate_optimal_distribution(self, patterns: List[PatternPerformance]) -> Dict:
        """Calculate optimal threshold distribution"""

        # Use optimization to find best threshold distribution
        def objective(thresholds):
            total_performance = 0
            for i, pattern in enumerate(patterns):
                estimated_performance = self._estimate_success_rate(pattern, thresholds[i])
                volume_weight = pattern.metrics.total_processed / 100
                total_performance += estimated_performance * volume_weight
            return -total_performance  # Minimize negative (maximize positive)

        # Initial guess: current thresholds
        x0 = [p.current_threshold for p in patterns]

        # Bounds: min and max threshold for each pattern
        bounds = [(self.config.min_threshold, self.config.max_threshold) for _ in patterns]

        try:
            # Optimize
            result = optimize.minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

            if result.success:
                optimal_thresholds = result.x
                return {
                    "optimal_thresholds": optimal_thresholds.tolist(),
                    "expected_improvement": -result.fun,
                    "optimization_success": True,
                }
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

        return {"optimal_thresholds": x0, "expected_improvement": 0.0, "optimization_success": False}

    def _analyze_threshold_performance_correlation(self, patterns: List[PatternPerformance]) -> Dict:
        """Analyze correlation between thresholds and performance"""
        if len(patterns) < 3:
            return {"error": "Insufficient data for correlation analysis"}

        thresholds = [p.current_threshold for p in patterns]
        success_rates = [p.metrics.success_rate for p in patterns]
        confidences = [p.metrics.average_confidence for p in patterns]

        # Calculate correlations
        threshold_success_corr = np.corrcoef(thresholds, success_rates)[0, 1]
        threshold_confidence_corr = np.corrcoef(thresholds, confidences)[0, 1]

        return {
            "threshold_success_correlation": float(threshold_success_corr),
            "threshold_confidence_correlation": float(threshold_confidence_corr),
            "analysis": {
                "strong_positive_correlation": threshold_success_corr > 0.7,
                "strong_negative_correlation": threshold_success_corr < -0.7,
                "weak_correlation": abs(threshold_success_corr) < 0.3,
            },
        }

    def _generate_distribution_recommendations(self, patterns: List[PatternPerformance]) -> List[str]:
        """Generate recommendations based on threshold distribution analysis"""
        recommendations = []

        thresholds = [p.current_threshold for p in patterns]
        threshold_variance = np.var(thresholds)

        if threshold_variance > 0.1:
            recommendations.append("High threshold variance detected - consider standardizing thresholds")

        low_performers = [p for p in patterns if p.metrics.success_rate < 0.7]
        if len(low_performers) > len(patterns) * 0.3:
            recommendations.append("30%+ patterns underperforming - consider global threshold reduction")

        high_confidence_patterns = [p for p in patterns if p.metrics.average_confidence > 0.9]
        if len(high_confidence_patterns) > len(patterns) * 0.7:
            recommendations.append("High confidence across patterns - consider reducing thresholds for better recall")

        return recommendations

    def _calculate_performance_gradient(self, pattern_perf: PatternPerformance) -> float:
        """Calculate performance gradient for gradient descent optimization"""
        if len(pattern_perf.historical_data) < 2:
            return 0.0

        # Use recent data points to estimate gradient
        recent_data = pattern_perf.historical_data[-5:]  # Last 5 points

        if len(recent_data) < 2:
            return 0.0

        # Simple gradient calculation
        x = [point["threshold"] for point in recent_data]
        y = [point["success_rate"] for point in recent_data]

        # Linear regression gradient
        n = len(x)
        gradient = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (
            n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2
        )

        return gradient

    def _validate_threshold_improvement(self, pattern_perf: PatternPerformance, new_threshold: float) -> bool:
        """Validate that new threshold will likely improve performance"""
        expected_improvement = self._estimate_improvement(pattern_perf, new_threshold)
        return expected_improvement > 0.01  # At least 1% improvement

    def _estimate_success_rate(self, pattern_perf: PatternPerformance, threshold: float) -> float:
        """Estimate success rate at given threshold"""
        current_rate = pattern_perf.metrics.success_rate
        current_threshold = pattern_perf.current_threshold

        # Simple linear model (in practice, use more sophisticated models)
        threshold_diff = threshold - current_threshold

        # Assume increasing threshold generally improves precision
        if threshold_diff > 0:
            improvement = min(0.2, threshold_diff * 0.5)  # Cap improvement
            return min(1.0, current_rate + improvement)
        else:
            decline = min(0.1, abs(threshold_diff) * 0.3)  # Less severe decline
            return max(0.0, current_rate - decline)

    def _estimate_confidence(self, pattern_perf: PatternPerformance, threshold: float) -> float:
        """Estimate confidence at given threshold"""
        current_conf = pattern_perf.metrics.average_confidence
        current_threshold = pattern_perf.current_threshold

        # Higher threshold generally means higher confidence
        threshold_diff = threshold - current_threshold
        conf_change = threshold_diff * 0.3

        return max(0.0, min(1.0, current_conf + conf_change))

    def _estimate_processing_time(self, pattern_perf: PatternPerformance, threshold: float) -> float:
        """Estimate processing time at given threshold"""
        current_time = pattern_perf.metrics.processing_time_avg
        # Assume threshold doesn't significantly affect processing time
        return current_time

    def _estimate_improvement(self, pattern_perf: PatternPerformance, new_threshold: float) -> float:
        """Estimate overall performance improvement"""
        new_success_rate = self._estimate_success_rate(pattern_perf, new_threshold)
        current_success_rate = pattern_perf.metrics.success_rate

        return new_success_rate - current_success_rate

    def _filter_pareto_optimal(self, solutions: List[Dict]) -> List[Dict]:
        """Filter Pareto-optimal solutions from candidate set"""
        pareto_optimal = []

        for i, sol1 in enumerate(solutions):
            is_dominated = False

            for j, sol2 in enumerate(solutions):
                if i != j:
                    # Check if sol2 dominates sol1
                    if (
                        sol2["success_rate"] >= sol1["success_rate"]
                        and sol2["confidence"] >= sol1["confidence"]
                        and sol2["processing_time"] <= sol1["processing_time"]
                        and (
                            sol2["success_rate"] > sol1["success_rate"]
                            or sol2["confidence"] > sol1["confidence"]
                            or sol2["processing_time"] < sol1["processing_time"]
                        )
                    ):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_optimal.append(sol1)

        return pareto_optimal

    def _select_best_pareto_solution(self, pareto_solutions: List[Dict], pattern_perf: PatternPerformance) -> Dict:
        """Select best solution from Pareto-optimal set based on weights"""
        if not pareto_solutions:
            return {"threshold": pattern_perf.current_threshold}

        # Weight preferences
        success_weight = 0.5
        confidence_weight = 0.3
        time_weight = 0.2

        best_score = -float("inf")
        best_solution = pareto_solutions[0]

        for solution in pareto_solutions:
            # Normalize processing time (lower is better)
            normalized_time = 1.0 / (1.0 + solution["processing_time"])

            score = (
                solution["success_rate"] * success_weight
                + solution["confidence"] * confidence_weight
                + normalized_time * time_weight
            )

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def _has_recent_optimization(self, pattern_id: str) -> bool:
        """Check if pattern has recent optimization in cache"""
        if pattern_id not in self.optimization_cache:
            return False

        cached_time = datetime.fromisoformat(self.optimization_cache[pattern_id]["timestamp"])
        time_diff = (datetime.now() - cached_time).total_seconds() / 3600

        return time_diff < 1.0  # Cache valid for 1 hour
