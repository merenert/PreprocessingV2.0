"""
Complete Adaptive Learning System Integration.

Main interface for the full adaptive learning system with automatic threshold
optimization, pattern performance tracking, and intelligent learning.
"""

import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading
from datetime import datetime

from .learning_engine import AdaptiveLearningEngine
from .threshold_optimizer import ThresholdOptimizer
from .pattern_performance_tracker import PatternPerformanceTracker
from .auto_updater import AutoUpdater
from .models import LearningConfig, OptimizationStrategy, OptimizationResult, PatternPerformance, PerformanceMetrics

logger = logging.getLogger(__name__)


class AdaptiveLearningSystem:
    """
    Complete adaptive learning system for automatic threshold optimization.

    Integrates all components:
    - Learning engine for pattern analysis
    - Threshold optimizer for intelligent optimization
    - Performance tracker for real-time monitoring
    - Auto updater for scheduled optimizations
    """

    def __init__(self, config: LearningConfig = None, storage_path: str = "data/adaptive_system"):
        """
        Initialize complete adaptive learning system.

        Args:
            config: Learning configuration
            storage_path: Base storage path for all components
        """
        # Use default config if none provided
        if config is None:
            config = LearningConfig()

        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.learning_engine = AdaptiveLearningEngine(config=config, storage_path=str(self.storage_path / "learning_engine"))

        self.threshold_optimizer = ThresholdOptimizer(config=config)

        self.performance_tracker = PatternPerformanceTracker(
            config=config, storage_path=str(self.storage_path / "performance_tracker")
        )

        self.auto_updater = AutoUpdater(
            learning_engine=self.learning_engine,
            optimizer=self.threshold_optimizer,
            tracker=self.performance_tracker,
            config=config,
        )

        # System state
        self.is_initialized = True
        self.auto_learning_enabled = False

        # Setup callbacks
        self._setup_callbacks()

        logger.info("AdaptiveLearningSystem initialized successfully")

    def enable_auto_learning(self, schedule_type: str = "daily") -> None:
        """
        Enable automatic learning with specified schedule.

        Args:
            schedule_type: Schedule type ('hourly', 'daily', 'weekly', 'custom')
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")

        # Enable performance tracking
        self.performance_tracker.set_tracking_enabled(True)

        # Schedule automatic optimizations
        self.auto_updater.schedule_optimization(schedule_type)

        # Start scheduler
        self.auto_updater.start_scheduler()

        self.auto_learning_enabled = True

        logger.info(f"Auto learning enabled with {schedule_type} schedule")

    def disable_auto_learning(self) -> None:
        """Disable automatic learning"""
        if self.auto_learning_enabled:
            self.auto_updater.stop_scheduler()
            self.performance_tracker.set_tracking_enabled(False)
            self.auto_learning_enabled = False

            logger.info("Auto learning disabled")

    def track_pattern_usage(
        self,
        pattern_id: str,
        pattern_type: str,
        success: bool,
        confidence: float,
        processing_time: float,
        threshold_used: float,
        metadata: Dict = None,
    ) -> None:
        """
        Track pattern usage for learning (main interface).

        Args:
            pattern_id: Unique pattern identifier
            pattern_type: Type of pattern
            success: Whether pattern matching was successful
            confidence: Confidence score of the match
            processing_time: Time taken to process
            threshold_used: Threshold value used
            metadata: Additional metadata
        """
        # Track in both learning engine and performance tracker
        self.learning_engine.record_pattern_usage(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            success=success,
            confidence=confidence,
            processing_time=processing_time,
            threshold_used=threshold_used,
        )

        self.performance_tracker.track_pattern_usage(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            success=success,
            confidence=confidence,
            processing_time=processing_time,
            threshold_used=threshold_used,
            metadata=metadata,
        )

    def get_optimal_threshold(self, pattern_id: str) -> float:
        """
        Get optimal threshold for a pattern.

        Args:
            pattern_id: Pattern identifier

        Returns:
            Optimal threshold value
        """
        if pattern_id in self.learning_engine.pattern_performances:
            pattern_perf = self.learning_engine.pattern_performances[pattern_id]
            return pattern_perf.optimal_threshold
        else:
            # Return default threshold if pattern not found
            return 0.7

    def optimize_pattern_threshold(self, pattern_id: str) -> OptimizationResult:
        """
        Optimize threshold for specific pattern.

        Args:
            pattern_id: Pattern to optimize

        Returns:
            Optimization result
        """
        return self.learning_engine.optimize_threshold(pattern_id)

    def optimize_all_patterns(self) -> List[OptimizationResult]:
        """
        Optimize thresholds for all patterns.

        Returns:
            List of optimization results
        """
        return self.learning_engine.batch_optimize_all_patterns()

    def get_system_performance_report(self) -> Dict:
        """
        Get comprehensive system performance report.

        Returns:
            Complete performance report
        """
        # Get reports from all components
        learning_report = self.learning_engine.get_system_performance_report()
        tracker_metrics = self.performance_tracker.get_real_time_metrics()
        scheduler_status = self.auto_updater.get_scheduler_status()

        # Get performance trends
        trends = self.performance_tracker.get_performance_trends(lookback_hours=24)

        # Combine into comprehensive report
        comprehensive_report = {
            "system_overview": {
                "auto_learning_enabled": self.auto_learning_enabled,
                "total_patterns": len(self.learning_engine.pattern_performances),
                "tracking_enabled": self.performance_tracker.tracking_enabled,
                "scheduler_running": scheduler_status["is_running"],
                "report_timestamp": datetime.now().isoformat(),
            },
            "learning_engine": learning_report,
            "performance_tracker": {
                "real_time_metrics": tracker_metrics.get("system_summary", {}),
                "alerts": self.performance_tracker.get_performance_alerts(),
                "trends": trends,
            },
            "auto_updater": scheduler_status,
            "optimization_recommendations": self._generate_optimization_recommendations(),
        }

        return comprehensive_report

    def get_pattern_analysis(self, pattern_id: str) -> Dict:
        """
        Get detailed analysis for specific pattern.

        Args:
            pattern_id: Pattern to analyze

        Returns:
            Detailed pattern analysis
        """
        # Get analysis from learning engine
        learning_analysis = self.learning_engine.analyze_pattern_performance(pattern_id)

        # Get real-time metrics from tracker
        tracker_metrics = self.performance_tracker.get_real_time_metrics(pattern_id)

        # Get optimization recommendations
        optimization_result = (
            self.threshold_optimizer.optimize_single_pattern(self.learning_engine.pattern_performances.get(pattern_id))
            if pattern_id in self.learning_engine.pattern_performances
            else None
        )

        return {
            "pattern_id": pattern_id,
            "learning_analysis": learning_analysis,
            "real_time_metrics": tracker_metrics,
            "optimization_recommendation": optimization_result.to_dict() if optimization_result else None,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def configure_learning_strategy(self, strategy: OptimizationStrategy, custom_params: Dict = None) -> None:
        """
        Configure learning strategy.

        Args:
            strategy: Optimization strategy
            custom_params: Custom parameters for strategy
        """
        # Update config
        self.config.optimization_strategy = strategy

        if custom_params:
            for param, value in custom_params.items():
                if hasattr(self.config, param):
                    setattr(self.config, param, value)

        # Update components with new config
        self.learning_engine.config = self.config
        self.threshold_optimizer.config = self.config
        self.performance_tracker.config = self.config

        logger.info(f"Learning strategy configured: {strategy.value}")

    def add_performance_alert_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Add callback for performance alerts.

        Args:
            callback: Function to call when alert is triggered
        """
        self.performance_tracker.add_alert_callback(callback)

    def add_optimization_callback(self, callback: Callable[[List[OptimizationResult]], None]) -> None:
        """
        Add callback for optimization completion.

        Args:
            callback: Function to call after optimization
        """
        self.auto_updater.add_optimization_callback(callback)

    def export_learning_data(self, format: str = "json", days: int = 30) -> str:
        """
        Export learning data for analysis.

        Args:
            format: Export format ('json', 'csv')
            days: Number of days to include

        Returns:
            Exported data
        """
        # Export from performance tracker (includes events)
        from datetime import timedelta

        tracker_data = self.performance_tracker.export_performance_data(
            format=format, start_time=datetime.now() - timedelta(days=days)
        )

        # Export optimization history
        optimization_history = self.auto_updater.export_optimization_history(days=days)

        if format.lower() == "json":
            import json

            combined_data = {
                "performance_data": json.loads(tracker_data),
                "optimization_history": optimization_history,
                "export_info": {"format": format, "days": days, "export_timestamp": datetime.now().isoformat()},
            }
            return json.dumps(combined_data, indent=2, ensure_ascii=False)
        else:
            return tracker_data  # CSV format

    def reset_learning_data(self, pattern_id: str = None) -> None:
        """
        Reset learning data for specific pattern or all patterns.

        Args:
            pattern_id: Pattern to reset, or None for all patterns
        """
        # Reset in all components
        (
            self.learning_engine.pattern_performances.clear()
            if pattern_id is None
            else self.learning_engine.pattern_performances.pop(pattern_id, None)
        )
        self.performance_tracker.reset_tracking_data(pattern_id)

        # Clear optimization cache
        if hasattr(self.threshold_optimizer, "optimization_cache"):
            if pattern_id:
                self.threshold_optimizer.optimization_cache.pop(pattern_id, None)
            else:
                self.threshold_optimizer.optimization_cache.clear()

        logger.info(f"Learning data reset for {'all patterns' if pattern_id is None else pattern_id}")

    def force_optimization_run(self, pattern_ids: List[str] = None) -> List[OptimizationResult]:
        """
        Force immediate optimization run.

        Args:
            pattern_ids: Specific patterns to optimize, or None for all

        Returns:
            Optimization results
        """
        return self.auto_updater.run_immediate_optimization(pattern_ids=pattern_ids, background=False)

    def get_threshold_recommendations(self) -> Dict[str, float]:
        """
        Get threshold recommendations for all patterns.

        Returns:
            Dictionary mapping pattern_id to recommended threshold
        """
        recommendations = {}

        for pattern_id, pattern_perf in self.learning_engine.pattern_performances.items():
            if pattern_perf.metrics.total_processed >= self.config.min_samples_for_optimization:
                optimal_threshold = self.learning_engine._calculate_optimal_threshold(pattern_perf)
                recommendations[pattern_id] = optimal_threshold

        return recommendations

    def _setup_callbacks(self) -> None:
        """Setup internal callbacks between components"""

        def alert_handler(alert: Dict) -> None:
            """Handle performance alerts"""
            logger.warning(f"Performance alert: {alert['message']}")

            # Trigger immediate optimization for critical alerts
            if alert["severity"] == "high":
                pattern_id = alert["pattern_id"]
                if pattern_id in self.learning_engine.pattern_performances:
                    logger.info(f"Triggering immediate optimization for {pattern_id} due to critical alert")
                    self.auto_updater.run_immediate_optimization([pattern_id], background=True)

        def optimization_handler(results: List[OptimizationResult]) -> None:
            """Handle optimization completion"""
            applied_count = sum(1 for r in results if r.optimization_applied)
            logger.info(f"Optimization completed: {applied_count}/{len(results)} patterns optimized")

        # Register callbacks
        self.performance_tracker.add_alert_callback(alert_handler)
        self.auto_updater.add_optimization_callback(optimization_handler)

    def _generate_optimization_recommendations(self) -> List[Dict]:
        """Generate optimization recommendations based on current state"""
        recommendations = []

        for pattern_id, pattern_perf in self.learning_engine.pattern_performances.items():
            metrics = pattern_perf.metrics

            # Low performance recommendation
            if metrics.success_rate < 0.7 and metrics.total_processed > 50:
                recommendations.append(
                    {
                        "type": "threshold_increase",
                        "pattern_id": pattern_id,
                        "current_threshold": pattern_perf.current_threshold,
                        "recommended_threshold": min(0.9, pattern_perf.current_threshold + 0.1),
                        "reason": f"Low success rate: {metrics.success_rate:.1%}",
                        "priority": "high",
                    }
                )

            # High performance, potentially too conservative
            elif metrics.success_rate > 0.95 and metrics.average_confidence > 0.9:
                recommendations.append(
                    {
                        "type": "threshold_decrease",
                        "pattern_id": pattern_id,
                        "current_threshold": pattern_perf.current_threshold,
                        "recommended_threshold": max(0.3, pattern_perf.current_threshold - 0.05),
                        "reason": "Very high performance, consider more aggressive threshold",
                        "priority": "medium",
                    }
                )

            # Volatile performance
            elif pattern_perf.trend.value == "volatile":
                recommendations.append(
                    {
                        "type": "stabilization",
                        "pattern_id": pattern_id,
                        "current_threshold": pattern_perf.current_threshold,
                        "recommended_threshold": pattern_perf.current_threshold + 0.05,
                        "reason": "Volatile performance detected",
                        "priority": "medium",
                    }
                )

        return recommendations

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        if self.auto_learning_enabled:
            self.disable_auto_learning()


# Factory function for easy system creation
def create_adaptive_learning_system(
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    auto_start: bool = False,
    storage_path: str = "data/adaptive_system",
) -> AdaptiveLearningSystem:
    """
    Factory function to create and configure adaptive learning system.

    Args:
        strategy: Optimization strategy to use
        auto_start: Whether to automatically start the learning system
        storage_path: Storage path for system data

    Returns:
        Configured adaptive learning system
    """
    config = LearningConfig(optimization_strategy=strategy)
    system = AdaptiveLearningSystem(config=config, storage_path=storage_path)

    if auto_start:
        system.enable_auto_learning()

    return system
