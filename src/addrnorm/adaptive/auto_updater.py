"""
Auto Updater - Scheduled automatic threshold updates and optimization.

Provides automated scheduling for threshold optimizations, background
processing, and system maintenance tasks.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import schedule
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

from .learning_engine import AdaptiveLearningEngine
from .threshold_optimizer import ThresholdOptimizer
from .pattern_performance_tracker import PatternPerformanceTracker
from .models import LearningConfig, OptimizationResult, LearningState

logger = logging.getLogger(__name__)


class AutoUpdater:
    """
    Automated threshold updater with scheduled optimization runs,
    background processing, and system maintenance.
    """

    def __init__(
        self,
        learning_engine: AdaptiveLearningEngine,
        optimizer: ThresholdOptimizer,
        tracker: PatternPerformanceTracker,
        config: LearningConfig,
    ):
        """
        Initialize auto updater.

        Args:
            learning_engine: Adaptive learning engine
            optimizer: Threshold optimizer
            tracker: Performance tracker
            config: Learning configuration
        """
        self.learning_engine = learning_engine
        self.optimizer = optimizer
        self.tracker = tracker
        self.config = config

        # Scheduling state
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Optimization schedules
        self.optimization_schedules = {
            "hourly": {"enabled": False, "last_run": None},
            "daily": {"enabled": True, "last_run": None},
            "weekly": {"enabled": True, "last_run": None},
            "custom": {"enabled": False, "last_run": None, "interval_hours": 6},
        }

        # Callbacks for optimization events
        self.optimization_callbacks: List[Callable] = []
        self.maintenance_callbacks: List[Callable] = []

        # Statistics
        self.auto_optimization_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_patterns_optimized": 0,
            "average_improvement": 0.0,
            "last_run_time": None,
            "next_scheduled_run": None,
        }

        # Configuration
        self.storage_path = Path("data/auto_updater")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("AutoUpdater initialized")

    def start_scheduler(self) -> None:
        """Start the automatic optimization scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        self.is_running = True

        # Setup schedules
        self._setup_schedules()

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info("Auto updater scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the automatic optimization scheduler"""
        if not self.is_running:
            logger.warning("Scheduler not running")
            return

        self.is_running = False

        # Clear schedule
        schedule.clear()

        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Auto updater scheduler stopped")

    def schedule_optimization(self, schedule_type: str = "daily", custom_interval_hours: int = None) -> None:
        """
        Schedule automatic optimizations.

        Args:
            schedule_type: Type of schedule ('hourly', 'daily', 'weekly', 'custom')
            custom_interval_hours: Interval for custom schedule
        """
        if schedule_type not in self.optimization_schedules:
            raise ValueError(f"Invalid schedule type: {schedule_type}")

        # Update schedule configuration
        self.optimization_schedules[schedule_type]["enabled"] = True

        if schedule_type == "custom" and custom_interval_hours:
            self.optimization_schedules[schedule_type]["interval_hours"] = custom_interval_hours

        # Re-setup schedules if running
        if self.is_running:
            self._setup_schedules()

        logger.info(f"Scheduled {schedule_type} optimizations")

    def run_immediate_optimization(
        self, pattern_ids: List[str] = None, background: bool = True
    ) -> Optional[List[OptimizationResult]]:
        """
        Run immediate optimization for specified patterns or all patterns.

        Args:
            pattern_ids: Specific pattern IDs to optimize, or None for all
            background: Run in background thread

        Returns:
            Optimization results (None if running in background)
        """
        if background:
            future = self.executor.submit(self._run_optimization_task, pattern_ids)
            logger.info("Started immediate optimization in background")
            return None
        else:
            return self._run_optimization_task(pattern_ids)

    def add_optimization_callback(self, callback: Callable[[List[OptimizationResult]], None]) -> None:
        """
        Add callback for optimization completion.

        Args:
            callback: Function to call after optimization
        """
        self.optimization_callbacks.append(callback)
        logger.info(f"Added optimization callback: {callback.__name__}")

    def add_maintenance_callback(self, callback: Callable[[], None]) -> None:
        """
        Add callback for maintenance tasks.

        Args:
            callback: Function to call during maintenance
        """
        self.maintenance_callbacks.append(callback)
        logger.info(f"Added maintenance callback: {callback.__name__}")

    def get_scheduler_status(self) -> Dict:
        """
        Get current scheduler status and statistics.

        Returns:
            Scheduler status
        """
        next_runs = {}
        for schedule_type, config in self.optimization_schedules.items():
            if config["enabled"]:
                if schedule_type == "hourly":
                    next_runs[schedule_type] = "Next hour"
                elif schedule_type == "daily":
                    next_runs[schedule_type] = "Tomorrow at midnight"
                elif schedule_type == "weekly":
                    next_runs[schedule_type] = "Next Sunday at midnight"
                elif schedule_type == "custom":
                    interval = config.get("interval_hours", 6)
                    if config["last_run"]:
                        next_run = datetime.fromisoformat(config["last_run"]) + timedelta(hours=interval)
                        next_runs[schedule_type] = next_run.isoformat()
                    else:
                        next_runs[schedule_type] = "Within custom interval"

        return {
            "is_running": self.is_running,
            "enabled_schedules": [k for k, v in self.optimization_schedules.items() if v["enabled"]],
            "next_scheduled_runs": next_runs,
            "statistics": self.auto_optimization_stats.copy(),
            "active_threads": threading.active_count(),
            "executor_active": not self.executor._shutdown,
        }

    def configure_adaptive_optimization(
        self, enable_adaptive: bool = True, performance_threshold: float = 0.8, adaptation_sensitivity: float = 0.5
    ) -> None:
        """
        Configure adaptive optimization based on system performance.

        Args:
            enable_adaptive: Enable adaptive optimization frequency
            performance_threshold: Performance threshold for adaptation
            adaptation_sensitivity: Sensitivity to performance changes
        """
        self.adaptive_config = {
            "enabled": enable_adaptive,
            "performance_threshold": performance_threshold,
            "adaptation_sensitivity": adaptation_sensitivity,
            "last_adaptation": None,
        }

        logger.info(f"Configured adaptive optimization: enabled={enable_adaptive}")

    def force_maintenance_run(self) -> Dict:
        """
        Force immediate maintenance run.

        Returns:
            Maintenance run results
        """
        logger.info("Starting forced maintenance run")

        maintenance_results = {"timestamp": datetime.now().isoformat(), "tasks_completed": [], "errors": []}

        try:
            # Data cleanup
            self._cleanup_old_data()
            maintenance_results["tasks_completed"].append("data_cleanup")

            # Performance analysis
            analysis_results = self._run_performance_analysis()
            maintenance_results["performance_analysis"] = analysis_results
            maintenance_results["tasks_completed"].append("performance_analysis")

            # System health check
            health_check = self._run_system_health_check()
            maintenance_results["health_check"] = health_check
            maintenance_results["tasks_completed"].append("health_check")

            # Call maintenance callbacks
            for callback in self.maintenance_callbacks:
                try:
                    callback()
                    maintenance_results["tasks_completed"].append(f"callback_{callback.__name__}")
                except Exception as e:
                    error_msg = f"Maintenance callback {callback.__name__} failed: {e}"
                    maintenance_results["errors"].append(error_msg)
                    logger.error(error_msg)

        except Exception as e:
            error_msg = f"Maintenance run failed: {e}"
            maintenance_results["errors"].append(error_msg)
            logger.error(error_msg)

        logger.info(f"Maintenance run completed with {len(maintenance_results['tasks_completed'])} tasks")
        return maintenance_results

    def export_optimization_history(self, days: int = 30) -> Dict:
        """
        Export optimization history for analysis.

        Args:
            days: Number of days to include

        Returns:
            Optimization history
        """
        # This would typically load from persistent storage
        # For now, return current statistics
        return {
            "export_period_days": days,
            "statistics": self.auto_optimization_stats.copy(),
            "schedules": self.optimization_schedules.copy(),
            "export_timestamp": datetime.now().isoformat(),
        }

    def _setup_schedules(self) -> None:
        """Setup optimization schedules"""
        # Clear existing schedules
        schedule.clear()

        # Hourly optimization
        if self.optimization_schedules["hourly"]["enabled"]:
            schedule.every().hour.at(":00").do(self._scheduled_optimization, "hourly")

        # Daily optimization
        if self.optimization_schedules["daily"]["enabled"]:
            schedule.every().day.at("02:00").do(self._scheduled_optimization, "daily")

        # Weekly optimization
        if self.optimization_schedules["weekly"]["enabled"]:
            schedule.every().sunday.at("03:00").do(self._scheduled_optimization, "weekly")

        # Custom optimization
        if self.optimization_schedules["custom"]["enabled"]:
            interval = self.optimization_schedules["custom"].get("interval_hours", 6)
            schedule.every(interval).hours.do(self._scheduled_optimization, "custom")

        # Maintenance tasks
        schedule.every().day.at("01:00").do(self._scheduled_maintenance)

        logger.info("Optimization schedules configured")

    def _run_scheduler(self) -> None:
        """Main scheduler loop"""
        logger.info("Scheduler thread started")

        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)

        logger.info("Scheduler thread stopped")

    def _scheduled_optimization(self, schedule_type: str) -> None:
        """Run scheduled optimization"""
        logger.info(f"Running scheduled {schedule_type} optimization")

        # Update last run time
        self.optimization_schedules[schedule_type]["last_run"] = datetime.now().isoformat()

        # Run optimization in background
        future = self.executor.submit(self._run_optimization_task, None, schedule_type)

        # Update next run time
        self.auto_optimization_stats["next_scheduled_run"] = self._calculate_next_run_time()

    def _scheduled_maintenance(self) -> None:
        """Run scheduled maintenance"""
        logger.info("Running scheduled maintenance")

        # Run maintenance in background
        future = self.executor.submit(self.force_maintenance_run)

    def _run_optimization_task(self, pattern_ids: List[str] = None, trigger: str = "manual") -> List[OptimizationResult]:
        """Run optimization task"""
        start_time = datetime.now()

        try:
            logger.info(f"Starting optimization task (trigger: {trigger})")

            # Update stats
            self.auto_optimization_stats["total_runs"] += 1
            self.auto_optimization_stats["last_run_time"] = start_time.isoformat()

            # Run optimization
            if pattern_ids:
                # Optimize specific patterns
                results = []
                for pattern_id in pattern_ids:
                    if pattern_id in self.learning_engine.pattern_performances:
                        result = self.learning_engine.optimize_threshold(pattern_id)
                        results.append(result)
            else:
                # Optimize all patterns
                results = self.learning_engine.batch_optimize_all_patterns()

            # Update statistics
            successful_optimizations = sum(1 for r in results if r.optimization_applied)
            self.auto_optimization_stats["successful_runs"] += 1
            self.auto_optimization_stats["total_patterns_optimized"] += successful_optimizations

            # Calculate average improvement
            improvements = [r.expected_improvement for r in results if r.optimization_applied]
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                current_avg = self.auto_optimization_stats["average_improvement"]
                total_runs = self.auto_optimization_stats["successful_runs"]

                # Update running average
                self.auto_optimization_stats["average_improvement"] = (
                    current_avg * (total_runs - 1) + avg_improvement
                ) / total_runs

            # Call optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(results)
                except Exception as e:
                    logger.error(f"Optimization callback failed: {e}")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Optimization task completed in {duration:.2f}s: "
                f"{successful_optimizations}/{len(results)} patterns optimized"
            )

            return results

        except Exception as e:
            self.auto_optimization_stats["failed_runs"] += 1
            logger.error(f"Optimization task failed: {e}")
            raise

    def _calculate_next_run_time(self) -> Optional[str]:
        """Calculate next scheduled run time"""
        try:
            next_run = schedule.next_run()
            return next_run.isoformat() if next_run else None
        except:
            return None

    def _cleanup_old_data(self) -> None:
        """Cleanup old data from learning engine and tracker"""
        try:
            # Cleanup learning engine data
            cutoff_date = datetime.now() - timedelta(days=30)

            for pattern_id, pattern_perf in self.learning_engine.pattern_performances.items():
                # Remove old historical data
                pattern_perf.historical_data = [
                    point for point in pattern_perf.historical_data if datetime.fromisoformat(point["timestamp"]) > cutoff_date
                ]

            # Remove old threshold updates
            self.learning_engine.threshold_history = [
                update for update in self.learning_engine.threshold_history if update.timestamp > cutoff_date
            ]

            logger.info("Data cleanup completed")

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

    def _run_performance_analysis(self) -> Dict:
        """Run system performance analysis"""
        try:
            # Get system performance report
            report = self.learning_engine.get_system_performance_report()

            # Get tracker metrics
            tracker_metrics = self.tracker.get_real_time_metrics()

            # Analyze trends
            trends = self.tracker.get_performance_trends(lookback_hours=24)

            analysis = {
                "system_report": report,
                "tracker_metrics": tracker_metrics.get("system_summary", {}),
                "trends": trends.get("system_trend", {}),
                "timestamp": datetime.now().isoformat(),
            }

            return analysis

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}

    def _run_system_health_check(self) -> Dict:
        """Run system health check"""
        health_status = {"timestamp": datetime.now().isoformat(), "overall_health": "healthy", "checks": {}}

        try:
            # Check learning engine status
            pattern_count = len(self.learning_engine.pattern_performances)
            health_status["checks"]["learning_engine"] = {
                "status": "healthy" if pattern_count > 0 else "warning",
                "pattern_count": pattern_count,
                "message": f"Tracking {pattern_count} patterns",
            }

            # Check tracker status
            tracker_status = self.tracker.get_real_time_metrics()
            health_status["checks"]["performance_tracker"] = {
                "status": "healthy" if self.tracker.tracking_enabled else "warning",
                "tracking_enabled": self.tracker.tracking_enabled,
                "total_patterns": tracker_status.get("total_patterns", 0),
            }

            # Check scheduler status
            health_status["checks"]["scheduler"] = {
                "status": "healthy" if self.is_running else "error",
                "is_running": self.is_running,
                "active_schedules": len([s for s in self.optimization_schedules.values() if s["enabled"]]),
            }

            # Determine overall health
            check_statuses = [check["status"] for check in health_status["checks"].values()]
            if "error" in check_statuses:
                health_status["overall_health"] = "error"
            elif "warning" in check_statuses:
                health_status["overall_health"] = "warning"

        except Exception as e:
            health_status["overall_health"] = "error"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        return health_status
