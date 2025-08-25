"""
Pattern Performance Tracker - Real-time performance monitoring and tracking.

Provides real-time tracking of pattern performance metrics, success rates,
and confidence scores with automatic data collection and analysis.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque, defaultdict
import json
from pathlib import Path
from dataclasses import asdict

from .models import PatternPerformance, PerformanceMetrics, PerformanceTrend, LearningConfig

logger = logging.getLogger(__name__)


class PatternPerformanceTracker:
    """
    Real-time pattern performance tracker with automatic data collection,
    analysis, and reporting capabilities.
    """

    def __init__(self, config: LearningConfig, storage_path: str = "data/performance_tracking"):
        """
        Initialize pattern performance tracker.

        Args:
            config: Learning configuration
            storage_path: Path to store tracking data
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Performance tracking data
        self.pattern_metrics: Dict[str, PerformanceMetrics] = {}
        self.recent_events: deque = deque(maxlen=10000)  # Last 10k events
        self.performance_alerts: List[Dict] = []

        # Real-time statistics
        self.hourly_stats: Dict[str, Dict] = defaultdict(dict)
        self.daily_stats: Dict[str, Dict] = defaultdict(dict)

        # Tracking state
        self.tracking_enabled = True
        self.last_cleanup = datetime.now()
        self.alert_callbacks: List[Callable] = []

        # Thread-safe operations
        self._lock = threading.Lock()

        # Load existing data
        self._load_tracking_data()

        logger.info("PatternPerformanceTracker initialized")

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
        Track pattern usage in real-time.

        Args:
            pattern_id: Unique pattern identifier
            pattern_type: Type of pattern
            success: Whether pattern matching was successful
            confidence: Confidence score of the match
            processing_time: Time taken to process
            threshold_used: Threshold value used
            metadata: Additional metadata
        """
        if not self.tracking_enabled:
            return

        timestamp = datetime.now()

        with self._lock:
            # Update or create pattern metrics
            if pattern_id not in self.pattern_metrics:
                self.pattern_metrics[pattern_id] = PerformanceMetrics(
                    pattern_id=pattern_id,
                    success_rate=0.0,
                    total_processed=0,
                    successful_matches=0,
                    failed_matches=0,
                    average_confidence=0.0,
                    confidence_variance=0.0,
                    processing_time_avg=0.0,
                    last_updated=timestamp,
                )

            # Update metrics
            metrics = self.pattern_metrics[pattern_id]
            metrics.update_metrics(success, confidence, processing_time)

            # Record event
            event = {
                "timestamp": timestamp.isoformat(),
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "success": success,
                "confidence": confidence,
                "processing_time": processing_time,
                "threshold_used": threshold_used,
                "metadata": metadata or {},
            }
            self.recent_events.append(event)

            # Update hourly/daily statistics
            self._update_time_based_stats(pattern_id, pattern_type, success, timestamp)

            # Check for performance alerts
            self._check_performance_alerts(pattern_id, metrics)

        # Periodic cleanup
        if (timestamp - self.last_cleanup).total_seconds() > 3600:  # Every hour
            self._cleanup_old_data()
            self.last_cleanup = timestamp

        logger.debug(f"Tracked usage: {pattern_id} - success={success}, confidence={confidence:.3f}")

    def get_real_time_metrics(self, pattern_id: str = None) -> Dict:
        """
        Get real-time performance metrics.

        Args:
            pattern_id: Specific pattern ID, or None for all patterns

        Returns:
            Real-time metrics
        """
        with self._lock:
            if pattern_id:
                if pattern_id not in self.pattern_metrics:
                    return {"error": f"Pattern {pattern_id} not found"}

                metrics = self.pattern_metrics[pattern_id]
                return {
                    "pattern_id": pattern_id,
                    "metrics": asdict(metrics),
                    "recent_trend": self._calculate_recent_trend(pattern_id),
                    "hourly_performance": self.hourly_stats.get(pattern_id, {}),
                    "daily_performance": self.daily_stats.get(pattern_id, {}),
                }
            else:
                # Return summary for all patterns
                return {
                    "total_patterns": len(self.pattern_metrics),
                    "patterns": {pid: asdict(metrics) for pid, metrics in self.pattern_metrics.items()},
                    "system_summary": self._get_system_summary(),
                    "recent_alerts": self.performance_alerts[-10:],  # Last 10 alerts
                }

    def get_performance_trends(self, lookback_hours: int = 24) -> Dict:
        """
        Get performance trends over specified time period.

        Args:
            lookback_hours: Hours to look back for trend analysis

        Returns:
            Performance trends
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        with self._lock:
            # Filter recent events
            recent_events = [event for event in self.recent_events if datetime.fromisoformat(event["timestamp"]) > cutoff_time]

        if not recent_events:
            return {"error": "No recent events found"}

        # Group by pattern
        pattern_trends = defaultdict(list)
        for event in recent_events:
            pattern_trends[event["pattern_id"]].append(event)

        # Calculate trends
        trends = {}
        for pattern_id, events in pattern_trends.items():
            trends[pattern_id] = self._analyze_pattern_trend(events)

        # System-wide trends
        system_trend = self._analyze_system_trend(recent_events)

        return {
            "lookback_hours": lookback_hours,
            "pattern_trends": trends,
            "system_trend": system_trend,
            "total_events": len(recent_events),
        }

    def get_performance_alerts(self, active_only: bool = True) -> List[Dict]:
        """
        Get performance alerts.

        Args:
            active_only: Return only active alerts

        Returns:
            List of performance alerts
        """
        with self._lock:
            if active_only:
                return [alert for alert in self.performance_alerts if alert.get("active", True)]
            else:
                return self.performance_alerts.copy()

    def add_alert_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Add callback function for performance alerts.

        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")

    def set_tracking_enabled(self, enabled: bool) -> None:
        """
        Enable or disable performance tracking.

        Args:
            enabled: Whether to enable tracking
        """
        self.tracking_enabled = enabled
        logger.info(f"Performance tracking {'enabled' if enabled else 'disabled'}")

    def export_performance_data(self, format: str = "json", start_time: datetime = None, end_time: datetime = None) -> str:
        """
        Export performance data in specified format.

        Args:
            format: Export format ('json', 'csv')
            start_time: Start time for export
            end_time: End time for export

        Returns:
            Exported data as string
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)  # Last week
        if end_time is None:
            end_time = datetime.now()

        with self._lock:
            # Filter events by time range
            filtered_events = [
                event for event in self.recent_events if start_time <= datetime.fromisoformat(event["timestamp"]) <= end_time
            ]

            # Include current metrics
            export_data = {
                "export_info": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_events": len(filtered_events),
                    "export_timestamp": datetime.now().isoformat(),
                },
                "current_metrics": {pid: asdict(metrics) for pid, metrics in self.pattern_metrics.items()},
                "events": filtered_events,
                "alerts": self.performance_alerts,
            }

        if format.lower() == "json":
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            return self._export_to_csv(export_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def reset_tracking_data(self, pattern_id: str = None) -> None:
        """
        Reset tracking data for specific pattern or all patterns.

        Args:
            pattern_id: Pattern to reset, or None for all patterns
        """
        with self._lock:
            if pattern_id:
                if pattern_id in self.pattern_metrics:
                    del self.pattern_metrics[pattern_id]
                    # Remove events for this pattern
                    self.recent_events = deque([e for e in self.recent_events if e["pattern_id"] != pattern_id], maxlen=10000)
                    logger.info(f"Reset tracking data for pattern: {pattern_id}")
            else:
                self.pattern_metrics.clear()
                self.recent_events.clear()
                self.performance_alerts.clear()
                self.hourly_stats.clear()
                self.daily_stats.clear()
                logger.info("Reset all tracking data")

        self._save_tracking_data()

    def _update_time_based_stats(self, pattern_id: str, pattern_type: str, success: bool, timestamp: datetime) -> None:
        """Update hourly and daily statistics"""
        hour_key = timestamp.strftime("%Y-%m-%d-%H")
        day_key = timestamp.strftime("%Y-%m-%d")

        # Update hourly stats
        if hour_key not in self.hourly_stats[pattern_id]:
            self.hourly_stats[pattern_id][hour_key] = {"total": 0, "successful": 0, "failed": 0, "success_rate": 0.0}

        hour_stats = self.hourly_stats[pattern_id][hour_key]
        hour_stats["total"] += 1
        if success:
            hour_stats["successful"] += 1
        else:
            hour_stats["failed"] += 1
        hour_stats["success_rate"] = hour_stats["successful"] / hour_stats["total"]

        # Update daily stats
        if day_key not in self.daily_stats[pattern_id]:
            self.daily_stats[pattern_id][day_key] = {"total": 0, "successful": 0, "failed": 0, "success_rate": 0.0}

        day_stats = self.daily_stats[pattern_id][day_key]
        day_stats["total"] += 1
        if success:
            day_stats["successful"] += 1
        else:
            day_stats["failed"] += 1
        day_stats["success_rate"] = day_stats["successful"] / day_stats["total"]

        # Keep only recent stats (last 7 days for hourly, last 30 days for daily)
        cutoff_hour = (timestamp - timedelta(days=7)).strftime("%Y-%m-%d-%H")
        cutoff_day = (timestamp - timedelta(days=30)).strftime("%Y-%m-%d")

        # Cleanup old hourly stats
        old_hours = [k for k in self.hourly_stats[pattern_id].keys() if k < cutoff_hour]
        for hour in old_hours:
            del self.hourly_stats[pattern_id][hour]

        # Cleanup old daily stats
        old_days = [k for k in self.daily_stats[pattern_id].keys() if k < cutoff_day]
        for day in old_days:
            del self.daily_stats[pattern_id][day]

    def _check_performance_alerts(self, pattern_id: str, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts and trigger callbacks"""
        alerts = []

        # Low success rate alert
        if metrics.success_rate < 0.5 and metrics.total_processed > 50:
            alerts.append(
                {
                    "type": "low_success_rate",
                    "pattern_id": pattern_id,
                    "severity": "high",
                    "message": f"Pattern {pattern_id} has low success rate: {metrics.success_rate:.1%}",
                    "value": metrics.success_rate,
                    "threshold": 0.5,
                    "timestamp": datetime.now().isoformat(),
                    "active": True,
                }
            )

        # Low confidence alert
        if metrics.average_confidence < 0.3 and metrics.total_processed > 30:
            alerts.append(
                {
                    "type": "low_confidence",
                    "pattern_id": pattern_id,
                    "severity": "medium",
                    "message": f"Pattern {pattern_id} has low confidence: {metrics.average_confidence:.3f}",
                    "value": metrics.average_confidence,
                    "threshold": 0.3,
                    "timestamp": datetime.now().isoformat(),
                    "active": True,
                }
            )

        # High processing time alert
        if metrics.processing_time_avg > 1000:  # > 1 second
            alerts.append(
                {
                    "type": "high_processing_time",
                    "pattern_id": pattern_id,
                    "severity": "medium",
                    "message": f"Pattern {pattern_id} has high processing time: {metrics.processing_time_avg:.0f}ms",
                    "value": metrics.processing_time_avg,
                    "threshold": 1000,
                    "timestamp": datetime.now().isoformat(),
                    "active": True,
                }
            )

        # Add alerts and trigger callbacks
        for alert in alerts:
            # Check if alert already exists
            existing_alert = next(
                (
                    a
                    for a in self.performance_alerts
                    if a["type"] == alert["type"] and a["pattern_id"] == alert["pattern_id"] and a.get("active", True)
                ),
                None,
            )

            if not existing_alert:
                self.performance_alerts.append(alert)
                logger.warning(f"Performance alert: {alert['message']}")

                # Trigger callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

    def _calculate_recent_trend(self, pattern_id: str, lookback_minutes: int = 60) -> Dict:
        """Calculate recent performance trend for a pattern"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)

        recent_events = [
            event
            for event in self.recent_events
            if (event["pattern_id"] == pattern_id and datetime.fromisoformat(event["timestamp"]) > cutoff_time)
        ]

        if len(recent_events) < 2:
            return {"trend": "insufficient_data", "events": len(recent_events)}

        # Calculate trend
        total_events = len(recent_events)
        successful_events = sum(1 for e in recent_events if e["success"])
        success_rate = successful_events / total_events

        avg_confidence = sum(e["confidence"] for e in recent_events) / total_events
        avg_processing_time = sum(e["processing_time"] for e in recent_events) / total_events

        # Compare with overall metrics
        overall_metrics = self.pattern_metrics[pattern_id]

        trend_indicators = {
            "success_rate_change": success_rate - overall_metrics.success_rate,
            "confidence_change": avg_confidence - overall_metrics.average_confidence,
            "processing_time_change": avg_processing_time - overall_metrics.processing_time_avg,
        }

        # Determine overall trend
        if trend_indicators["success_rate_change"] > 0.05:
            trend = "improving"
        elif trend_indicators["success_rate_change"] < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_success_rate": success_rate,
            "recent_confidence": avg_confidence,
            "recent_processing_time": avg_processing_time,
            "trend_indicators": trend_indicators,
            "events_analyzed": total_events,
            "lookback_minutes": lookback_minutes,
        }

    def _get_system_summary(self) -> Dict:
        """Get system-wide performance summary"""
        if not self.pattern_metrics:
            return {"error": "No metrics available"}

        total_processed = sum(m.total_processed for m in self.pattern_metrics.values())
        total_successful = sum(m.successful_matches for m in self.pattern_metrics.values())
        overall_success_rate = total_successful / total_processed if total_processed > 0 else 0.0

        avg_confidence = sum(m.average_confidence for m in self.pattern_metrics.values()) / len(self.pattern_metrics)
        avg_processing_time = sum(m.processing_time_avg for m in self.pattern_metrics.values()) / len(self.pattern_metrics)

        active_alerts = len([a for a in self.performance_alerts if a.get("active", True)])

        return {
            "total_patterns": len(self.pattern_metrics),
            "total_processed": total_processed,
            "overall_success_rate": overall_success_rate,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "active_alerts": active_alerts,
            "tracking_enabled": self.tracking_enabled,
        }

    def _analyze_pattern_trend(self, events: List[Dict]) -> Dict:
        """Analyze trend for pattern events"""
        if len(events) < 5:
            return {"trend": "insufficient_data", "events": len(events)}

        # Sort by timestamp
        events = sorted(events, key=lambda e: e["timestamp"])

        # Split into first and second half
        n = len(events)
        first_half = events[: n // 2]
        second_half = events[n // 2 :]

        # Calculate metrics for each half
        first_success_rate = sum(1 for e in first_half if e["success"]) / len(first_half)
        second_success_rate = sum(1 for e in second_half if e["success"]) / len(second_half)

        first_confidence = sum(e["confidence"] for e in first_half) / len(first_half)
        second_confidence = sum(e["confidence"] for e in second_half) / len(second_half)

        # Determine trend
        success_change = second_success_rate - first_success_rate
        confidence_change = second_confidence - first_confidence

        if success_change > 0.1:
            trend = "strongly_improving"
        elif success_change > 0.03:
            trend = "improving"
        elif success_change < -0.1:
            trend = "strongly_declining"
        elif success_change < -0.03:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "success_rate_change": success_change,
            "confidence_change": confidence_change,
            "first_half_success_rate": first_success_rate,
            "second_half_success_rate": second_success_rate,
            "events_analyzed": len(events),
        }

    def _analyze_system_trend(self, events: List[Dict]) -> Dict:
        """Analyze system-wide trend"""
        if len(events) < 10:
            return {"trend": "insufficient_data", "events": len(events)}

        # Group by hour
        hourly_data = defaultdict(list)
        for event in events:
            hour = datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d-%H")
            hourly_data[hour].append(event)

        # Calculate hourly success rates
        hourly_success_rates = {}
        for hour, hour_events in hourly_data.items():
            success_rate = sum(1 for e in hour_events if e["success"]) / len(hour_events)
            hourly_success_rates[hour] = success_rate

        # Analyze trend
        sorted_hours = sorted(hourly_success_rates.keys())
        if len(sorted_hours) < 3:
            return {"trend": "insufficient_data", "hours": len(sorted_hours)}

        # Simple linear trend
        rates = [hourly_success_rates[hour] for hour in sorted_hours]
        n = len(rates)

        # Calculate slope
        x = list(range(n))
        slope = (n * sum(x[i] * rates[i] for i in range(n)) - sum(x) * sum(rates)) / (
            n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2
        )

        if slope > 0.01:
            trend = "improving"
        elif slope < -0.01:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": slope,
            "hourly_success_rates": hourly_success_rates,
            "hours_analyzed": len(sorted_hours),
            "total_events": len(events),
        }

    def _export_to_csv(self, data: Dict) -> str:
        """Export data to CSV format"""
        import csv
        from io import StringIO

        output = StringIO()

        # Write events
        if data["events"]:
            writer = csv.DictWriter(output, fieldnames=data["events"][0].keys())
            writer.writeheader()
            writer.writerows(data["events"])

        return output.getvalue()

    def _cleanup_old_data(self) -> None:
        """Cleanup old tracking data"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days

        # Remove old events
        self.recent_events = deque(
            [e for e in self.recent_events if datetime.fromisoformat(e["timestamp"]) > cutoff_time], maxlen=10000
        )

        # Remove old alerts
        self.performance_alerts = [
            alert for alert in self.performance_alerts if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]

        logger.debug("Cleaned up old tracking data")

    def _load_tracking_data(self) -> None:
        """Load tracking data from storage"""
        try:
            metrics_file = self.storage_path / "pattern_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Simplified loading for now
                    logger.info(f"Loaded tracking data for {len(data)} patterns")

        except Exception as e:
            logger.warning(f"Failed to load tracking data: {e}")

    def _save_tracking_data(self) -> None:
        """Save tracking data to storage"""
        try:
            # Save current metrics
            metrics_file = self.storage_path / "pattern_metrics.json"
            metrics_data = {pid: asdict(metrics) for pid, metrics in self.pattern_metrics.items()}

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

            logger.debug("Tracking data saved successfully")

        except Exception as e:
            logger.error(f"Failed to save tracking data: {e}")
