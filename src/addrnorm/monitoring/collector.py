"""
Metrics collector for monitoring system
"""

from typing import Dict, Any, List, Optional
import time
from collections import defaultdict, deque


class MetricsCollector:
    """Collects and manages system metrics"""

    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector

        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics = defaultdict(deque)
        self.counters = defaultdict(int)
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        timestamp = time.time()
        metric_data = {"name": name, "value": value, "timestamp": timestamp, "tags": tags or {}}

        if len(self.metrics[name]) >= self.max_history:
            self.metrics[name].popleft()

        self.metrics[name].append(metric_data)

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric

        Args:
            name: Counter name
            value: Increment value
        """
        self.counters[name] += value

    def get_metric_history(self, name: str) -> List[Dict[str, Any]]:
        """Get metric history

        Args:
            name: Metric name

        Returns:
            List of metric values
        """
        return list(self.metrics.get(name, []))

    def get_counter_value(self, name: str) -> int:
        """Get counter value

        Args:
            name: Counter name

        Returns:
            Counter value
        """
        return self.counters.get(name, 0)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary

        Returns:
            Summary of all metrics
        """
        summary = {
            "uptime": time.time() - self.start_time,
            "metrics_count": len(self.metrics),
            "counters": dict(self.counters),
            "recent_metrics": {},
        }

        # Get latest value for each metric
        for name, history in self.metrics.items():
            if history:
                summary["recent_metrics"][name] = history[-1]["value"]

        return summary
