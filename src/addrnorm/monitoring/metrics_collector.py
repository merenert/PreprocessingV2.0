"""
Metrics Collector

Real-time performance tracking and metrics collection for address normalization.
Collects pattern usage, success rates, processing times, and system health metrics.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""

    PERFORMANCE = "performance"
    PATTERN_USAGE = "pattern_usage"
    SUCCESS_RATE = "success_rate"
    CONFIDENCE = "confidence"
    GEOGRAPHIC = "geographic"
    THRESHOLD = "threshold"
    ERROR = "error"


class ProcessingMethod(Enum):
    """Processing methods for tracking"""

    PATTERN_PRIMARY = "pattern_primary"
    PATTERN_SECONDARY = "pattern_secondary"
    ML_PRIMARY = "ml_primary"
    ML_SECONDARY = "ml_secondary"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass
class MetricEvent:
    """Individual metric event"""

    timestamp: datetime
    metric_type: MetricType
    method: ProcessingMethod
    pattern_id: Optional[str] = None
    success: bool = True
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    address_length: int = 0
    components_extracted: int = 0
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "method": self.method.value,
            "pattern_id": self.pattern_id,
            "success": self.success,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "address_length": self.address_length,
            "components_extracted": self.components_extracted,
            "error_type": self.error_type,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics aggregation"""

    avg_processing_time_ms: float = 0.0
    median_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    total_processed: int = 0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "median_processing_time_ms": round(self.median_processing_time_ms, 2),
            "p95_processing_time_ms": round(self.p95_processing_time_ms, 2),
            "p99_processing_time_ms": round(self.p99_processing_time_ms, 2),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "total_processed": self.total_processed,
            "error_rate": round(self.error_rate, 3),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
        }


@dataclass
class PatternMetrics:
    """Pattern-specific metrics"""

    pattern_id: str
    usage_count: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_processing_time_ms: float = 0.0
    common_failures: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "usage_count": self.usage_count,
            "success_rate": round(self.success_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "common_failures": self.common_failures,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


@dataclass
class SystemMetrics:
    """Overall system health metrics"""

    uptime_seconds: float = 0.0
    method_distribution: Dict[str, int] = field(default_factory=dict)
    geographic_validation_rate: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    threshold_adjustments: int = 0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": round(self.uptime_seconds, 2),
            "method_distribution": self.method_distribution,
            "geographic_validation_rate": round(self.geographic_validation_rate, 3),
            "confidence_distribution": self.confidence_distribution,
            "threshold_adjustments": self.threshold_adjustments,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
        }


class MetricsCollector:
    """
    Real-time metrics collector for address normalization system

    Features:
    - Thread-safe event collection
    - Real-time aggregation
    - Configurable retention
    - Automatic persistence
    - Performance monitoring
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        aggregation_interval_seconds: int = 60,
        retention_days: int = 30,
        storage_path: Optional[Path] = None,
        enable_persistence: bool = True,
    ):
        """
        Initialize metrics collector

        Args:
            buffer_size: Maximum events in memory buffer
            aggregation_interval_seconds: How often to aggregate metrics
            retention_days: How long to keep historical data
            storage_path: Path for persistent storage
            enable_persistence: Whether to persist metrics to disk
        """
        self.buffer_size = buffer_size
        self.aggregation_interval = aggregation_interval_seconds
        self.retention_days = retention_days
        self.storage_path = storage_path or Path("monitoring_data")
        self.enable_persistence = enable_persistence

        # Thread-safe event buffer
        self._events = deque(maxlen=buffer_size)
        self._lock = threading.RLock()

        # Aggregated metrics
        self._performance_metrics = PerformanceMetrics()
        self._pattern_metrics: Dict[str, PatternMetrics] = {}
        self._system_metrics = SystemMetrics()

        # Tracking data
        self._start_time = datetime.now()
        self._processing_times = deque(maxlen=1000)  # For percentile calculation
        self._method_counts = defaultdict(int)
        self._confidence_buckets = defaultdict(int)
        self._error_counts = defaultdict(int)

        # Background aggregation
        self._aggregation_thread = None
        self._running = False

        self.logger = logging.getLogger(__name__)

        if enable_persistence:
            self.storage_path.mkdir(exist_ok=True)

        self.start_aggregation()

    def record_event(self, event: MetricEvent) -> None:
        """Record a metric event"""
        with self._lock:
            self._events.append(event)
            self._update_real_time_metrics(event)

    def record_processing(
        self,
        method: ProcessingMethod,
        pattern_id: Optional[str] = None,
        success: bool = True,
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
        address_length: int = 0,
        components_extracted: int = 0,
        error_type: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Convenience method to record processing event"""
        event = MetricEvent(
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
            method=method,
            pattern_id=pattern_id,
            success=success,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            address_length=address_length,
            components_extracted=components_extracted,
            error_type=error_type,
            metadata=metadata or {},
        )
        self.record_event(event)

    def record_pattern_usage(
        self,
        pattern_id: str,
        method: ProcessingMethod,
        success: bool = True,
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
    ) -> None:
        """Record pattern usage event"""
        event = MetricEvent(
            timestamp=datetime.now(),
            metric_type=MetricType.PATTERN_USAGE,
            method=method,
            pattern_id=pattern_id,
            success=success,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )
        self.record_event(event)

    def record_threshold_adjustment(self, threshold_type: str, old_value: float, new_value: float, reason: str) -> None:
        """Record threshold adjustment event"""
        event = MetricEvent(
            timestamp=datetime.now(),
            metric_type=MetricType.THRESHOLD,
            method=ProcessingMethod.HYBRID,  # Generic
            metadata={"threshold_type": threshold_type, "old_value": old_value, "new_value": new_value, "reason": reason},
        )
        self.record_event(event)

    def _update_real_time_metrics(self, event: MetricEvent) -> None:
        """Update real-time metrics with new event"""
        # Update method counts
        self._method_counts[event.method.value] += 1

        # Update processing times
        if event.processing_time_ms > 0:
            self._processing_times.append(event.processing_time_ms)

        # Update confidence buckets
        if event.confidence > 0:
            bucket = self._get_confidence_bucket(event.confidence)
            self._confidence_buckets[bucket] += 1

        # Update error counts
        if not event.success and event.error_type:
            self._error_counts[event.error_type] += 1

        # Update pattern metrics
        if event.pattern_id:
            self._update_pattern_metrics(event)

    def _update_pattern_metrics(self, event: MetricEvent) -> None:
        """Update pattern-specific metrics"""
        pattern_id = event.pattern_id

        if pattern_id not in self._pattern_metrics:
            self._pattern_metrics[pattern_id] = PatternMetrics(pattern_id=pattern_id)

        pattern = self._pattern_metrics[pattern_id]
        pattern.usage_count += 1
        pattern.last_used = event.timestamp

        # Update rolling averages
        if event.confidence > 0:
            current_avg = pattern.avg_confidence
            count = pattern.usage_count
            pattern.avg_confidence = ((current_avg * (count - 1)) + event.confidence) / count

        if event.processing_time_ms > 0:
            current_avg = pattern.avg_processing_time_ms
            count = pattern.usage_count
            pattern.avg_processing_time_ms = ((current_avg * (count - 1)) + event.processing_time_ms) / count

        # Track failures
        if not event.success and event.error_type:
            if event.error_type not in pattern.common_failures:
                pattern.common_failures.append(event.error_type)
                # Keep only top 5 failures
                pattern.common_failures = pattern.common_failures[-5:]

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for distribution"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self._lock:
            total_events = len(self._events)

            if total_events == 0:
                return PerformanceMetrics()

            # Calculate processing time statistics
            processing_times = list(self._processing_times)
            if processing_times:
                processing_times.sort()
                n = len(processing_times)

                avg_time = sum(processing_times) / n
                median_time = processing_times[n // 2]
                p95_time = processing_times[int(n * 0.95)] if n > 0 else 0
                p99_time = processing_times[int(n * 0.99)] if n > 0 else 0
            else:
                avg_time = median_time = p95_time = p99_time = 0

            # Calculate throughput
            uptime = (datetime.now() - self._start_time).total_seconds()
            throughput = total_events / max(uptime, 1)

            # Calculate error rate
            error_events = sum(1 for e in self._events if not e.success)
            error_rate = error_events / total_events if total_events > 0 else 0

            return PerformanceMetrics(
                avg_processing_time_ms=avg_time,
                median_processing_time_ms=median_time,
                p95_processing_time_ms=p95_time,
                p99_processing_time_ms=p99_time,
                throughput_per_second=throughput,
                total_processed=total_events,
                error_rate=error_rate,
                memory_usage_mb=self._get_memory_usage(),
            )

    def get_pattern_metrics(self) -> Dict[str, PatternMetrics]:
        """Get pattern-specific metrics"""
        with self._lock:
            # Calculate success rates
            for pattern_id, pattern in self._pattern_metrics.items():
                pattern_events = [e for e in self._events if e.pattern_id == pattern_id]
                if pattern_events:
                    successful = sum(1 for e in pattern_events if e.success)
                    pattern.success_rate = successful / len(pattern_events)

            return self._pattern_metrics.copy()

    def get_system_metrics(self) -> SystemMetrics:
        """Get overall system metrics"""
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()

            # Geographic validation rate (placeholder - would integrate with actual validation)
            geo_validated = sum(1 for e in self._events if e.metadata.get("geographic_validation", False))
            geo_rate = geo_validated / len(self._events) if self._events else 0

            # Threshold adjustments
            threshold_events = sum(1 for e in self._events if e.metric_type == MetricType.THRESHOLD)

            return SystemMetrics(
                uptime_seconds=uptime,
                method_distribution=dict(self._method_counts),
                geographic_validation_rate=geo_rate,
                confidence_distribution=dict(self._confidence_buckets),
                threshold_adjustments=threshold_events,
                cache_hit_rate=0.8,  # Placeholder - would integrate with actual cache
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def start_aggregation(self) -> None:
        """Start background aggregation thread"""
        if self._aggregation_thread is None or not self._aggregation_thread.is_alive():
            self._running = True
            self._aggregation_thread = threading.Thread(target=self._aggregation_loop)
            self._aggregation_thread.daemon = True
            self._aggregation_thread.start()
            self.logger.info("Metrics aggregation started")

    def stop_aggregation(self) -> None:
        """Stop background aggregation"""
        self._running = False
        if self._aggregation_thread and self._aggregation_thread.is_alive():
            self._aggregation_thread.join(timeout=5)
        self.logger.info("Metrics aggregation stopped")

    def _aggregation_loop(self) -> None:
        """Background aggregation loop"""
        while self._running:
            try:
                time.sleep(self.aggregation_interval)
                if self._running:
                    self._aggregate_metrics()
                    if self.enable_persistence:
                        self._persist_metrics()
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")

    def _aggregate_metrics(self) -> None:
        """Aggregate current metrics"""
        with self._lock:
            self._performance_metrics = self.get_performance_metrics()
            self._system_metrics = self.get_system_metrics()

    def _persist_metrics(self) -> None:
        """Persist metrics to storage"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save aggregated metrics
            metrics_file = self.storage_path / f"metrics_{timestamp}.json"
            metrics_data = {
                "timestamp": timestamp,
                "performance": self._performance_metrics.to_dict(),
                "patterns": {pid: pm.to_dict() for pid, pm in self._pattern_metrics.items()},
                "system": self._system_metrics.to_dict(),
            }

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)

            # Cleanup old files
            self._cleanup_old_files()

        except Exception as e:
            self.logger.error(f"Error persisting metrics: {e}")

    def _cleanup_old_files(self) -> None:
        """Remove old metric files based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for file_path in self.storage_path.glob("metrics_*.json"):
            try:
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_date < cutoff_date:
                    file_path.unlink()
            except Exception:
                pass  # Skip files we can't process

    def export_metrics(self, format: str = "json") -> str:
        """Export current metrics in specified format"""
        data = {
            "performance": self.get_performance_metrics().to_dict(),
            "patterns": {pid: pm.to_dict() for pid, pm in self.get_pattern_metrics().items()},
            "system": self.get_system_metrics().to_dict(),
            "export_timestamp": datetime.now().isoformat(),
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self._events.clear()
            self._pattern_metrics.clear()
            self._processing_times.clear()
            self._method_counts.clear()
            self._confidence_buckets.clear()
            self._error_counts.clear()
            self._start_time = datetime.now()

        self.logger.info("All metrics reset")

    def __del__(self):
        """Cleanup when collector is destroyed"""
        self.stop_aggregation()
