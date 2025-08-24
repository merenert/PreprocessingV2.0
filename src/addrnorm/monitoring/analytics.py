"""
Analytics Engine

Advanced analytics for pattern performance, system health, and trend analysis.
Provides insights into usage patterns, success rates, and optimization opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
from collections import defaultdict
from pathlib import Path
import statistics

from .metrics_collector import MetricsCollector, MetricEvent, MetricType, ProcessingMethod

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction indicators"""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TrendAnalysis:
    """Trend analysis result"""

    metric_name: str
    direction: TrendDirection
    change_percentage: float
    confidence: float
    time_period: str
    current_value: float
    previous_value: float
    analysis_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "change_percentage": round(self.change_percentage, 2),
            "confidence": round(self.confidence, 3),
            "time_period": self.time_period,
            "current_value": round(self.current_value, 3),
            "previous_value": round(self.previous_value, 3),
            "analysis_notes": self.analysis_notes,
        }


@dataclass
class PerformanceInsight:
    """Performance insight"""

    category: str
    insight_type: str
    description: str
    impact_level: AlertLevel
    recommendation: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "insight_type": self.insight_type,
            "description": self.description,
            "impact_level": self.impact_level.value,
            "recommendation": self.recommendation,
            "supporting_data": self.supporting_data,
        }


@dataclass
class PatternAnalysisResult:
    """Pattern performance analysis result"""

    pattern_id: str
    usage_rank: int
    success_rate: float
    avg_confidence: float
    performance_score: float
    optimization_potential: float
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "usage_rank": self.usage_rank,
            "success_rate": round(self.success_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "performance_score": round(self.performance_score, 3),
            "optimization_potential": round(self.optimization_potential, 3),
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }


class TrendAnalyzer:
    """
    Analyzes trends in system metrics over time

    Features:
    - Multi-timeframe trend analysis
    - Statistical significance testing
    - Anomaly detection
    - Seasonal pattern recognition
    """

    def __init__(self, min_data_points: int = 10):
        """
        Initialize trend analyzer

        Args:
            min_data_points: Minimum data points needed for analysis
        """
        self.min_data_points = min_data_points
        self.logger = logging.getLogger(__name__)

    def analyze_performance_trends(self, events: List[MetricEvent], time_windows: List[str] = None) -> List[TrendAnalysis]:
        """
        Analyze performance trends across multiple time windows

        Args:
            events: List of metric events
            time_windows: Time windows to analyze ['1h', '24h', '7d', '30d']

        Returns:
            List of trend analyses
        """
        if time_windows is None:
            time_windows = ["1h", "24h", "7d", "30d"]

        results = []

        for window in time_windows:
            try:
                window_events = self._filter_events_by_window(events, window)
                if len(window_events) < self.min_data_points:
                    continue

                # Analyze different metrics
                trends = [
                    self._analyze_processing_time_trend(window_events, window),
                    self._analyze_success_rate_trend(window_events, window),
                    self._analyze_confidence_trend(window_events, window),
                    self._analyze_throughput_trend(window_events, window),
                ]

                results.extend([t for t in trends if t])

            except Exception as e:
                self.logger.error(f"Error analyzing trends for window {window}: {e}")

        return results

    def _filter_events_by_window(self, events: List[MetricEvent], window: str) -> List[MetricEvent]:
        """Filter events by time window"""
        now = datetime.now()

        if window == "1h":
            cutoff = now - timedelta(hours=1)
        elif window == "24h":
            cutoff = now - timedelta(days=1)
        elif window == "7d":
            cutoff = now - timedelta(days=7)
        elif window == "30d":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(hours=1)

        return [e for e in events if e.timestamp >= cutoff]

    def _analyze_processing_time_trend(self, events: List[MetricEvent], window: str) -> Optional[TrendAnalysis]:
        """Analyze processing time trend"""
        processing_times = [e.processing_time_ms for e in events if e.processing_time_ms > 0]

        if len(processing_times) < self.min_data_points:
            return None

        # Split into first and second half
        mid_point = len(processing_times) // 2
        first_half = processing_times[:mid_point]
        second_half = processing_times[mid_point:]

        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)

        change_pct = ((avg_second - avg_first) / avg_first) * 100 if avg_first > 0 else 0

        # Determine trend direction
        if abs(change_pct) < 5:
            direction = TrendDirection.STABLE
        elif change_pct > 0:
            direction = TrendDirection.DECLINING  # Higher processing time is worse
        else:
            direction = TrendDirection.IMPROVING

        # Calculate confidence based on data consistency
        std_dev = statistics.stdev(processing_times)
        confidence = max(0.1, min(1.0, 1.0 - (std_dev / avg_second if avg_second > 0 else 1.0)))

        notes = []
        if std_dev > avg_second * 0.5:
            notes.append("High variability in processing times detected")
        if max(processing_times) > avg_second * 3:
            notes.append("Outliers detected - possible performance spikes")

        return TrendAnalysis(
            metric_name="processing_time_ms",
            direction=direction,
            change_percentage=change_pct,
            confidence=confidence,
            time_period=window,
            current_value=avg_second,
            previous_value=avg_first,
            analysis_notes=notes,
        )

    def _analyze_success_rate_trend(self, events: List[MetricEvent], window: str) -> Optional[TrendAnalysis]:
        """Analyze success rate trend"""
        if len(events) < self.min_data_points:
            return None

        # Split into first and second half
        mid_point = len(events) // 2
        first_half = events[:mid_point]
        second_half = events[mid_point:]

        success_first = sum(1 for e in first_half if e.success) / len(first_half)
        success_second = sum(1 for e in second_half if e.success) / len(second_half)

        change_pct = ((success_second - success_first) / success_first) * 100 if success_first > 0 else 0

        # Determine trend direction
        if abs(change_pct) < 2:
            direction = TrendDirection.STABLE
        elif change_pct > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING

        confidence = 0.8  # High confidence for binary metric

        notes = []
        if success_second < 0.9:
            notes.append("Success rate below 90% - investigation recommended")

        return TrendAnalysis(
            metric_name="success_rate",
            direction=direction,
            change_percentage=change_pct,
            confidence=confidence,
            time_period=window,
            current_value=success_second,
            previous_value=success_first,
            analysis_notes=notes,
        )

    def _analyze_confidence_trend(self, events: List[MetricEvent], window: str) -> Optional[TrendAnalysis]:
        """Analyze confidence score trend"""
        confidences = [e.confidence for e in events if e.confidence > 0]

        if len(confidences) < self.min_data_points:
            return None

        # Split into first and second half
        mid_point = len(confidences) // 2
        first_half = confidences[:mid_point]
        second_half = confidences[mid_point:]

        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)

        change_pct = ((avg_second - avg_first) / avg_first) * 100 if avg_first > 0 else 0

        # Determine trend direction
        if abs(change_pct) < 3:
            direction = TrendDirection.STABLE
        elif change_pct > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING

        std_dev = statistics.stdev(confidences)
        confidence = max(0.1, min(1.0, 1.0 - std_dev))

        notes = []
        if avg_second < 0.7:
            notes.append("Average confidence below 70% - pattern quality may need improvement")

        return TrendAnalysis(
            metric_name="confidence_score",
            direction=direction,
            change_percentage=change_pct,
            confidence=confidence,
            time_period=window,
            current_value=avg_second,
            previous_value=avg_first,
            analysis_notes=notes,
        )

    def _analyze_throughput_trend(self, events: List[MetricEvent], window: str) -> Optional[TrendAnalysis]:
        """Analyze throughput trend"""
        if len(events) < self.min_data_points:
            return None

        # Group events by hour for throughput calculation
        hourly_counts = defaultdict(int)
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        if len(hourly_counts) < 2:
            return None

        counts = list(hourly_counts.values())
        mid_point = len(counts) // 2
        first_half = counts[:mid_point]
        second_half = counts[mid_point:]

        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)

        change_pct = ((avg_second - avg_first) / avg_first) * 100 if avg_first > 0 else 0

        # Determine trend direction
        if abs(change_pct) < 10:
            direction = TrendDirection.STABLE
        elif change_pct > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING

        confidence = 0.7  # Medium confidence for throughput

        return TrendAnalysis(
            metric_name="throughput",
            direction=direction,
            change_percentage=change_pct,
            confidence=confidence,
            time_period=window,
            current_value=avg_second,
            previous_value=avg_first,
            analysis_notes=[],
        )


class PerformanceAnalyzer:
    """
    Analyzes system performance and identifies optimization opportunities

    Features:
    - Bottleneck identification
    - Resource utilization analysis
    - Performance regression detection
    - Optimization recommendations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_system_performance(self, events: List[MetricEvent], time_window_hours: int = 24) -> List[PerformanceInsight]:
        """
        Comprehensive system performance analysis

        Args:
            events: List of metric events
            time_window_hours: Analysis time window in hours

        Returns:
            List of performance insights
        """
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        recent_events = [e for e in events if e.timestamp >= cutoff]

        insights = []

        try:
            # Processing time analysis
            insights.extend(self._analyze_processing_times(recent_events))

            # Method performance analysis
            insights.extend(self._analyze_method_performance(recent_events))

            # Error pattern analysis
            insights.extend(self._analyze_error_patterns(recent_events))

            # Resource utilization analysis
            insights.extend(self._analyze_resource_utilization(recent_events))

            # Confidence distribution analysis
            insights.extend(self._analyze_confidence_distribution(recent_events))

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")

        return insights

    def _analyze_processing_times(self, events: List[MetricEvent]) -> List[PerformanceInsight]:
        """Analyze processing time patterns"""
        insights = []

        processing_times = [e.processing_time_ms for e in events if e.processing_time_ms > 0]
        if not processing_times:
            return insights

        avg_time = statistics.mean(processing_times)
        median_time = statistics.median(processing_times)
        std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        max_time = max(processing_times)

        # High processing time variance
        if std_dev > avg_time * 0.5:
            insights.append(
                PerformanceInsight(
                    category="processing_time",
                    insight_type="high_variance",
                    description=f"High variance in processing times detected (std: {std_dev:.1f}ms)",
                    impact_level=AlertLevel.WARNING,
                    recommendation="Investigate outliers and consider implementing caching for common patterns",
                    supporting_data={"avg_time": avg_time, "std_dev": std_dev, "variance_ratio": std_dev / avg_time},
                )
            )

        # Slow outliers
        if max_time > avg_time * 5:
            insights.append(
                PerformanceInsight(
                    category="processing_time",
                    insight_type="slow_outliers",
                    description=f"Slow processing outliers detected (max: {max_time:.1f}ms vs avg: {avg_time:.1f}ms)",
                    impact_level=AlertLevel.WARNING,
                    recommendation="Profile slow operations and optimize pattern matching algorithms",
                    supporting_data={"max_time": max_time, "avg_time": avg_time, "outlier_ratio": max_time / avg_time},
                )
            )

        # Overall performance assessment
        if avg_time > 100:  # Threshold for concern
            level = AlertLevel.CRITICAL if avg_time > 500 else AlertLevel.WARNING
            insights.append(
                PerformanceInsight(
                    category="processing_time",
                    insight_type="performance_degradation",
                    description=f"Average processing time is high: {avg_time:.1f}ms",
                    impact_level=level,
                    recommendation="Optimize pattern matching algorithms and consider parallel processing",
                    supporting_data={
                        "avg_time": avg_time,
                        "median_time": median_time,
                        "p95_time": np.percentile(processing_times, 95) if processing_times else 0,
                    },
                )
            )

        return insights

    def _analyze_method_performance(self, events: List[MetricEvent]) -> List[PerformanceInsight]:
        """Analyze performance by processing method"""
        insights = []

        method_stats = defaultdict(lambda: {"times": [], "successes": 0, "total": 0})

        for event in events:
            method = event.method.value
            method_stats[method]["total"] += 1
            if event.success:
                method_stats[method]["successes"] += 1
            if event.processing_time_ms > 0:
                method_stats[method]["times"].append(event.processing_time_ms)

        # Compare method performance
        method_performance = {}
        for method, stats in method_stats.items():
            if stats["total"] > 0:
                success_rate = stats["successes"] / stats["total"]
                avg_time = statistics.mean(stats["times"]) if stats["times"] else 0
                method_performance[method] = {
                    "success_rate": success_rate,
                    "avg_time": avg_time,
                    "usage_count": stats["total"],
                }

        # Identify underperforming methods
        if len(method_performance) > 1:
            sorted_methods = sorted(
                method_performance.items(), key=lambda x: (x[1]["success_rate"], -x[1]["avg_time"]), reverse=True
            )

            best_method = sorted_methods[0]
            worst_method = sorted_methods[-1]

            success_diff = best_method[1]["success_rate"] - worst_method[1]["success_rate"]

            if success_diff > 0.1:  # 10% difference
                insights.append(
                    PerformanceInsight(
                        category="method_performance",
                        insight_type="method_disparity",
                        description=f"Significant performance gap between methods: {best_method[0]} vs {worst_method[0]}",
                        impact_level=AlertLevel.WARNING,
                        recommendation="Consider optimizing underperforming methods or adjusting method selection logic",
                        supporting_data={
                            "best_method": best_method[0],
                            "worst_method": worst_method[0],
                            "success_rate_diff": success_diff,
                            "method_stats": method_performance,
                        },
                    )
                )

        return insights

    def _analyze_error_patterns(self, events: List[MetricEvent]) -> List[PerformanceInsight]:
        """Analyze error patterns"""
        insights = []

        error_events = [e for e in events if not e.success]
        total_events = len(events)

        if total_events == 0:
            return insights

        error_rate = len(error_events) / total_events

        # High error rate
        if error_rate > 0.1:  # 10% threshold
            level = AlertLevel.CRITICAL if error_rate > 0.2 else AlertLevel.WARNING
            insights.append(
                PerformanceInsight(
                    category="error_rate",
                    insight_type="high_error_rate",
                    description=f"High error rate detected: {error_rate:.1%}",
                    impact_level=level,
                    recommendation="Investigate common error causes and improve input validation",
                    supporting_data={
                        "error_rate": error_rate,
                        "total_errors": len(error_events),
                        "total_events": total_events,
                    },
                )
            )

        # Analyze error types
        error_types = defaultdict(int)
        for event in error_events:
            if event.error_type:
                error_types[event.error_type] += 1

        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            if most_common_error[1] > len(error_events) * 0.3:  # 30% of errors
                insights.append(
                    PerformanceInsight(
                        category="error_patterns",
                        insight_type="dominant_error_type",
                        description=f"Dominant error type: {most_common_error[0]} ({most_common_error[1]} occurrences)",
                        impact_level=AlertLevel.WARNING,
                        recommendation=f"Focus on resolving {most_common_error[0]} errors as they represent the majority",
                        supporting_data={
                            "error_type": most_common_error[0],
                            "count": most_common_error[1],
                            "percentage": most_common_error[1] / len(error_events),
                            "all_error_types": dict(error_types),
                        },
                    )
                )

        return insights

    def _analyze_resource_utilization(self, events: List[MetricEvent]) -> List[PerformanceInsight]:
        """Analyze resource utilization patterns"""
        insights = []

        # Analyze processing load over time
        hourly_loads = defaultdict(int)
        for event in events:
            hour = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_loads[hour] += 1

        if len(hourly_loads) < 2:
            return insights

        loads = list(hourly_loads.values())
        avg_load = statistics.mean(loads)
        max_load = max(loads)
        std_dev = statistics.stdev(loads) if len(loads) > 1 else 0

        # High load variance
        if std_dev > avg_load * 0.5:
            insights.append(
                PerformanceInsight(
                    category="resource_utilization",
                    insight_type="uneven_load_distribution",
                    description=f"Uneven load distribution detected (std: {std_dev:.1f}, avg: {avg_load:.1f})",
                    impact_level=AlertLevel.INFO,
                    recommendation="Consider implementing load balancing or caching during peak hours",
                    supporting_data={
                        "avg_load": avg_load,
                        "max_load": max_load,
                        "std_dev": std_dev,
                        "load_variance": std_dev / avg_load,
                    },
                )
            )

        return insights

    def _analyze_confidence_distribution(self, events: List[MetricEvent]) -> List[PerformanceInsight]:
        """Analyze confidence score distribution"""
        insights = []

        confidences = [e.confidence for e in events if e.confidence > 0]
        if not confidences:
            return insights

        avg_confidence = statistics.mean(confidences)
        low_confidence_count = sum(1 for c in confidences if c < 0.6)
        low_confidence_rate = low_confidence_count / len(confidences)

        # Low confidence rate
        if low_confidence_rate > 0.2:  # 20% threshold
            insights.append(
                PerformanceInsight(
                    category="confidence",
                    insight_type="low_confidence_rate",
                    description=f"High rate of low-confidence results: {low_confidence_rate:.1%}",
                    impact_level=AlertLevel.WARNING,
                    recommendation="Review and improve pattern quality, consider additional training data",
                    supporting_data={
                        "low_confidence_rate": low_confidence_rate,
                        "avg_confidence": avg_confidence,
                        "total_low_confidence": low_confidence_count,
                        "total_with_confidence": len(confidences),
                    },
                )
            )

        return insights


class PatternAnalytics:
    """
    Advanced analytics for pattern performance and optimization

    Features:
    - Pattern ranking and scoring
    - Usage pattern analysis
    - Optimization recommendations
    - Performance benchmarking
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_pattern_performance(
        self, events: List[MetricEvent], pattern_events_only: bool = True
    ) -> List[PatternAnalysisResult]:
        """
        Analyze individual pattern performance

        Args:
            events: List of metric events
            pattern_events_only: Whether to only analyze events with pattern IDs

        Returns:
            List of pattern analysis results
        """
        if pattern_events_only:
            pattern_events = [e for e in events if e.pattern_id]
        else:
            pattern_events = events

        if not pattern_events:
            return []

        # Group events by pattern
        pattern_groups = defaultdict(list)
        for event in pattern_events:
            pattern_groups[event.pattern_id or "unknown"].append(event)

        results = []

        for pattern_id, pattern_events_list in pattern_groups.items():
            try:
                analysis = self._analyze_single_pattern(pattern_id, pattern_events_list)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                self.logger.error(f"Error analyzing pattern {pattern_id}: {e}")

        # Rank patterns by performance score
        results.sort(key=lambda x: x.performance_score, reverse=True)
        for i, result in enumerate(results):
            result.usage_rank = i + 1

        return results

    def _analyze_single_pattern(self, pattern_id: str, events: List[MetricEvent]) -> Optional[PatternAnalysisResult]:
        """Analyze a single pattern's performance"""
        if not events:
            return None

        # Basic metrics
        total_events = len(events)
        successful_events = sum(1 for e in events if e.success)
        success_rate = successful_events / total_events

        # Confidence metrics
        confidences = [e.confidence for e in events if e.confidence > 0]
        avg_confidence = statistics.mean(confidences) if confidences else 0

        # Processing time metrics
        processing_times = [e.processing_time_ms for e in events if e.processing_time_ms > 0]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0

        # Calculate performance score (0-1)
        # Weighted combination of success rate, confidence, and speed
        time_score = max(0, 1 - (avg_processing_time / 1000))  # Normalize to 1 second
        performance_score = success_rate * 0.4 + avg_confidence * 0.4 + time_score * 0.2

        # Identify bottlenecks
        bottlenecks = []
        if success_rate < 0.8:
            bottlenecks.append("Low success rate")
        if avg_confidence < 0.7:
            bottlenecks.append("Low confidence scores")
        if avg_processing_time > 200:
            bottlenecks.append("Slow processing")

        # Error analysis
        error_events = [e for e in events if not e.success]
        error_types = defaultdict(int)
        for event in error_events:
            if event.error_type:
                error_types[event.error_type] += 1

        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            bottlenecks.append(f"Common error: {most_common_error[0]}")

        # Generate recommendations
        recommendations = self._generate_pattern_recommendations(
            success_rate, avg_confidence, avg_processing_time, error_types
        )

        # Calculate optimization potential
        optimization_potential = max(0, 1 - performance_score)

        return PatternAnalysisResult(
            pattern_id=pattern_id,
            usage_rank=0,  # Will be set after sorting
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            performance_score=performance_score,
            optimization_potential=optimization_potential,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _generate_pattern_recommendations(
        self, success_rate: float, avg_confidence: float, avg_processing_time: float, error_types: Dict[str, int]
    ) -> List[str]:
        """Generate optimization recommendations for a pattern"""
        recommendations = []

        if success_rate < 0.8:
            recommendations.append("Review pattern regex for edge cases and improve matching logic")

        if avg_confidence < 0.7:
            recommendations.append("Enhance pattern specificity and add validation rules")

        if avg_processing_time > 200:
            recommendations.append("Optimize regex compilation and consider caching compiled patterns")

        if error_types:
            most_common = max(error_types.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on resolving {most_common[0]} errors (most common)")

        if success_rate > 0.9 and avg_confidence > 0.8:
            recommendations.append("Pattern performing well - consider using as template for new patterns")

        return recommendations


class SystemAnalytics:
    """
    Main analytics coordinator that combines all analysis components

    Features:
    - Comprehensive system analysis
    - Multi-perspective insights
    - Actionable recommendations
    - Historical comparison
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize system analytics

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.trend_analyzer = TrendAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.pattern_analytics = PatternAnalytics()
        self.logger = logging.getLogger(__name__)

    def generate_comprehensive_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report

        Args:
            time_window_hours: Analysis time window in hours

        Returns:
            Complete analytics report
        """
        try:
            # Get events from metrics collector
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            all_events = list(self.metrics_collector._events)
            recent_events = [e for e in all_events if e.timestamp >= cutoff]

            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "time_window_hours": time_window_hours,
                    "total_events_analyzed": len(recent_events),
                    "analysis_period": {"start": cutoff.isoformat(), "end": datetime.now().isoformat()},
                },
                "trend_analysis": [],
                "performance_insights": [],
                "pattern_analysis": [],
                "system_metrics": {},
                "recommendations": {"high_priority": [], "medium_priority": [], "low_priority": []},
            }

            # Trend analysis
            if len(recent_events) >= 10:
                trends = self.trend_analyzer.analyze_performance_trends(recent_events)
                report["trend_analysis"] = [t.to_dict() for t in trends]

            # Performance analysis
            insights = self.performance_analyzer.analyze_system_performance(recent_events, time_window_hours)
            report["performance_insights"] = [i.to_dict() for i in insights]

            # Pattern analysis
            pattern_results = self.pattern_analytics.analyze_pattern_performance(recent_events)
            report["pattern_analysis"] = [p.to_dict() for p in pattern_results]

            # System metrics
            report["system_metrics"] = {
                "performance": self.metrics_collector.get_performance_metrics().to_dict(),
                "system": self.metrics_collector.get_system_metrics().to_dict(),
            }

            # Prioritize recommendations
            self._prioritize_recommendations(report, insights, pattern_results)

            return report

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e)}

    def _prioritize_recommendations(
        self, report: Dict[str, Any], insights: List[PerformanceInsight], pattern_results: List[PatternAnalysisResult]
    ) -> None:
        """Prioritize and categorize recommendations"""
        high_priority = []
        medium_priority = []
        low_priority = []

        # Categorize performance insights
        for insight in insights:
            if insight.impact_level == AlertLevel.CRITICAL:
                high_priority.append(f"[CRITICAL] {insight.description}: {insight.recommendation}")
            elif insight.impact_level == AlertLevel.WARNING:
                medium_priority.append(f"[WARNING] {insight.description}: {insight.recommendation}")
            else:
                low_priority.append(f"[INFO] {insight.description}: {insight.recommendation}")

        # Add pattern recommendations
        for pattern in pattern_results:
            if pattern.optimization_potential > 0.3:
                priority = high_priority if pattern.optimization_potential > 0.5 else medium_priority
                priority.append(f"Pattern {pattern.pattern_id}: {'; '.join(pattern.recommendations)}")

        report["recommendations"]["high_priority"] = high_priority
        report["recommendations"]["medium_priority"] = medium_priority
        report["recommendations"]["low_priority"] = low_priority
