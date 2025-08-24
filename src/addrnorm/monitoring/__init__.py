"""
Monitoring Module

Real-time system monitoring, analytics, and reporting for address normalization.
Provides comprehensive insights into pattern performance, system health, and usage analytics.
"""

# Import statements are now functional since we've created all the modules
from .metrics_collector import (
    MetricsCollector,
    MetricEvent,
    MetricType,
    ProcessingMethod,
    PerformanceMetrics,
    PatternMetrics,
    SystemMetrics,
)

from .analytics import (
    SystemAnalytics,
    TrendAnalyzer,
    PerformanceAnalyzer,
    PatternAnalytics,
    TrendAnalysis,
    PerformanceInsight,
    PatternAnalysisResult,
    TrendDirection,
    AlertLevel,
)

from .reporter import (
    SystemReporter,
    PatternReporter,
    ReportConfig,
    HTMLReportGenerator,
    JSONReportGenerator,
    CSVReportGenerator,
)

from .dashboard import CLIDashboard, DashboardCLI, DashboardConfig, ColorFormatter, LiveMetricsDisplay, AlertsDisplay

__all__ = [
    # Metrics Collection
    "MetricsCollector",
    "MetricEvent",
    "MetricType",
    "ProcessingMethod",
    "PerformanceMetrics",
    "PatternMetrics",
    "SystemMetrics",
    # Analytics
    "SystemAnalytics",
    "TrendAnalyzer",
    "PerformanceAnalyzer",
    "PatternAnalytics",
    "TrendAnalysis",
    "PerformanceInsight",
    "PatternAnalysisResult",
    "TrendDirection",
    "AlertLevel",
    # Reporting
    "SystemReporter",
    "PatternReporter",
    "ReportConfig",
    "HTMLReportGenerator",
    "JSONReportGenerator",
    "CSVReportGenerator",
    # Dashboard
    "CLIDashboard",
    "DashboardCLI",
    "DashboardConfig",
    "ColorFormatter",
    "LiveMetricsDisplay",
    "AlertsDisplay",
]


# Convenience function to create a complete monitoring system
def create_monitoring_system(buffer_size: int = 10000, aggregation_interval: int = 60, enable_persistence: bool = True):
    """
    Create a complete monitoring system with all components

    Args:
        buffer_size: Maximum events in memory buffer
        aggregation_interval: Metrics aggregation interval in seconds
        enable_persistence: Whether to persist metrics to disk

    Returns:
        Tuple of (metrics_collector, analytics, reporter, dashboard)
    """
    # Create metrics collector
    metrics_collector = MetricsCollector(
        buffer_size=buffer_size, aggregation_interval_seconds=aggregation_interval, enable_persistence=enable_persistence
    )

    # Create analytics system
    analytics = SystemAnalytics(metrics_collector)

    # Create reporter
    reporter = SystemReporter(analytics)

    # Create dashboard
    dashboard = CLIDashboard(metrics_collector, analytics)

    return metrics_collector, analytics, reporter, dashboard
