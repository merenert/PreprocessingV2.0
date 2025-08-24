"""
Reporting System

Multi-format report generation for system monitoring and analytics.
Generates JSON, HTML, CSV reports with charts and visualizations.
"""

import json
import csv
import html
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import logging
from io import StringIO

from .analytics import SystemAnalytics, TrendAnalysis, PerformanceInsight, PatternAnalysisResult
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Report generation configuration"""

    include_charts: bool = True
    include_raw_data: bool = False
    chart_width: int = 800
    chart_height: int = 400
    theme: str = "light"  # light, dark
    detail_level: str = "summary"  # summary, detailed, full
    export_format: str = "html"  # html, json, csv


class ChartGenerator:
    """
    Generate charts and visualizations for reports
    Uses Chart.js for web-based charts
    """

    @staticmethod
    def generate_line_chart(data: List[Dict], x_field: str, y_field: str, title: str, config: ReportConfig) -> str:
        """Generate line chart HTML"""
        chart_id = f"chart_{hash(title) % 10000}"

        labels = [str(item[x_field]) for item in data]
        values = [float(item[y_field]) if item[y_field] is not None else 0 for item in data]

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="{config.chart_width}" height="{config.chart_height}"></canvas>
        </div>
        <script>
        var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
        var chart_{chart_id} = new Chart(ctx_{chart_id}, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: '{title}',
                    data: {json.dumps(values)},
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def generate_bar_chart(data: Dict[str, Union[int, float]], title: str, config: ReportConfig) -> str:
        """Generate bar chart HTML"""
        chart_id = f"chart_{hash(title) % 10000}"

        labels = list(data.keys())
        values = list(data.values())

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="{config.chart_width}" height="{config.chart_height}"></canvas>
        </div>
        <script>
        var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
        var chart_{chart_id} = new Chart(ctx_{chart_id}, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: '{title}',
                    data: {json.dumps(values)},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 205, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 205, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def generate_pie_chart(data: Dict[str, Union[int, float]], title: str, config: ReportConfig) -> str:
        """Generate pie chart HTML"""
        chart_id = f"chart_{hash(title) % 10000}"

        labels = list(data.keys())
        values = list(data.values())

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="{config.chart_width}" height="{config.chart_height}"></canvas>
        </div>
        <script>
        var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
        var chart_{chart_id} = new Chart(ctx_{chart_id}, {{
            type: 'pie',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    data: {json.dumps(values)},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)',
                        'rgba(255, 159, 64, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }},
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        </script>
        """


class HTMLReportGenerator:
    """Generate HTML reports with charts and styling"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.chart_generator = ChartGenerator()

    def generate_report(self, analytics_data: Dict[str, Any]) -> str:
        """Generate complete HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Address Normalization System - Analytics Report</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {self._get_styles()}
        </head>
        <body>
            <div class="container">
                {self._generate_header(analytics_data)}
                {self._generate_executive_summary(analytics_data)}
                {self._generate_performance_section(analytics_data)}
                {self._generate_trends_section(analytics_data)}
                {self._generate_patterns_section(analytics_data)}
                {self._generate_recommendations_section(analytics_data)}
                {self._generate_footer()}
            </div>
        </body>
        </html>
        """
        return html_content

    def _get_styles(self) -> str:
        """Get CSS styles for the report"""
        theme_colors = {
            "light": {"bg": "#ffffff", "text": "#333333", "border": "#e0e0e0", "accent": "#2196F3"},
            "dark": {"bg": "#1e1e1e", "text": "#ffffff", "border": "#404040", "accent": "#64B5F6"},
        }

        colors = theme_colors[self.config.theme]

        return f"""
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: {colors['bg']};
                color: {colors['text']};
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                border-bottom: 2px solid {colors['accent']};
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid {colors['border']};
                border-radius: 8px;
                background-color: {colors['bg']};
            }}
            .section h2 {{
                color: {colors['accent']};
                margin-top: 0;
                border-bottom: 1px solid {colors['border']};
                padding-bottom: 10px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                padding: 15px;
                border: 1px solid {colors['border']};
                border-radius: 5px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: {colors['accent']};
            }}
            .metric-label {{
                font-size: 0.9em;
                color: {colors['text']};
                opacity: 0.8;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .alert {{
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
                border-left: 4px solid;
            }}
            .alert-critical {{
                background-color: #ffebee;
                border-color: #f44336;
                color: #c62828;
            }}
            .alert-warning {{
                background-color: #fff3e0;
                border-color: #ff9800;
                color: #e65100;
            }}
            .alert-info {{
                background-color: #e3f2fd;
                border-color: #2196f3;
                color: #1565c0;
            }}
            .pattern-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .pattern-table th,
            .pattern-table td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid {colors['border']};
            }}
            .pattern-table th {{
                background-color: {colors['accent']};
                color: white;
            }}
            .recommendations {{
                list-style: none;
                padding: 0;
            }}
            .recommendations li {{
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid {colors['accent']};
                background-color: rgba(33, 150, 243, 0.1);
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid {colors['border']};
                opacity: 0.7;
            }}
        </style>
        """

    def _generate_header(self, data: Dict[str, Any]) -> str:
        """Generate report header"""
        metadata = data.get("report_metadata", {})
        generated_at = metadata.get("generated_at", "Unknown")
        time_window = metadata.get("time_window_hours", "Unknown")

        return f"""
        <div class="header">
            <h1>Address Normalization System</h1>
            <h2>Analytics Report</h2>
            <p>Generated: {generated_at}</p>
            <p>Analysis Window: {time_window} hours</p>
        </div>
        """

    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        system_metrics = data.get("system_metrics", {})
        performance = system_metrics.get("performance", {})

        total_processed = performance.get("total_processed", 0)
        avg_time = performance.get("avg_processing_time_ms", 0)
        error_rate = performance.get("error_rate", 0)
        throughput = performance.get("throughput_per_second", 0)

        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_processed:,}</div>
                    <div class="metric-label">Total Processed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_time:.1f}ms</div>
                    <div class="metric-label">Avg Processing Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{error_rate:.1%}</div>
                    <div class="metric-label">Error Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{throughput:.1f}/s</div>
                    <div class="metric-label">Throughput</div>
                </div>
            </div>
        </div>
        """

    def _generate_performance_section(self, data: Dict[str, Any]) -> str:
        """Generate performance metrics section"""
        performance = data.get("system_metrics", {}).get("performance", {})
        insights = data.get("performance_insights", [])

        charts_html = ""
        if self.config.include_charts:
            # Processing time distribution chart
            time_data = {
                "Average": performance.get("avg_processing_time_ms", 0),
                "Median": performance.get("median_processing_time_ms", 0),
                "P95": performance.get("p95_processing_time_ms", 0),
                "P99": performance.get("p99_processing_time_ms", 0),
            }
            charts_html = self.chart_generator.generate_bar_chart(time_data, "Processing Time Distribution (ms)", self.config)

        insights_html = ""
        for insight in insights:
            alert_class = f"alert-{insight['impact_level']}"
            insights_html += f"""
            <div class="alert {alert_class}">
                <strong>{insight['insight_type'].replace('_', ' ').title()}:</strong>
                {html.escape(insight['description'])}
                <br><em>Recommendation: {html.escape(insight['recommendation'])}</em>
            </div>
            """

        return f"""
        <div class="section">
            <h2>Performance Analysis</h2>
            {charts_html}
            <h3>Performance Insights</h3>
            {insights_html if insights_html else '<p>No performance issues detected.</p>'}
        </div>
        """

    def _generate_trends_section(self, data: Dict[str, Any]) -> str:
        """Generate trends analysis section"""
        trends = data.get("trend_analysis", [])

        if not trends:
            return """
            <div class="section">
                <h2>Trend Analysis</h2>
                <p>Insufficient data for trend analysis.</p>
            </div>
            """

        trends_html = ""
        for trend in trends:
            direction_icon = {"improving": "📈", "declining": "📉", "stable": "➡️", "volatile": "📊"}.get(
                trend["direction"], "📊"
            )

            trends_html += f"""
            <div class="metric-card">
                <h4>{direction_icon} {trend['metric_name'].replace('_', ' ').title()}</h4>
                <p><strong>Direction:</strong> {trend['direction'].title()}</p>
                <p><strong>Change:</strong> {trend['change_percentage']:+.1f}%</p>
                <p><strong>Period:</strong> {trend['time_period']}</p>
                <p><strong>Confidence:</strong> {trend['confidence']:.1%}</p>
            </div>
            """

        return f"""
        <div class="section">
            <h2>Trend Analysis</h2>
            <div class="metric-grid">
                {trends_html}
            </div>
        </div>
        """

    def _generate_patterns_section(self, data: Dict[str, Any]) -> str:
        """Generate pattern analysis section"""
        patterns = data.get("pattern_analysis", [])

        if not patterns:
            return """
            <div class="section">
                <h2>Pattern Performance</h2>
                <p>No pattern data available.</p>
            </div>
            """

        # Top 10 patterns table
        top_patterns = patterns[:10]
        table_html = """
        <table class="pattern-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Pattern ID</th>
                    <th>Success Rate</th>
                    <th>Avg Confidence</th>
                    <th>Performance Score</th>
                    <th>Optimization Potential</th>
                </tr>
            </thead>
            <tbody>
        """

        for pattern in top_patterns:
            table_html += f"""
            <tr>
                <td>{pattern['usage_rank']}</td>
                <td>{html.escape(pattern['pattern_id'])}</td>
                <td>{pattern['success_rate']:.1%}</td>
                <td>{pattern['avg_confidence']:.3f}</td>
                <td>{pattern['performance_score']:.3f}</td>
                <td>{pattern['optimization_potential']:.3f}</td>
            </tr>
            """

        table_html += "</tbody></table>"

        # Pattern performance distribution chart
        charts_html = ""
        if self.config.include_charts and len(patterns) > 1:
            score_distribution = {
                "Excellent (>0.9)": len([p for p in patterns if p["performance_score"] > 0.9]),
                "Good (0.7-0.9)": len([p for p in patterns if 0.7 < p["performance_score"] <= 0.9]),
                "Fair (0.5-0.7)": len([p for p in patterns if 0.5 < p["performance_score"] <= 0.7]),
                "Poor (<0.5)": len([p for p in patterns if p["performance_score"] <= 0.5]),
            }
            charts_html = self.chart_generator.generate_pie_chart(
                score_distribution, "Pattern Performance Distribution", self.config
            )

        return f"""
        <div class="section">
            <h2>Pattern Performance</h2>
            {charts_html}
            <h3>Top Performing Patterns</h3>
            {table_html}
        </div>
        """

    def _generate_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = data.get("recommendations", {})

        high_priority = recommendations.get("high_priority", [])
        medium_priority = recommendations.get("medium_priority", [])
        low_priority = recommendations.get("low_priority", [])

        sections = []

        if high_priority:
            high_html = '<ul class="recommendations">'
            for rec in high_priority:
                high_html += f'<li class="alert alert-critical">{html.escape(rec)}</li>'
            high_html += "</ul>"
            sections.append(f"<h3>🔴 High Priority</h3>{high_html}")

        if medium_priority:
            medium_html = '<ul class="recommendations">'
            for rec in medium_priority:
                medium_html += f'<li class="alert alert-warning">{html.escape(rec)}</li>'
            medium_html += "</ul>"
            sections.append(f"<h3>🟡 Medium Priority</h3>{medium_html}")

        if low_priority and self.config.detail_level == "full":
            low_html = '<ul class="recommendations">'
            for rec in low_priority:
                low_html += f'<li class="alert alert-info">{html.escape(rec)}</li>'
            low_html += "</ul>"
            sections.append(f"<h3>🟢 Low Priority</h3>{low_html}")

        if not sections:
            sections.append("<p>No specific recommendations at this time. System is performing well.</p>")

        return f"""
        <div class="section">
            <h2>Recommendations</h2>
            {"".join(sections)}
        </div>
        """

    def _generate_footer(self) -> str:
        """Generate report footer"""
        return """
        <div class="footer">
            <p>Address Normalization System Analytics Report</p>
            <p>Generated automatically by the monitoring system</p>
        </div>
        """


class JSONReportGenerator:
    """Generate JSON reports for API consumption"""

    def __init__(self, config: ReportConfig):
        self.config = config

    def generate_report(self, analytics_data: Dict[str, Any]) -> str:
        """Generate JSON report"""
        if self.config.detail_level == "summary":
            # Include only high-level metrics
            report_data = {
                "metadata": analytics_data.get("report_metadata", {}),
                "summary": {
                    "performance": analytics_data.get("system_metrics", {}).get("performance", {}),
                    "high_priority_recommendations": analytics_data.get("recommendations", {}).get("high_priority", []),
                },
            }
        elif self.config.detail_level == "detailed":
            # Include most data but exclude raw events
            report_data = analytics_data.copy()
            if not self.config.include_raw_data:
                report_data.pop("raw_events", None)
        else:  # full
            report_data = analytics_data

        return json.dumps(report_data, indent=2, default=str)


class CSVReportGenerator:
    """Generate CSV reports for data analysis"""

    def __init__(self, config: ReportConfig):
        self.config = config

    def generate_report(self, analytics_data: Dict[str, Any]) -> str:
        """Generate CSV report with multiple sheets as sections"""
        output = StringIO()

        # Performance metrics
        output.write("=== PERFORMANCE METRICS ===\\n")
        performance = analytics_data.get("system_metrics", {}).get("performance", {})
        if performance:
            writer = csv.writer(output)
            writer.writerow(["Metric", "Value"])
            for key, value in performance.items():
                writer.writerow([key.replace("_", " ").title(), value])
            output.write("\\n")

        # Pattern analysis
        output.write("=== PATTERN ANALYSIS ===\\n")
        patterns = analytics_data.get("pattern_analysis", [])
        if patterns:
            writer = csv.writer(output)
            headers = [
                "Pattern ID",
                "Usage Rank",
                "Success Rate",
                "Avg Confidence",
                "Performance Score",
                "Optimization Potential",
            ]
            writer.writerow(headers)
            for pattern in patterns:
                writer.writerow(
                    [
                        pattern["pattern_id"],
                        pattern["usage_rank"],
                        pattern["success_rate"],
                        pattern["avg_confidence"],
                        pattern["performance_score"],
                        pattern["optimization_potential"],
                    ]
                )
            output.write("\\n")

        # Trend analysis
        output.write("=== TREND ANALYSIS ===\\n")
        trends = analytics_data.get("trend_analysis", [])
        if trends:
            writer = csv.writer(output)
            headers = ["Metric", "Direction", "Change %", "Confidence", "Time Period"]
            writer.writerow(headers)
            for trend in trends:
                writer.writerow(
                    [
                        trend["metric_name"],
                        trend["direction"],
                        trend["change_percentage"],
                        trend["confidence"],
                        trend["time_period"],
                    ]
                )
            output.write("\\n")

        # Recommendations
        output.write("=== RECOMMENDATIONS ===\\n")
        recommendations = analytics_data.get("recommendations", {})
        writer = csv.writer(output)
        writer.writerow(["Priority", "Recommendation"])

        for rec in recommendations.get("high_priority", []):
            writer.writerow(["High", rec])
        for rec in recommendations.get("medium_priority", []):
            writer.writerow(["Medium", rec])
        for rec in recommendations.get("low_priority", []):
            writer.writerow(["Low", rec])

        return output.getvalue()


class SystemReporter:
    """
    Main reporting system coordinator

    Features:
    - Multi-format report generation
    - Configurable detail levels
    - Automatic report scheduling
    - Export capabilities
    """

    def __init__(self, analytics: SystemAnalytics):
        """
        Initialize system reporter

        Args:
            analytics: System analytics instance
        """
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

    def generate_report(self, format: str = "html", time_window_hours: int = 24, config: Optional[ReportConfig] = None) -> str:
        """
        Generate report in specified format

        Args:
            format: Output format (html, json, csv)
            time_window_hours: Analysis time window
            config: Report configuration

        Returns:
            Generated report content
        """
        if config is None:
            config = ReportConfig(export_format=format)

        try:
            # Get analytics data
            analytics_data = self.analytics.generate_comprehensive_report(time_window_hours)

            # Generate report based on format
            if format.lower() == "html":
                generator = HTMLReportGenerator(config)
                return generator.generate_report(analytics_data)
            elif format.lower() == "json":
                generator = JSONReportGenerator(config)
                return generator.generate_report(analytics_data)
            elif format.lower() == "csv":
                generator = CSVReportGenerator(config)
                return generator.generate_report(analytics_data)
            else:
                raise ValueError(f"Unsupported report format: {format}")

        except Exception as e:
            self.logger.error(f"Error generating {format} report: {e}")
            return f"Error generating report: {str(e)}"

    def save_report(
        self, output_path: Path, format: str = "html", time_window_hours: int = 24, config: Optional[ReportConfig] = None
    ) -> bool:
        """
        Generate and save report to file

        Args:
            output_path: Path to save the report
            format: Output format
            time_window_hours: Analysis time window
            config: Report configuration

        Returns:
            Success status
        """
        try:
            report_content = self.generate_report(format, time_window_hours, config)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.logger.info(f"Report saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving report to {output_path}: {e}")
            return False

    def get_available_formats(self) -> List[str]:
        """Get list of available report formats"""
        return ["html", "json", "csv"]


class PatternReporter:
    """
    Specialized reporter for pattern-specific analysis

    Features:
    - Pattern comparison reports
    - Performance tracking
    - Usage analytics
    - Optimization insights
    """

    def __init__(self, analytics: SystemAnalytics):
        """
        Initialize pattern reporter

        Args:
            analytics: System analytics instance
        """
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

    def generate_pattern_comparison_report(self, pattern_ids: List[str], time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comparison report for specific patterns

        Args:
            pattern_ids: List of pattern IDs to compare
            time_window_hours: Analysis time window

        Returns:
            Pattern comparison data
        """
        try:
            # Get analytics data
            full_report = self.analytics.generate_comprehensive_report(time_window_hours)
            pattern_analysis = full_report.get("pattern_analysis", [])

            # Filter for requested patterns
            target_patterns = [p for p in pattern_analysis if p["pattern_id"] in pattern_ids]

            if not target_patterns:
                return {"error": "No data found for specified patterns"}

            # Generate comparison
            comparison = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "patterns_compared": len(target_patterns),
                    "time_window_hours": time_window_hours,
                },
                "patterns": target_patterns,
                "summary": {
                    "best_performing": max(target_patterns, key=lambda x: x["performance_score"]),
                    "worst_performing": min(target_patterns, key=lambda x: x["performance_score"]),
                    "avg_success_rate": sum(p["success_rate"] for p in target_patterns) / len(target_patterns),
                    "avg_confidence": sum(p["avg_confidence"] for p in target_patterns) / len(target_patterns),
                },
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Error generating pattern comparison report: {e}")
            return {"error": str(e)}

    def generate_pattern_health_report(self, pattern_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate detailed health report for a specific pattern

        Args:
            pattern_id: Pattern ID to analyze
            time_window_hours: Analysis time window

        Returns:
            Pattern health data
        """
        try:
            # Get analytics data
            full_report = self.analytics.generate_comprehensive_report(time_window_hours)
            pattern_analysis = full_report.get("pattern_analysis", [])

            # Find target pattern
            target_pattern = next((p for p in pattern_analysis if p["pattern_id"] == pattern_id), None)

            if not target_pattern:
                return {"error": f"Pattern {pattern_id} not found in analysis"}

            # Calculate health score
            health_score = (
                target_pattern["success_rate"] * 0.4
                + target_pattern["avg_confidence"] * 0.3
                + target_pattern["performance_score"] * 0.3
            )

            # Determine health status
            if health_score >= 0.8:
                health_status = "Excellent"
            elif health_score >= 0.6:
                health_status = "Good"
            elif health_score >= 0.4:
                health_status = "Fair"
            else:
                health_status = "Poor"

            return {
                "pattern_id": pattern_id,
                "health_score": health_score,
                "health_status": health_status,
                "metrics": target_pattern,
                "analysis_timestamp": datetime.now().isoformat(),
                "recommendations": target_pattern["recommendations"],
            }

        except Exception as e:
            self.logger.error(f"Error generating pattern health report for {pattern_id}: {e}")
            return {"error": str(e)}
