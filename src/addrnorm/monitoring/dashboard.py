"""
CLI Dashboard

Interactive command-line dashboard for real-time system monitoring.
Provides live metrics, alerts, and system control capabilities.
"""

import time
import threading
import argparse
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
from pathlib import Path
import logging

from .metrics_collector import MetricsCollector, ProcessingMethod
from .analytics import SystemAnalytics
from .reporter import SystemReporter, ReportConfig

# For CLI styling and interaction
try:
    import click
    import colorama
    from colorama import Fore, Back, Style

    colorama.init()
    HAS_STYLING = True
except ImportError:
    HAS_STYLING = False
    Fore = Back = Style = type("MockStyle", (), {"__getattr__": lambda self, name: ""})()

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration"""

    refresh_interval: int = 5  # seconds
    max_rows: int = 50
    show_patterns: bool = True
    show_trends: bool = True
    show_alerts: bool = True
    auto_refresh: bool = True
    log_level: str = "INFO"


class ColorFormatter:
    """Format text with colors for better readability"""

    @staticmethod
    def success(text: str) -> str:
        """Format success message"""
        return f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}" if HAS_STYLING else f"✓ {text}"

    @staticmethod
    def warning(text: str) -> str:
        """Format warning message"""
        return f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}" if HAS_STYLING else f"⚠ {text}"

    @staticmethod
    def error(text: str) -> str:
        """Format error message"""
        return f"{Fore.RED}✗ {text}{Style.RESET_ALL}" if HAS_STYLING else f"✗ {text}"

    @staticmethod
    def info(text: str) -> str:
        """Format info message"""
        return f"{Fore.CYAN}ℹ {text}{Style.RESET_ALL}" if HAS_STYLING else f"ℹ {text}"

    @staticmethod
    def highlight(text: str) -> str:
        """Highlight text"""
        return f"{Fore.YELLOW}{text}{Style.RESET_ALL}" if HAS_STYLING else text

    @staticmethod
    def header(text: str) -> str:
        """Format header"""
        return f"{Fore.BLUE}{Style.BRIGHT}{text}{Style.RESET_ALL}" if HAS_STYLING else text

    @staticmethod
    def metric_value(value: str, status: str = "normal") -> str:
        """Format metric value with status color"""
        colors = {"good": Fore.GREEN, "warning": Fore.YELLOW, "critical": Fore.RED, "normal": Fore.WHITE}
        color = colors.get(status, Fore.WHITE) if HAS_STYLING else ""
        reset = Style.RESET_ALL if HAS_STYLING else ""
        return f"{color}{value}{reset}"


class LiveMetricsDisplay:
    """Display live metrics in formatted layout"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.formatter = ColorFormatter()

    def display_performance_metrics(self) -> List[str]:
        """Display performance metrics"""
        metrics = self.metrics_collector.get_performance_metrics()

        lines = [
            self.formatter.header("📊 PERFORMANCE METRICS"),
            "─" * 50,
        ]

        # Processing time
        avg_time = metrics.avg_processing_time_ms
        time_status = "good" if avg_time < 100 else "warning" if avg_time < 500 else "critical"
        lines.append(f"Avg Processing Time: {self.formatter.metric_value(f'{avg_time:.1f}ms', time_status)}")

        # Throughput
        throughput = metrics.throughput_per_second
        throughput_status = "good" if throughput > 10 else "warning" if throughput > 1 else "normal"
        lines.append(f"Throughput: {self.formatter.metric_value(f'{throughput:.1f}/sec', throughput_status)}")

        # Error rate
        error_rate = metrics.error_rate
        error_status = "good" if error_rate < 0.05 else "warning" if error_rate < 0.15 else "critical"
        lines.append(f"Error Rate: {self.formatter.metric_value(f'{error_rate:.1%}', error_status)}")

        # Total processed
        lines.append(f"Total Processed: {self.formatter.metric_value(f'{metrics.total_processed:,}', 'normal')}")

        # Memory usage
        memory = metrics.memory_usage_mb
        memory_status = "good" if memory < 100 else "warning" if memory < 500 else "critical"
        lines.append(f"Memory Usage: {self.formatter.metric_value(f'{memory:.1f}MB', memory_status)}")

        return lines

    def display_system_metrics(self) -> List[str]:
        """Display system metrics"""
        metrics = self.metrics_collector.get_system_metrics()

        lines = [
            "",
            self.formatter.header("🔧 SYSTEM METRICS"),
            "─" * 50,
        ]

        # Uptime
        uptime_hours = metrics.uptime_seconds / 3600
        lines.append(f"Uptime: {self.formatter.metric_value(f'{uptime_hours:.1f} hours', 'normal')}")

        # Method distribution
        if metrics.method_distribution:
            lines.append("Method Distribution:")
            total_methods = sum(metrics.method_distribution.values())
            for method, count in sorted(metrics.method_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_methods) * 100 if total_methods > 0 else 0
                lines.append(f"  {method}: {self.formatter.metric_value(f'{count} ({percentage:.1f}%)', 'normal')}")

        # Confidence distribution
        if metrics.confidence_distribution:
            lines.append("Confidence Distribution:")
            total_conf = sum(metrics.confidence_distribution.values())
            for bucket, count in sorted(metrics.confidence_distribution.items()):
                percentage = (count / total_conf) * 100 if total_conf > 0 else 0
                status = "good" if bucket in ["very_high", "high"] else "warning" if bucket == "medium" else "critical"
                lines.append(f"  {bucket}: {self.formatter.metric_value(f'{count} ({percentage:.1f}%)', status)}")

        return lines

    def display_top_patterns(self, limit: int = 5) -> List[str]:
        """Display top performing patterns"""
        pattern_metrics = self.metrics_collector.get_pattern_metrics()

        if not pattern_metrics:
            return ["", self.formatter.header("📋 TOP PATTERNS"), "─" * 50, "No pattern data available"]

        # Sort by usage count
        sorted_patterns = sorted(pattern_metrics.items(), key=lambda x: x[1].usage_count, reverse=True)[:limit]

        lines = [
            "",
            self.formatter.header("📋 TOP PATTERNS"),
            "─" * 50,
        ]

        for i, (pattern_id, metrics) in enumerate(sorted_patterns, 1):
            success_status = "good" if metrics.success_rate > 0.9 else "warning" if metrics.success_rate > 0.7 else "critical"
            conf_status = "good" if metrics.avg_confidence > 0.8 else "warning" if metrics.avg_confidence > 0.6 else "critical"

            lines.extend(
                [
                    f"{i}. Pattern: {self.formatter.highlight(pattern_id[:30])}",
                    f"   Usage: {self.formatter.metric_value(str(metrics.usage_count), 'normal')} | "
                    f"Success: {self.formatter.metric_value(f'{metrics.success_rate:.1%}', success_status)} | "
                    f"Confidence: {self.formatter.metric_value(f'{metrics.avg_confidence:.3f}', conf_status)}",
                ]
            )

        return lines


class AlertsDisplay:
    """Display system alerts and notifications"""

    def __init__(self, analytics: SystemAnalytics):
        self.analytics = analytics
        self.formatter = ColorFormatter()
        self.last_alerts = []

    def get_current_alerts(self) -> List[str]:
        """Get current system alerts"""
        try:
            # Generate quick report for alerts
            report = self.analytics.generate_comprehensive_report(time_window_hours=1)
            insights = report.get("performance_insights", [])
            recommendations = report.get("recommendations", {})

            alerts = []

            # Convert insights to alerts
            for insight in insights:
                if insight["impact_level"] == "critical":
                    alerts.append(self.formatter.error(f"CRITICAL: {insight['description']}"))
                elif insight["impact_level"] == "warning":
                    alerts.append(self.formatter.warning(f"WARNING: {insight['description']}"))

            # Add high priority recommendations as alerts
            for rec in recommendations.get("high_priority", []):
                alerts.append(self.formatter.warning(f"RECOMMENDATION: {rec}"))

            # Limit alerts and cache
            self.last_alerts = alerts[:5]
            return self.last_alerts

        except Exception as e:
            return [self.formatter.error(f"Error getting alerts: {str(e)}")]

    def display_alerts(self) -> List[str]:
        """Display current alerts"""
        alerts = self.get_current_alerts()

        lines = [
            "",
            self.formatter.header("🚨 ALERTS"),
            "─" * 50,
        ]

        if alerts:
            lines.extend(alerts)
        else:
            lines.append(self.formatter.success("No active alerts"))

        return lines


class CLIDashboard:
    """
    Interactive command-line dashboard

    Features:
    - Real-time metrics display
    - Interactive commands
    - Report generation
    - System control
    """

    def __init__(
        self, metrics_collector: MetricsCollector, analytics: SystemAnalytics, config: Optional[DashboardConfig] = None
    ):
        """
        Initialize CLI dashboard

        Args:
            metrics_collector: Metrics collector instance
            analytics: Analytics instance
            config: Dashboard configuration
        """
        self.metrics_collector = metrics_collector
        self.analytics = analytics
        self.config = config or DashboardConfig()

        self.reporter = SystemReporter(analytics)
        self.metrics_display = LiveMetricsDisplay(metrics_collector)
        self.alerts_display = AlertsDisplay(analytics)
        self.formatter = ColorFormatter()

        self.running = False
        self.display_thread = None

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def start_dashboard(self) -> None:
        """Start the interactive dashboard"""
        self.running = True

        print(self.formatter.header("Address Normalization System - Monitoring Dashboard"))
        print("=" * 70)
        print()

        if self.config.auto_refresh:
            # Start auto-refresh in background
            self.display_thread = threading.Thread(target=self._auto_refresh_loop)
            self.display_thread.daemon = True
            self.display_thread.start()

            # Interactive command loop
            self._command_loop()
        else:
            # Single display
            self._display_all_metrics()

    def stop_dashboard(self) -> None:
        """Stop the dashboard"""
        self.running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)

    def _auto_refresh_loop(self) -> None:
        """Auto-refresh display loop"""
        while self.running:
            try:
                self._clear_screen()
                self._display_all_metrics()
                time.sleep(self.config.refresh_interval)
            except Exception as e:
                self.logger.error(f"Error in auto-refresh: {e}")
                time.sleep(1)

    def _command_loop(self) -> None:
        """Interactive command loop"""
        print(self.formatter.info("Interactive mode - Press Ctrl+C to exit"))
        print(self.formatter.info("Commands: 'r' (refresh), 'report' (generate), 'help' (commands)"))
        print()

        try:
            while self.running:
                try:
                    command = input("> ").strip().lower()
                    self._handle_command(command)
                except KeyboardInterrupt:
                    print("\\n" + self.formatter.info("Shutting down dashboard..."))
                    break
                except EOFError:
                    break
        finally:
            self.stop_dashboard()

    def _handle_command(self, command: str) -> None:
        """Handle interactive commands"""
        if command == "r" or command == "refresh":
            self._clear_screen()
            self._display_all_metrics()

        elif command == "report":
            self._generate_report_interactive()

        elif command == "alerts":
            self._display_alerts_only()

        elif command == "patterns":
            self._display_patterns_detailed()

        elif command == "reset":
            self._reset_metrics()

        elif command == "help":
            self._display_help()

        elif command == "q" or command == "quit" or command == "exit":
            print(self.formatter.info("Shutting down dashboard..."))
            self.running = False

        elif command:
            print(self.formatter.warning(f"Unknown command: {command}. Type 'help' for available commands."))

    def _display_all_metrics(self) -> None:
        """Display all metrics sections"""
        lines = []

        # Header with timestamp
        lines.append(self.formatter.header(f"📈 DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        lines.append("=" * 70)

        # Performance metrics
        lines.extend(self.metrics_display.display_performance_metrics())

        # System metrics
        lines.extend(self.metrics_display.display_system_metrics())

        # Top patterns
        if self.config.show_patterns:
            lines.extend(self.metrics_display.display_top_patterns())

        # Alerts
        if self.config.show_alerts:
            lines.extend(self.alerts_display.display_alerts())

        # Footer
        lines.extend(
            [
                "",
                "─" * 70,
                self.formatter.info(f"Auto-refresh: {self.config.refresh_interval}s | Commands: r, report, help, q"),
            ]
        )

        # Print all lines
        for line in lines[-self.config.max_rows :]:
            print(line)

    def _display_alerts_only(self) -> None:
        """Display only alerts section"""
        alerts = self.alerts_display.display_alerts()
        for line in alerts:
            print(line)

    def _display_patterns_detailed(self) -> None:
        """Display detailed pattern information"""
        pattern_metrics = self.metrics_collector.get_pattern_metrics()

        print(self.formatter.header("📋 DETAILED PATTERN ANALYSIS"))
        print("=" * 70)

        if not pattern_metrics:
            print("No pattern data available")
            return

        # Sort by performance score (calculated)
        sorted_patterns = sorted(pattern_metrics.items(), key=lambda x: x[1].success_rate * x[1].avg_confidence, reverse=True)

        for i, (pattern_id, metrics) in enumerate(sorted_patterns[:10], 1):
            performance_score = metrics.success_rate * metrics.avg_confidence

            print(f"\\n{i}. Pattern ID: {self.formatter.highlight(pattern_id)}")
            print(f"   Usage Count: {metrics.usage_count}")
            print(f"   Success Rate: {metrics.success_rate:.1%}")
            print(f"   Avg Confidence: {metrics.avg_confidence:.3f}")
            print(f"   Avg Processing Time: {metrics.avg_processing_time_ms:.1f}ms")
            print(f"   Performance Score: {performance_score:.3f}")

            if metrics.last_used:
                print(f"   Last Used: {metrics.last_used.strftime('%Y-%m-%d %H:%M:%S')}")

            if metrics.common_failures:
                print(f"   Common Failures: {', '.join(metrics.common_failures)}")

    def _generate_report_interactive(self) -> None:
        """Interactive report generation"""
        print(self.formatter.info("Report Generation"))
        print("Available formats: html, json, csv")

        try:
            format_choice = input("Choose format (html): ").strip().lower() or "html"
            if format_choice not in ["html", "json", "csv"]:
                print(self.formatter.warning("Invalid format. Using HTML."))
                format_choice = "html"

            hours = input("Time window in hours (24): ").strip() or "24"
            try:
                hours = int(hours)
            except ValueError:
                print(self.formatter.warning("Invalid hours. Using 24."))
                hours = 24

            # Generate report
            print(self.formatter.info("Generating report..."))

            config = ReportConfig(export_format=format_choice, detail_level="detailed", include_charts=True)

            report_content = self.reporter.generate_report(format_choice, hours, config)

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.{format_choice}"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            print(self.formatter.success(f"Report saved to: {filename}"))

        except Exception as e:
            print(self.formatter.error(f"Error generating report: {e}"))

    def _reset_metrics(self) -> None:
        """Reset all metrics"""
        confirm = input("Are you sure you want to reset all metrics? (y/N): ").strip().lower()
        if confirm == "y" or confirm == "yes":
            self.metrics_collector.reset_metrics()
            print(self.formatter.success("Metrics reset successfully"))
        else:
            print(self.formatter.info("Reset cancelled"))

    def _display_help(self) -> None:
        """Display help information"""
        help_text = f"""
{self.formatter.header("📖 DASHBOARD COMMANDS")}
{"─" * 50}

{self.formatter.highlight("Navigation:")}
  r, refresh     - Refresh the display
  q, quit, exit  - Exit the dashboard

{self.formatter.highlight("Information:")}
  alerts         - Show only alerts section
  patterns       - Show detailed pattern analysis
  help           - Show this help message

{self.formatter.highlight("Actions:")}
  report         - Generate analytics report
  reset          - Reset all metrics (with confirmation)

{self.formatter.highlight("Keyboard Shortcuts:")}
  Ctrl+C         - Exit dashboard
  Enter          - Refresh display
        """
        print(help_text)

    def _clear_screen(self) -> None:
        """Clear the terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")


class DashboardCLI:
    """
    Command-line interface for the dashboard

    Provides main entry point and argument parsing
    """

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Address Normalization System - Monitoring Dashboard",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                          # Start interactive dashboard
  %(prog)s --no-auto-refresh        # Single display mode
  %(prog)s --refresh-interval 10    # Refresh every 10 seconds
  %(prog)s --report html            # Generate HTML report and exit
            """,
        )

        parser.add_argument("--refresh-interval", type=int, default=5, help="Auto-refresh interval in seconds (default: 5)")

        parser.add_argument("--no-auto-refresh", action="store_true", help="Disable auto-refresh (single display mode)")

        parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to display (default: 50)")

        parser.add_argument("--no-patterns", action="store_true", help="Hide pattern information")

        parser.add_argument("--no-alerts", action="store_true", help="Hide alerts section")

        parser.add_argument("--report", choices=["html", "json", "csv"], help="Generate report in specified format and exit")

        parser.add_argument("--report-hours", type=int, default=24, help="Time window for report in hours (default: 24)")

        parser.add_argument("--output", type=str, help="Output file for report (auto-generated if not specified)")

        parser.add_argument(
            "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level (default: INFO)"
        )

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI dashboard"""
        try:
            parsed_args = self.parser.parse_args(args)

            # Create configuration
            config = DashboardConfig(
                refresh_interval=parsed_args.refresh_interval,
                max_rows=parsed_args.max_rows,
                show_patterns=not parsed_args.no_patterns,
                show_alerts=not parsed_args.no_alerts,
                auto_refresh=not parsed_args.no_auto_refresh,
                log_level=parsed_args.log_level,
            )

            # Create metrics collector and analytics
            # Note: In real usage, these would be passed in or created from config
            metrics_collector = MetricsCollector()
            analytics = SystemAnalytics(metrics_collector)

            # Report generation mode
            if parsed_args.report:
                return self._generate_report_cli(analytics, parsed_args.report, parsed_args.report_hours, parsed_args.output)

            # Dashboard mode
            dashboard = CLIDashboard(metrics_collector, analytics, config)
            dashboard.start_dashboard()

            return 0

        except KeyboardInterrupt:
            print("\\nDashboard interrupted by user")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            return 1

    def _generate_report_cli(self, analytics: SystemAnalytics, format: str, hours: int, output: Optional[str]) -> int:
        """Generate report from CLI"""
        try:
            reporter = SystemReporter(analytics)

            if output is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = f"monitoring_report_{timestamp}.{format}"

            config = ReportConfig(export_format=format, detail_level="detailed", include_charts=True)

            success = reporter.save_report(Path(output), format, hours, config)

            if success:
                print(f"Report generated successfully: {output}")
                return 0
            else:
                print("Failed to generate report")
                return 1

        except Exception as e:
            print(f"Error generating report: {e}")
            return 1


def main() -> int:
    """Main entry point for the CLI dashboard"""
    cli = DashboardCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
