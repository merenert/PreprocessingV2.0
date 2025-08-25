#!/usr/bin/env python3
"""
Address Normalization Monitoring CLI

Command-line interface for the monitoring system with real-time dashboard,
analytics reporting, and system control capabilities.
"""

import sys
import argparse
from pathlib import Path
import json
import time
from datetime import datetime, timedelta

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from addrnorm.monitoring.metrics_collector import MetricsCollector
from addrnorm.monitoring.analytics import SystemAnalytics
from addrnorm.monitoring.reporter import SystemReporter
from addrnorm.monitoring.dashboard import CLIDashboard


def cmd_dashboard(args):
    """Run interactive dashboard"""
    print("🚀 Starting Address Normalization Monitoring Dashboard...")
    print("=" * 60)

    try:
        # Initialize monitoring components
        collector = MetricsCollector()
        analytics = SystemAnalytics(collector)
        dashboard = CLIDashboard(metrics_collector=collector, analytics=analytics)

        if args.live:
            print("📊 Live monitoring mode enabled")
            print("Press Ctrl+C to exit")

            # Start metrics collection
            collector.start_aggregation()

            try:
                while True:
                    # Display current system status
                    print("\033[2J\033[H")  # Clear screen
                    print("🖥️ LIVE DASHBOARD - Address Normalization Monitoring")
                    print("=" * 70)
                    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    # Get current metrics
                    perf_metrics = collector.get_performance_metrics()
                    system_metrics = collector.get_system_metrics()

                    print(f"\n📈 Performance Overview:")
                    print(f"  Processing Time: {perf_metrics.avg_processing_time_ms:.1f}ms avg")
                    print(f"  Throughput: {perf_metrics.throughput_per_second:.1f} req/sec")
                    print(f"  Error Rate: {perf_metrics.error_rate:.2%}")
                    print(f"  Memory Usage: {perf_metrics.memory_usage_mb:.1f}MB")

                    print(f"\n🔧 Method Distribution:")
                    for method, count in system_metrics.method_distribution.items():
                        print(f"  {method}: {count}")

                    print("\n💡 Commands: Ctrl+C to exit")

                    time.sleep(args.refresh_interval)

            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped")
            finally:
                collector.stop_aggregation()
        else:
            print("📊 Single snapshot mode")
            # Show current system snapshot

    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return 1

    return 0


def cmd_analytics(args):
    """Generate analytics report"""
    print("📊 Generating Analytics Report...")

    try:
        # Initialize components
        collector = MetricsCollector()
        analytics = SystemAnalytics(collector)

        # Parse time period
        if args.period.endswith("d"):
            hours = int(args.period[:-1]) * 24
        elif args.period.endswith("h"):
            hours = int(args.period[:-1])
        else:
            hours = 24  # default

        # Generate report
        report = analytics.generate_comprehensive_report(time_window_hours=hours)

        if args.output:
            # Save to file
            output_path = Path(args.output)
            if args.format == "json":
                with open(output_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)
            else:
                # HTML format
                reporter = SystemReporter(analytics)
                html_content = reporter.generate_report(format="html", time_window_hours=hours)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
            print(f"✅ Report saved to: {output_path}")
        else:
            # Print to console
            print("\n📋 Analytics Summary:")
            print(f"  Period: {args.period}")
            print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if "performance_insights" in report:
                print(f"\n🚀 Key Insights:")
                for insight in report["performance_insights"][:3]:
                    print(f"  • {insight.get('description', 'N/A')}")

            recommendations = report.get("recommendations", {})
            high_priority = recommendations.get("high_priority", [])
            if high_priority:
                print(f"\n⚠️ High Priority Actions:")
                for rec in high_priority[:3]:
                    print(f"  🔴 {rec}")

    except Exception as e:
        print(f"❌ Analytics error: {e}")
        return 1

    return 0


def cmd_status(args):
    """Show system status"""
    print("🔍 System Status Check...")

    try:
        collector = MetricsCollector()

        # Get basic metrics
        perf_metrics = collector.get_performance_metrics()
        system_metrics = collector.get_system_metrics()

        print(f"\n📊 Current Status:")
        print(f"  System Uptime: {system_metrics.uptime_seconds:.1f}s")
        print(f"  Processing Time: {perf_metrics.avg_processing_time_ms:.1f}ms")
        print(f"  Error Rate: {perf_metrics.error_rate:.2%}")
        print(f"  Total Processed: {perf_metrics.total_processed}")

        # Health check
        if perf_metrics.error_rate > 0.1:
            print("  🔴 HIGH ERROR RATE")
        elif perf_metrics.avg_processing_time_ms > 200:
            print("  🟡 SLOW PERFORMANCE")
        else:
            print("  ✅ HEALTHY")

    except Exception as e:
        print(f"❌ Status check error: {e}")
        return 1

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Address Normalization Monitoring CLI", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Interactive monitoring dashboard")
    dashboard_parser.add_argument("--live", action="store_true", help="Enable live monitoring mode")
    dashboard_parser.add_argument(
        "--refresh-interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2.0)"
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # Analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Generate analytics report")
    analytics_parser.add_argument("--period", default="24h", help="Analysis period (e.g., 1d, 12h, 30d)")
    analytics_parser.add_argument("--format", choices=["json", "html"], default="json", help="Output format")
    analytics_parser.add_argument("--output", help="Output file path")
    analytics_parser.set_defaults(func=cmd_analytics)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
