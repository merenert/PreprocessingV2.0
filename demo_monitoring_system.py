"""
Demo Script - Address Normalization Monitoring System

Demonstrates the complete monitoring system including:
- Real-time metrics collection
- Performance analytics
- Report generation
- CLI dashboard simulation

This script simulates the monitoring system with sample data to show
all features without requiring the full address normalization system.
"""

import time
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import monitoring components
from src.addrnorm.monitoring import (
    MetricsCollector,
    SystemAnalytics,
    SystemReporter,
    CLIDashboard,
    DashboardConfig,
    ReportConfig,
    MetricEvent,
    MetricType,
    ProcessingMethod,
)


class MonitoringDemo:
    """
    Demo class to simulate monitoring system functionality
    """

    def __init__(self):
        """Initialize the monitoring demo"""
        print("🔧 Initializing Address Normalization Monitoring System...")

        # Create monitoring components
        self.metrics_collector = MetricsCollector(
            buffer_size=1000, aggregation_interval_seconds=30, enable_persistence=False  # Disable for demo
        )

        self.analytics = SystemAnalytics(self.metrics_collector)
        self.reporter = SystemReporter(self.analytics)

        # Simulation control
        self.simulation_running = False
        self.simulation_thread = None

        print("✅ Monitoring system initialized successfully!")

    def simulate_processing_events(self, duration_seconds: int = 60):
        """
        Simulate address processing events for demonstration

        Args:
            duration_seconds: How long to simulate events
        """
        print(f"📊 Starting simulation for {duration_seconds} seconds...")

        start_time = time.time()
        self.simulation_running = True

        # Simulate different patterns and processing methods
        patterns = ["TR_STANDARD_001", "TR_POSTAL_002", "TR_URBAN_003", "TR_RURAL_004", "TR_COMPLEX_005"]

        methods = list(ProcessingMethod)

        while self.simulation_running and (time.time() - start_time) < duration_seconds:
            try:
                # Generate random processing event
                pattern_id = random.choice(patterns)
                method = random.choice(methods)

                # Simulate realistic processing metrics
                success = random.random() > 0.1  # 90% success rate
                confidence = random.uniform(0.6, 0.95) if success else random.uniform(0.2, 0.6)
                processing_time = random.uniform(50, 200) if success else random.uniform(200, 800)

                # Simulate errors occasionally
                error_type = None
                if not success:
                    error_types = ["regex_mismatch", "validation_failed", "timeout", "format_error"]
                    error_type = random.choice(error_types)

                # Record the event
                self.metrics_collector.record_processing(
                    method=method,
                    pattern_id=pattern_id,
                    success=success,
                    confidence=confidence,
                    processing_time_ms=processing_time,
                    address_length=random.randint(20, 150),
                    components_extracted=random.randint(3, 8),
                    error_type=error_type,
                    metadata={"geographic_validation": random.random() > 0.3, "similarity_score": random.uniform(0.7, 1.0)},
                )

                # Variable delay to simulate realistic load
                delay = random.uniform(0.1, 0.5)
                time.sleep(delay)

            except Exception as e:
                print(f"❌ Error in simulation: {e}")
                break

        self.simulation_running = False
        print("🔄 Simulation completed")

    def start_simulation_background(self, duration_seconds: int = 120):
        """Start simulation in background thread"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            print("⚠️ Simulation already running")
            return

        self.simulation_thread = threading.Thread(target=self.simulate_processing_events, args=(duration_seconds,))
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def stop_simulation(self):
        """Stop the background simulation"""
        self.simulation_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2)
        print("🛑 Simulation stopped")

    def show_live_metrics(self):
        """Display current metrics"""
        print("\n" + "=" * 60)
        print("📈 LIVE METRICS")
        print("=" * 60)

        # Performance metrics
        perf_metrics = self.metrics_collector.get_performance_metrics()
        print(f"📊 Performance Metrics:")
        print(f"   Total Processed: {perf_metrics.total_processed:,}")
        print(f"   Avg Processing Time: {perf_metrics.avg_processing_time_ms:.1f}ms")
        print(f"   Throughput: {perf_metrics.throughput_per_second:.1f}/sec")
        print(f"   Error Rate: {perf_metrics.error_rate:.1%}")
        print(f"   Memory Usage: {perf_metrics.memory_usage_mb:.1f}MB")

        # System metrics
        sys_metrics = self.metrics_collector.get_system_metrics()
        print(f"\n🔧 System Metrics:")
        print(f"   Uptime: {sys_metrics.uptime_seconds/3600:.1f} hours")
        print(f"   Geographic Validation Rate: {sys_metrics.geographic_validation_rate:.1%}")

        # Top patterns
        pattern_metrics = self.metrics_collector.get_pattern_metrics()
        if pattern_metrics:
            print(f"\n📋 Top Patterns:")
            sorted_patterns = sorted(pattern_metrics.items(), key=lambda x: x[1].usage_count, reverse=True)[:3]

            for i, (pattern_id, metrics) in enumerate(sorted_patterns, 1):
                print(
                    f"   {i}. {pattern_id}: {metrics.usage_count} uses, "
                    f"{metrics.success_rate:.1%} success, "
                    f"{metrics.avg_confidence:.3f} confidence"
                )

    def generate_analytics_report(self):
        """Generate and display analytics report"""
        print("\n" + "=" * 60)
        print("📊 ANALYTICS REPORT")
        print("=" * 60)

        try:
            # Generate comprehensive report
            report_data = self.analytics.generate_comprehensive_report(time_window_hours=1)

            # Display key insights
            metadata = report_data.get("report_metadata", {})
            print(f"📅 Report generated at: {metadata.get('generated_at', 'Unknown')}")
            print(f"📈 Events analyzed: {metadata.get('total_events_analyzed', 0):,}")

            # Performance insights
            insights = report_data.get("performance_insights", [])
            if insights:
                print(f"\n🔍 Performance Insights ({len(insights)} found):")
                for insight in insights[:3]:  # Show top 3
                    level_emoji = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(insight["impact_level"], "📊")
                    print(f"   {level_emoji} {insight['insight_type'].replace('_', ' ').title()}")
                    print(f"     {insight['description']}")
            else:
                print("\n✅ No performance issues detected")

            # Recommendations
            recommendations = report_data.get("recommendations", {})
            high_priority = recommendations.get("high_priority", [])
            if high_priority:
                print(f"\n⚡ High Priority Recommendations:")
                for rec in high_priority[:2]:  # Show top 2
                    print(f"   • {rec}")

            # Pattern analysis
            pattern_analysis = report_data.get("pattern_analysis", [])
            if pattern_analysis:
                print(f"\n📋 Pattern Analysis ({len(pattern_analysis)} patterns):")
                top_patterns = sorted(pattern_analysis, key=lambda x: x["performance_score"], reverse=True)[:3]
                for pattern in top_patterns:
                    print(f"   🏆 {pattern['pattern_id']}: Score {pattern['performance_score']:.3f}")

            return report_data

        except Exception as e:
            print(f"❌ Error generating analytics report: {e}")
            return None

    def generate_html_report(self, filename: str = None):
        """Generate HTML report file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_demo_report_{timestamp}.html"

        try:
            print(f"\n📄 Generating HTML report: {filename}")

            config = ReportConfig(export_format="html", include_charts=True, detail_level="detailed")

            report_content = self.reporter.generate_report("html", time_window_hours=1, config=config)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            print(f"✅ HTML report saved: {filename}")
            print(f"🌐 Open {filename} in your browser to view the report")

            return filename

        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            return None

    def run_demo_sequence(self):
        """Run the complete demo sequence"""
        print("\n🚀 Starting Address Normalization Monitoring Demo")
        print("=" * 60)

        try:
            # Step 1: Start simulation
            print("\n1️⃣ Starting background simulation...")
            self.start_simulation_background(duration_seconds=45)

            # Step 2: Let it run for a bit
            print("⏳ Letting simulation run for 15 seconds...")
            time.sleep(15)

            # Step 3: Show live metrics
            print("\n2️⃣ Displaying live metrics...")
            self.show_live_metrics()

            # Step 4: Wait more for more data
            print("\n⏳ Collecting more data for 15 seconds...")
            time.sleep(15)

            # Step 5: Generate analytics
            print("\n3️⃣ Generating analytics report...")
            report_data = self.generate_analytics_report()

            # Step 6: Show updated metrics
            print("\n4️⃣ Updated live metrics...")
            self.show_live_metrics()

            # Step 7: Generate HTML report
            print("\n5️⃣ Generating HTML report...")
            html_file = self.generate_html_report()

            # Step 8: Wait for simulation to finish
            print("\n⏳ Waiting for simulation to complete...")
            time.sleep(15)

            # Step 9: Final metrics
            print("\n6️⃣ Final metrics summary...")
            self.show_live_metrics()

            # Step 10: Export final data
            print("\n7️⃣ Exporting final metrics...")
            final_metrics = self.metrics_collector.export_metrics("json")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = f"monitoring_demo_metrics_{timestamp}.json"

            with open(metrics_file, "w", encoding="utf-8") as f:
                f.write(final_metrics)

            print(f"✅ Metrics exported to: {metrics_file}")

            print("\n🎉 Demo completed successfully!")
            print("\n📋 Demo Summary:")
            print(f"   • Simulated address processing events")
            print(f"   • Collected real-time metrics")
            print(f"   • Generated performance analytics")
            print(f"   • Created HTML report: {html_file or 'Failed'}")
            print(f"   • Exported metrics: {metrics_file}")

        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        finally:
            # Cleanup
            self.stop_simulation()
            self.metrics_collector.stop_aggregation()
            print("\n🧹 Demo cleanup completed")

    def run_interactive_dashboard(self):
        """Run the interactive CLI dashboard"""
        print("\n🖥️ Starting Interactive CLI Dashboard...")
        print("💡 This will start a live dashboard with auto-refresh")
        print("⚠️ Press Ctrl+C to exit the dashboard")

        try:
            # Start simulation
            self.start_simulation_background(duration_seconds=300)  # 5 minutes

            # Create and start dashboard
            config = DashboardConfig(refresh_interval=3, auto_refresh=True, show_patterns=True, show_alerts=True)

            dashboard = CLIDashboard(self.metrics_collector, self.analytics, config)
            dashboard.start_dashboard()

        except KeyboardInterrupt:
            print("\n🛑 Dashboard interrupted by user")
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")
        finally:
            self.stop_simulation()


def main():
    """Main demo function"""
    print("🏠 Address Normalization Monitoring System Demo")
    print("=" * 60)
    print("🎯 This demo shows the monitoring system capabilities:")
    print("   • Real-time metrics collection")
    print("   • Performance analytics")
    print("   • Report generation")
    print("   • CLI dashboard")
    print()

    # Create demo instance
    demo = MonitoringDemo()

    # Ask user for demo type
    print("Choose demo type:")
    print("1. 🚀 Automated Demo Sequence (recommended)")
    print("2. 🖥️ Interactive CLI Dashboard")
    print("3. 📊 Quick Metrics Test")

    try:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1" or choice == "":
            demo.run_demo_sequence()
        elif choice == "2":
            demo.run_interactive_dashboard()
        elif choice == "3":
            print("\n📊 Running quick metrics test...")
            demo.start_simulation_background(30)
            time.sleep(10)
            demo.show_live_metrics()
            time.sleep(10)
            demo.show_live_metrics()
            demo.stop_simulation()
        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

    print("\n👋 Demo finished. Thank you!")


if __name__ == "__main__":
    main()
