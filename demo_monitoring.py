"""
Monitoring System Demo

Comprehensive demonstration of the monitoring and analytics system
for address normalization performance tracking.
"""

import sys
import os
from pathlib import Path
import json
import time
import random
from datetime import datetime, timedelta

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from addrnorm.monitoring.metrics_collector import MetricsCollector, MetricEvent, MetricType, ProcessingMethod
from addrnorm.monitoring.analytics import SystemAnalytics
from addrnorm.monitoring.reporter import SystemReporter
from addrnorm.monitoring.dashboard import CLIDashboard


def create_sample_processing_events():
    """Create sample processing events for demo"""
    events = []

    # Sample addresses with different processing outcomes
    sample_addresses = [
        ("İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15", "pattern_primary", 0.95, True),
        ("Amorium Hotel karşısı", "pattern_fallback", 0.72, True),
        ("Ankara çankaya", "ml_primary", 0.68, True),
        ("belirsiz adres xyz", "fallback", 0.35, False),
        ("İzmir Alsancak Cumhuriyet Bulvarı 123", "pattern_primary", 0.88, True),
        ("McDonald's yanı", "pattern_secondary", 0.65, True),
        ("invalid address format", "fallback", 0.25, False),
        ("Bursa Nilüfer Özlüce Mahallesi", "ml_primary", 0.79, True),
        ("123 xyz street", "fallback", 0.30, False),
        ("Antalya Muratpaşa Lara Caddesi No: 45", "hybrid", 0.92, True),
    ]

    # Generate events for the last 7 days
    base_time = datetime.now() - timedelta(days=7)

    for i in range(500):  # Generate 500 events
        # Random time within last 7 days
        event_time = base_time + timedelta(seconds=random.randint(0, 7 * 24 * 3600))

        # Pick random address
        address, method, confidence, success = random.choice(sample_addresses)

        # Add some randomization
        confidence += random.uniform(-0.1, 0.1)
        confidence = max(0.0, min(1.0, confidence))

        processing_time = random.uniform(10, 200)  # 10-200ms

        # Method mapping
        method_map = {
            "pattern_primary": ProcessingMethod.PATTERN_PRIMARY,
            "pattern_fallback": ProcessingMethod.PATTERN_SECONDARY,
            "pattern_secondary": ProcessingMethod.PATTERN_SECONDARY,
            "ml_primary": ProcessingMethod.ML_PRIMARY,
            "hybrid": ProcessingMethod.HYBRID,
            "fallback": ProcessingMethod.FALLBACK,
        }

        event = MetricEvent(
            timestamp=event_time,
            metric_type=MetricType.PERFORMANCE,
            method=method_map.get(method, ProcessingMethod.FALLBACK),
            pattern_id=f"pattern_{random.randint(1, 20)}",
            success=success,
            confidence=confidence,
            processing_time_ms=processing_time,
            address_length=len(address),
            components_extracted=random.randint(2, 6),
            metadata={
                "address": address,
                "geographic_region": random.choice(["istanbul", "ankara", "izmir", "bursa", "antalya"]),
            },
        )
        events.append(event)

    return sorted(events, key=lambda x: x.timestamp)


def demo_metrics_collection():
    """Demo metrics collection functionality"""
    print("=" * 60)
    print("📊 METRICS COLLECTION DEMO")
    print("=" * 60)

    # Initialize metrics collector
    collector = MetricsCollector(buffer_size=1000, aggregation_interval_seconds=5, retention_days=7)

    # Start collection
    collector.start_aggregation()
    print("✅ Metrics collector started")

    # Generate some sample events
    sample_events = create_sample_processing_events()
    print(f"📄 Generated {len(sample_events)} sample events")

    # Record events (simulate batch processing)
    print("🔄 Recording events...")
    for i, event in enumerate(sample_events[:50]):  # Record first 50 for demo
        collector.record_event(event)

        if (i + 1) % 10 == 0:
            print(f"  Recorded {i + 1} events...")

    # Get current metrics
    perf_metrics = collector.get_performance_metrics()
    system_metrics = collector.get_system_metrics()

    # Calculate success rate and confidence manually from events
    events_list = list(collector._events)
    total_events = len(events_list)

    if total_events > 0:
        success_count = sum(1 for event in events_list if event.success)
        success_rate = success_count / total_events
        avg_confidence = sum(event.confidence for event in events_list) / total_events
    else:
        success_rate = 0.0
        avg_confidence = 0.0

    print("\n📈 Current Metrics Summary:")
    print(f"  Total Events: {total_events}")
    print(f"  Avg Processing Time: {perf_metrics.avg_processing_time_ms:.2f}ms")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Avg Confidence: {avg_confidence:.3f}")

    # Method distribution
    print(f"\n🔧 Method Distribution:")
    for method, count in system_metrics.method_distribution.items():
        if total_events > 0:
            percentage = (count / total_events) * 100
            print(f"  {method}: {count} ({percentage:.1f}%)")

    collector.stop_aggregation()
    print("\n✅ Metrics collection demo complete")
    return collector


def demo_analytics():
    """Demo analytics functionality"""
    print("\n" + "=" * 60)
    print("📊 SYSTEM ANALYTICS DEMO")
    print("=" * 60)

    # Initialize analytics with a dummy collector for demo
    demo_collector = MetricsCollector()
    analytics = SystemAnalytics(demo_collector)

    # Generate sample data for analysis
    events = create_sample_processing_events()

    # Add events to demo collector to simulate real data
    print("📊 Preparing analytics data...")
    for event in events:
        demo_collector.record_event(event)

    print(f"📈 Analyzing {len(events)} processing events...")

    # Generate comprehensive analytics report
    print("\n🚀 Comprehensive Analytics Report:")
    report = analytics.generate_comprehensive_report(time_window_hours=24)

    if "error" in report:
        print(f"⚠️ Analytics error: {report['error']}")
        return analytics

    # Display key findings
    print(f"\n📊 Summary:")
    summary = report.get("summary", {})
    print(f"  Report Period: {summary.get('time_window', 'N/A')}")
    print(f"  Total Events: {summary.get('total_events', 0)}")
    print(f"  Analysis Timestamp: {summary.get('generated_at', 'N/A')}")

    # Performance insights
    if "performance_insights" in report:
        print(f"\n🚀 Performance Insights:")
        insights = report["performance_insights"][:3]  # Show top 3
        for insight in insights:
            print(f"  • {insight.get('description', 'N/A')}")
            if insight.get("recommendation"):
                print(f"    → {insight['recommendation']}")

    # Pattern analysis
    if "pattern_analysis" in report:
        print(f"\n🔍 Pattern Analysis:")
        patterns = report["pattern_analysis"][:3]  # Show top 3
        for pattern in patterns:
            print(f"  Pattern {pattern.get('pattern_id', 'N/A')}:")
            print(f"    Usage: {pattern.get('usage_count', 0)} times")
            print(f"    Success Rate: {pattern.get('success_rate', 0):.2%}")
            print(f"    Optimization Potential: {pattern.get('optimization_potential', 0):.1%}")

    # Recommendations
    recommendations = report.get("recommendations", {})
    high_priority = recommendations.get("high_priority", [])

    if high_priority:
        print(f"\n⚠️ High Priority Recommendations:")
        for rec in high_priority[:3]:  # Show top 3
            print(f"  🔴 {rec}")

    # System metrics
    if "system_metrics" in report:
        system_metrics = report["system_metrics"]
        if "performance" in system_metrics:
            perf = system_metrics["performance"]
            print(f"\n📈 Performance Metrics:")
            print(f"  Avg Processing Time: {perf.get('avg_processing_time_ms', 0):.1f}ms")
            print(f"  Throughput: {perf.get('throughput_per_second', 0):.1f} req/sec")
            print(f"  Error Rate: {perf.get('error_rate', 0):.2%}")

    print("✅ Analytics demo complete")
    return analytics


def demo_reporting():
    """Demo reporting functionality"""
    print("\n" + "=" * 60)
    print("📋 REPORTING DEMO")
    print("=" * 60)

    # Initialize reporter
    dummy_analytics = SystemAnalytics(MetricsCollector())
    reporter = SystemReporter(dummy_analytics)

    # Generate sample data
    events = create_sample_processing_events()

    # Add events to analytics (simulate data collection)
    print("📊 Preparing reporting data...")
    for event in events:
        dummy_analytics.metrics_collector.record_event(event)

    print("📊 Generating reports...")

    # Generate comprehensive report
    print("  📈 System Report...")
    try:
        report_content = reporter.generate_report(format="html", time_window_hours=24)
        print(f"    ✅ HTML Report generated ({len(report_content)} characters)")
    except Exception as e:
        print(f"    ⚠️ HTML Report error: {e}")

    # JSON report
    print("  � JSON Report...")
    try:
        json_report = reporter.generate_report(format="json", time_window_hours=24)
        print(f"    ✅ JSON Report generated ({len(json_report)} characters)")
    except Exception as e:
        print(f"    ⚠️ JSON Report error: {e}")

    # CSV report
    print("  � CSV Report...")
    try:
        csv_report = reporter.generate_report(format="csv", time_window_hours=24)
        print(f"    ✅ CSV Report generated ({len(csv_report)} characters)")
    except Exception as e:
        print(f"    ⚠️ CSV Report error: {e}")

    # Create monitoring_reports directory and save sample files
    reports_dir = Path("monitoring_reports")
    reports_dir.mkdir(exist_ok=True)

    print("💾 Exporting sample reports...")

    # Save sample reports
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # HTML report
        html_file = reports_dir / f"system_report_{timestamp}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(report_content if "report_content" in locals() else "<html><body><h1>Sample Report</h1></body></html>")
        print(f"  ✅ HTML Report: {html_file}")

        # JSON report
        json_file = reports_dir / f"system_metrics_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            if "json_report" in locals():
                f.write(json_report)
            else:
                json.dump({"sample": "data", "timestamp": timestamp}, f, indent=2)
        print(f"  ✅ JSON Report: {json_file}")

        # CSV report
        csv_file = reports_dir / f"performance_data_{timestamp}.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            if "csv_report" in locals():
                f.write(csv_report)
            else:
                f.write("metric,value,timestamp\nprocessing_time,100.5,2025-08-24\nsuccess_rate,85.2,2025-08-24\n")
        print(f"  ✅ CSV Report: {csv_file}")

    except Exception as e:
        print(f"  ⚠️ Export error: {e}")

    print("✅ Reporting demo complete")
    return reporter


def demo_dashboard():
    """Demo CLI dashboard functionality"""
    print("\n" + "=" * 60)
    print("🖥️ CLI DASHBOARD DEMO")
    print("=" * 60)

    # Create temporary analytics and collector for dashboard
    from addrnorm.monitoring.analytics import SystemAnalytics

    collector = MetricsCollector()
    analytics = SystemAnalytics(collector)

    # Generate some sample data
    events = create_sample_processing_events()
    for event in events:
        collector.record_event(event)

    # Initialize dashboard with required parameters
    dashboard = CLIDashboard(metrics_collector=collector, analytics=analytics)

    print("🎮 Dashboard initialized")
    print("📊 Displaying current system status...")

    # Generate some sample metrics for display
    events = create_sample_processing_events()

    # Simulate real-time data for dashboard
    print("\n" + "-" * 50)
    print("📈 LIVE SYSTEM METRICS")
    print("-" * 50)

    # Recent activity summary
    recent_events = events[-20:]  # Last 20 events
    success_count = sum(1 for e in recent_events if e.success)
    avg_confidence = sum(e.confidence for e in recent_events) / len(recent_events)
    avg_processing_time = sum(e.processing_time_ms for e in recent_events) / len(recent_events)

    print(f"🔄 Recent Activity (Last 20 requests):")
    print(f"  Success Rate: {success_count}/{len(recent_events)} ({success_count/len(recent_events):.1%})")
    print(f"  Avg Confidence: {avg_confidence:.3f}")
    print(f"  Avg Processing Time: {avg_processing_time:.1f}ms")

    # Method distribution
    methods = {}
    for event in recent_events:
        method = event.method.value
        methods[method] = methods.get(method, 0) + 1

    print(f"\n🔧 Method Distribution:")
    for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int((count / len(recent_events)) * 20)
        print(f"  {method:15} {count:2} │{bar:<20}│ {count/len(recent_events):.1%}")

    # Geographic distribution
    regions = {}
    for event in recent_events:
        region = event.metadata.get("geographic_region", "unknown")
        regions[region] = regions.get(region, 0) + 1

    print(f"\n🌍 Geographic Distribution:")
    for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True):
        bar = "▓" * int((count / len(recent_events)) * 15)
        print(f"  {region:10} {count:2} │{bar:<15}│ {count/len(recent_events):.1%}")

    # Performance alerts
    print(f"\n⚠️ System Alerts:")
    if avg_confidence < 0.7:
        print("  🔴 LOW CONFIDENCE: Average confidence below threshold (0.7)")
    if avg_processing_time > 150:
        print("  🟡 SLOW PROCESSING: Average processing time above 150ms")
    if success_count / len(recent_events) < 0.8:
        print("  🔴 LOW SUCCESS RATE: Success rate below 80%")

    if avg_confidence >= 0.7 and avg_processing_time <= 150 and success_count / len(recent_events) >= 0.8:
        print("  ✅ All systems operating normally")

    print("\n✅ Dashboard demo complete")
    print("💡 In production, use: 'addrnorm dashboard --live' for real-time monitoring")

    return dashboard


def demo_integration():
    """Demo integration with address normalization"""
    print("\n" + "=" * 60)
    print("🔗 INTEGRATION DEMO")
    print("=" * 60)

    # Simulate integration with main address normalization system
    collector = MetricsCollector()
    collector.start_aggregation()

    print("🔄 Simulating address processing with monitoring...")

    # Sample addresses to process
    test_addresses = [
        "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
        "Amorium Hotel karşısı",
        "Ankara Çankaya belirsiz adres",
        "İzmir Alsancak Cumhuriyet Bulvarı 123",
    ]

    for i, address in enumerate(test_addresses):
        print(f"\n📍 Processing: {address}")

        # Simulate processing
        start_time = time.time()

        # Mock normalization result
        if "belirsiz" in address:
            success = False
            confidence = random.uniform(0.2, 0.4)
            method = "fallback"
        else:
            success = True
            confidence = random.uniform(0.7, 0.95)
            method = random.choice(["pattern_primary", "ml_primary", "hybrid"])

        processing_time = (time.time() - start_time) * 1000 + random.uniform(10, 100)

        # Record metrics
        method_map = {
            "pattern_primary": ProcessingMethod.PATTERN_PRIMARY,
            "pattern_fallback": ProcessingMethod.PATTERN_SECONDARY,
            "pattern_secondary": ProcessingMethod.PATTERN_SECONDARY,
            "ml_primary": ProcessingMethod.ML_PRIMARY,
            "hybrid": ProcessingMethod.HYBRID,
            "fallback": ProcessingMethod.FALLBACK,
        }

        event = MetricEvent(
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
            method=method_map.get(method, ProcessingMethod.FALLBACK),
            pattern_id=f"pattern_{random.randint(1, 10)}",
            success=success,
            confidence=confidence,
            processing_time_ms=processing_time,
            address_length=len(address),
            components_extracted=random.randint(2, 6),
            metadata={"address": address},
        )

        collector.record_event(event)

        # Show result
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} | Method: {method} | Confidence: {confidence:.3f} | Time: {processing_time:.1f}ms")

    # Get final metrics
    perf_metrics = collector.get_performance_metrics()
    events_list = list(collector._events)
    total_events = len(events_list)

    if total_events > 0:
        success_count = sum(1 for event in events_list if event.success)
        success_rate = success_count / total_events
        avg_confidence = sum(event.confidence for event in events_list) / total_events
    else:
        success_rate = 0.0
        avg_confidence = 0.0

    print(f"\n📊 Session Summary:")
    print(f"  Processed: {total_events} addresses")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Avg Confidence: {avg_confidence:.3f}")
    print(f"  Avg Processing Time: {perf_metrics.avg_processing_time_ms:.1f}ms")

    collector.stop_aggregation()
    print("✅ Integration demo complete")


def main():
    """Run all monitoring system demos"""
    print("🚀 MONITORING SYSTEM COMPREHENSIVE DEMO")
    print("=" * 80)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Core monitoring demos
        collector = demo_metrics_collection()
        analytics = demo_analytics()
        reporter = demo_reporting()
        dashboard = demo_dashboard()
        demo_integration()

        print("\n" + "=" * 80)
        print("✅ ALL MONITORING DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("📁 Generated files:")
        print("  • monitoring_reports/ - System reports in JSON/HTML/CSV")
        print("  • Metrics stored in memory for real-time access")
        print()
        print("🎮 Next steps:")
        print("  • Run 'python -m addrnorm.monitoring.dashboard --live' for real-time monitoring")
        print("  • Use reporting API for automated report generation")
        print("  • Integrate metrics collection into production normalization pipeline")
        print()

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
