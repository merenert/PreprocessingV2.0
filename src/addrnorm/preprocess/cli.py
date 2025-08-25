#!/usr/bin/env python3
"""
CLI interface for the address preprocessing module with enhanced output support.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    # Try relative import first (when run as module)
    from ..preprocess import preprocess
    from ..utils.contracts import AddressOut, ExplanationParsed, MethodEnum
    from ..integration.hybrid import HybridAddressNormalizer, IntegrationConfig
except ImportError:
    # Fall back to absolute import (when run directly)
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from addrnorm.preprocess import preprocess
    from addrnorm.utils.contracts import AddressOut, ExplanationParsed, MethodEnum
    from addrnorm.integration.hybrid import HybridAddressNormalizer, IntegrationConfig


def process_file(input_file: str, output_file: str = None, enhanced_output: bool = False, use_hybrid: bool = False):
    """Process addresses from input file and save to output file."""
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            addresses = [line.strip() for line in f if line.strip()]

        results = []

        # Initialize hybrid normalizer if requested
        if use_hybrid:
            config = IntegrationConfig(enhanced_output=enhanced_output)
            normalizer = HybridAddressNormalizer(config)

        for raw_address in addresses:

            if use_hybrid:
                # Use hybrid ML + pattern approach
                result = normalizer.normalize(raw_address, enhanced_output=enhanced_output)

                if enhanced_output:
                    # Enhanced output format
                    result_dict = result.to_dict() if hasattr(result, "to_dict") else result.__dict__
                else:
                    # Legacy format from hybrid
                    result_dict = result.model_dump() if hasattr(result, "model_dump") else result.__dict__

                results.append(result_dict)
            else:
                # Legacy preprocessing only
                processed = preprocess(raw_address)

                # Create output contract
                explanation = ExplanationParsed(
                    confidence=0.85,  # Default confidence for preprocessing
                    method=MethodEnum.PATTERN,
                    warnings=[],
                )

                normalized_output = AddressOut(
                    explanation_raw=raw_address,
                    explanation_parsed=explanation,
                    normalized_address=processed["text"],
                )

                results.append(normalized_output.model_dump())

        # Output results
        if output_file:
            output_path = Path(output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Processed {len(results)} addresses, saved to '{output_file}'")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))

        # Show processing statistics if hybrid was used
        if use_hybrid:
            stats = normalizer.get_performance_stats()
            if stats["total_processed"] > 0:
                print(f"\n📊 Processing Statistics:", file=sys.stderr)
                print(f"   Total processed: {stats['total_processed']}", file=sys.stderr)
                print(f"   Success rate: {stats['success_rate']:.1%}", file=sys.stderr)
                print(f"   Average time: {stats.get('avg_processing_time', 0)*1000:.1f}ms", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        return False


def process_single_address(address: str, enhanced_output: bool = False, use_hybrid: bool = False):
    """Process a single address and return the result."""

    if use_hybrid:
        # Use hybrid ML + pattern approach
        config = IntegrationConfig(enhanced_output=enhanced_output)
        normalizer = HybridAddressNormalizer(config)
        result = normalizer.normalize(address, enhanced_output=enhanced_output)

        if enhanced_output:
            return result.to_dict() if hasattr(result, "to_dict") else result.__dict__
        else:
            return result.model_dump() if hasattr(result, "model_dump") else result.__dict__
    else:
        # Legacy preprocessing
        processed = preprocess(address)
        explanation = ExplanationParsed(confidence=0.85, method=MethodEnum.PATTERN, warnings=[])

        result = AddressOut(
            explanation_raw=address,
            explanation_parsed=explanation,
            normalized_address=processed["text"],
        )

        return result.model_dump()


def _add_processing_args(parser):
    """Add processing arguments to a parser"""
    parser.add_argument("--input", "-i", help="Input file containing raw addresses (one per line)")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin instead of file")
    parser.add_argument("--text", "-t", help="Process a single text string")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid ML + pattern processing (default: pattern only)")
    parser.add_argument(
        "--enhanced-output", action="store_true", help="Use enhanced output format with confidence scores and quality metrics"
    )
    parser.add_argument("--format", choices=["json", "compact"], default="json", help="Output format (default: json)")


def _handle_monitoring_commands(args):
    """Handle monitoring-related commands"""
    try:
        if args.command == "monitor":
            from ..cli.monitoring import cmd_monitor

            cmd_monitor(args)
        elif args.command == "analytics":
            from ..cli.monitoring import cmd_analytics

            cmd_analytics(args)
    except ImportError as e:
        print(f"Error: Monitoring functionality not available: {e}")
        print("Make sure all monitoring dependencies are installed.")
        return False
    except Exception as e:
        print(f"Error in monitoring command: {e}")
        return False
    return True


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Turkish Address Preprocessing CLI with Enhanced Output Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing
  %(prog)s --input data/examples/raw_addresses_tr.txt --output processed.json

  # With hybrid ML + pattern processing
  %(prog)s --input addresses.txt --hybrid --enhanced-output

  # Process single address with enhanced output
  %(prog)s --text "Kızılay Mahallesi Ankara" --hybrid --enhanced-output

  # Read from stdin with compact output
  echo "İstanbul Kadıköy Moda Mah." | %(prog)s --stdin --hybrid

  # Monitoring commands
  %(prog)s monitor --live              # Real-time dashboard
  %(prog)s analytics --period 30d     # Analytics report
  %(prog)s analytics --optimize-thresholds  # Threshold optimization
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main processing command (default)
    process_parser = subparsers.add_parser("process", help="Process addresses (default)")
    _add_processing_args(process_parser)

    # Monitoring commands
    monitor_parser = subparsers.add_parser("monitor", help="Real-time monitoring dashboard")
    monitor_parser.add_argument("--live", action="store_true", help="Enable live monitoring mode")

    analytics_parser = subparsers.add_parser("analytics", help="Generate analytics reports")
    analytics_parser.add_argument("--period", help="Analysis period (e.g., 30d, 24h)", default="30d")
    analytics_parser.add_argument("--output", help="Output file path (JSON or HTML)")
    analytics_parser.add_argument("--pattern-id", help="Specific pattern to analyze")
    analytics_parser.add_argument("--detailed", action="store_true", help="Include detailed metrics")
    analytics_parser.add_argument(
        "--optimize-thresholds", action="store_true", help="Include threshold optimization recommendations"
    )

    # Legacy direct arguments (for backward compatibility)
    _add_processing_args(parser)

    args = parser.parse_args()

    # Handle monitoring commands
    if args.command in ["monitor", "analytics"]:
        return _handle_monitoring_commands(args)

    # Handle processing (original functionality)
    if args.text:
        # Process single text
        result = process_single_address(args.text, args.enhanced_output, args.hybrid)

        if args.format == "compact" and args.hybrid and args.enhanced_output:
            # Show compact format for enhanced output
            try:
                config = IntegrationConfig(enhanced_output=True)
                normalizer = HybridAddressNormalizer(config)
                enhanced_result = normalizer.normalize(args.text, enhanced_output=True)
                compact_result = normalizer.enhanced_formatter.format_compact(enhanced_result)
                print(json.dumps(compact_result, ensure_ascii=False, indent=2))
            except:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.stdin:
        # Read from stdin
        addresses = [line.strip() for line in sys.stdin if line.strip()]
        results = []

        if args.hybrid:
            # Use hybrid processing
            config = IntegrationConfig(enhanced_output=args.enhanced_output)
            normalizer = HybridAddressNormalizer(config)

            for raw_address in addresses:
                result = normalizer.normalize(raw_address, enhanced_output=args.enhanced_output)

                if args.enhanced_output:
                    result_dict = result.to_dict() if hasattr(result, "to_dict") else result.__dict__
                else:
                    result_dict = result.model_dump() if hasattr(result, "model_dump") else result.__dict__

                results.append(result_dict)
        else:
            # Legacy processing
            for raw_address in addresses:
                result = process_single_address(raw_address, args.enhanced_output, args.hybrid)
                results.append(result)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))

    elif args.input:
        # Process input file
        success = process_file(args.input, args.output, args.enhanced_output, args.hybrid)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
