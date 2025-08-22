#!/usr/bin/env python3
"""
Turkish Address Normalization CLI Tool

A command-line interface for normalizing Turkish addresses.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

# Import from installed package
try:
    from addrnorm.pipeline import PipelineConfig, create_pipeline
except ImportError:
    # Fallback for development - add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    from addrnorm.pipeline import PipelineConfig, create_pipeline


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""

    parser = argparse.ArgumentParser(
        prog="addrnorm",
        description="Turkish Address Normalization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic normalization to JSONL
  addrnorm normalize --in addresses.txt --out results.jsonl

  # Export to CSV format
  addrnorm normalize --in data.txt --out output.csv --format csv

  # Parallel processing with metrics
  addrnorm normalize --in large_file.txt \\
      --out results.jsonl --jobs 4 --metrics stats.json

  # Single address processing
  addrnorm normalize --address "Atatürk Mahallesi Cumhuriyet Caddesi No:15"

  # Show processing statistics
  addrnorm normalize --in data.txt --out results.jsonl --verbose --stats

For more information and examples, visit: https://github.com/merenert/PreprocessingV2.0
        """,
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize command
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize Turkish addresses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Normalize Turkish addresses using ML and rule-based methods.",
        epilog="""
Input Formats:
  - Text file with one address per line
  - CSV file (first column will be treated as addresses)

Output Formats:
  - jsonl: JSON Lines format (one JSON object per line)
  - csv: Comma-separated values with headers

Processing Methods:
  1. Pattern Matching: Fast regex-based extraction
  2. ML NER: spaCy-based Named Entity Recognition
  3. Rule-based Fallback: Heuristic extraction for edge cases
  4. Geographic Validation: City/district consistency checks
        """,
    )

    # Input options
    input_group = normalize_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--in",
        "--input",
        dest="input_file",
        type=str,
        help="Input file containing addresses (one per line or CSV)",
    )
    input_group.add_argument(
        "--address", type=str, help="Single address to normalize (output to stdout)"
    )

    # Output options
    normalize_parser.add_argument(
        "--out",
        "--output",
        dest="output_file",
        type=str,
        help="Output file path (required for file input, optional for single address)",
    )
    normalize_parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )

    # Processing options
    normalize_parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs (disabled due to ML constraints)",
    )
    normalize_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )

    # Model configuration
    normalize_parser.add_argument(
        "--model-path",
        type=str,
        default="models/turkish_address_ner_improved",
        help="Path to ML model (default: models/turkish_address_ner_improved)",
    )
    normalize_parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Disable ML model (use only patterns and fallback)",
    )
    normalize_parser.add_argument(
        "--no-validation", action="store_true", help="Disable geographic validation"
    )

    # Threshold configuration
    normalize_parser.add_argument(
        "--pattern-threshold",
        type=float,
        default=0.8,
        help="Pattern matching confidence threshold (default: 0.8)",
    )
    normalize_parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.7,
        help="ML model confidence threshold (default: 0.7)",
    )

    # Output options
    normalize_parser.add_argument(
        "--metrics", type=str, help="Output file for processing metrics (JSON format)"
    )
    normalize_parser.add_argument(
        "--stats", action="store_true", help="Show processing statistics"
    )
    normalize_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    normalize_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress all output except errors"
    )
    normalize_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (for single address)",
    )

    # Add version command
    subparsers.add_parser("version", help="Show version information")

    return parser


def validate_args(args) -> None:
    """Validate command line arguments."""

    if args.command == "normalize":
        # Check input file exists
        if args.input_file and not Path(args.input_file).exists():
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)

        # Check output file is specified for file input
        if args.input_file and not args.output_file:
            print(
                "Error: Output file (--out) is required when processing input file",
                file=sys.stderr,
            )
            sys.exit(1)

        # Check conflicting options
        if args.verbose and args.quiet:
            print(
                "Error: --verbose and --quiet cannot be used together", file=sys.stderr
            )
            sys.exit(1)

        # Warn about jobs parameter
        if args.jobs > 1:
            print(
                "Warning: Parallel processing (--jobs) is disabled due to "
                "ML model constraints. Processing sequentially.",
                file=sys.stderr,
            )


def process_single_address(pipeline, address: str, args) -> Dict:
    """Process a single address and return the result."""

    result = pipeline.process_single(address)

    if not result.success:
        if not args.quiet:
            print(f"Error processing address: {result.error}", file=sys.stderr)
        return None

    # Convert to dict for output
    address_dict = result.address_out.to_dict()

    # Add processing metadata if verbose
    if args.verbose:
        address_dict["_processing"] = {
            "method": result.processing_method,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms,
        }

    return address_dict


def process_file_input(pipeline, args) -> Dict:
    """Process addresses from file and return processing statistics."""

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not args.quiet:
        print(f"Processing addresses from: {input_path}")
        print(f"Output format: {args.format}")
        print(f"Writing results to: {output_path}")

    start_time = time.time()

    # Read input file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            if input_path.suffix.lower() == ".csv":
                # For CSV, read first column as addresses
                import csv

                reader = csv.reader(f)
                # Skip header if it exists
                first_row = next(reader, None)
                if first_row and not any(
                    keyword in first_row[0].lower()
                    for keyword in ["adres", "address", "sokak", "mahalle"]
                ):
                    addresses = [first_row[0]]  # Include first row if it's an address
                else:
                    addresses = []

                for row in reader:
                    if row and row[0].strip():
                        addresses.append(row[0].strip())
            else:
                # Text file - one address per line
                addresses = [line.strip() for line in f if line.strip()]

    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    if not addresses:
        print("Error: No addresses found in input file", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(addresses)} addresses to process")

    # Process addresses
    all_results = []
    batch_size = args.batch_size

    for i in range(0, len(addresses), batch_size):
        batch = addresses[i : i + batch_size]
        batch_results = pipeline.process_batch(batch)
        all_results.extend(batch_results)

        if not args.quiet:
            progress = ((i + len(batch)) / len(addresses)) * 100
            print(f"Progress: {progress:.1f}% ({i + len(batch)}/{len(addresses)})")

    # Write output file
    successful_results = [r for r in all_results if r.success]

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if args.format == "jsonl":
                # JSON Lines format
                for result in all_results:
                    if result.success:
                        address_dict = result.address_out.to_dict()
                        if args.verbose:
                            address_dict["_processing"] = {
                                "method": result.processing_method,
                                "confidence": result.confidence,
                                "processing_time_ms": result.processing_time_ms,
                            }
                        f.write(json.dumps(address_dict, ensure_ascii=False) + "\n")
                    else:
                        # Write error record
                        error_record = {
                            "explanation_raw": result.raw_input,
                            "error": result.error,
                            "processing_method": "error",
                        }
                        f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

            elif args.format == "csv":
                # CSV format
                import csv

                from addrnorm.utils.contracts import AddressOut

                writer = csv.writer(f)

                # Write header
                headers = AddressOut.csv_headers()
                if args.verbose:
                    headers.extend(
                        [
                            "processing_method",
                            "processing_confidence",
                            "processing_time_ms",
                        ]
                    )
                writer.writerow(headers)

                # Write data
                for result in all_results:
                    if result.success:
                        row = result.address_out.to_csv_row()
                        if args.verbose:
                            row.extend(
                                [
                                    result.processing_method,
                                    str(result.confidence),
                                    str(result.processing_time_ms),
                                ]
                            )
                        writer.writerow(row)
                    else:
                        # Write error row
                        error_row = [""] * len(headers)
                        error_row[headers.index("explanation_raw")] = result.raw_input
                        if "processing_method" in headers:
                            error_row[headers.index("processing_method")] = "error"
                        writer.writerow(error_row)

    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate statistics
    total_time = time.time() - start_time
    successful = len(successful_results)
    failed = len(all_results) - successful

    if successful > 0:
        avg_confidence = sum(r.confidence for r in successful_results) / successful
        avg_time_ms = sum(r.processing_time_ms for r in successful_results) / successful
    else:
        avg_confidence = 0.0
        avg_time_ms = 0.0

    # Method distribution
    methods = {}
    for result in all_results:
        method = result.processing_method
        methods[method] = methods.get(method, 0) + 1

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "output_format": args.format,
        "total_addresses": len(all_results),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(all_results) if all_results else 0.0,
        "average_confidence": avg_confidence,
        "average_processing_time_ms": avg_time_ms,
        "total_processing_time_seconds": total_time,
        "throughput_addresses_per_second": (
            len(all_results) / total_time if total_time > 0 else 0.0
        ),
        "methods_used": methods,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return stats


def save_metrics(stats: Dict, metrics_file: str) -> None:
    """Save processing metrics to JSON file."""

    try:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(
            f"Warning: Failed to save metrics to {metrics_file}: {e}", file=sys.stderr
        )


def print_stats(stats: Dict, verbose: bool = False) -> None:
    """Print processing statistics to console."""

    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)

    print(f"Total addresses: {stats['total_addresses']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print(f"Average time per address: {stats['average_processing_time_ms']:.1f}ms")
    print(f"Total processing time: {stats['total_processing_time_seconds']:.1f}s")
    print(
        f"Throughput: {stats['throughput_addresses_per_second']:.1f} addresses/second"
    )

    if verbose:
        print("\nMethods used:")
        for method, count in stats["methods_used"].items():
            percentage = (count / stats["total_addresses"]) * 100
            print(f"  {method}: {count} ({percentage:.1f}%)")

    print("=" * 60)


def run_normalize_command(args) -> None:
    """Execute the normalize command."""

    # Configure pipeline
    config = PipelineConfig(
        pattern_threshold_high=args.pattern_threshold,
        ml_confidence_threshold=args.ml_threshold,
        ml_model_path=args.model_path if not args.no_ml else None,
        enable_multiprocessing=False,  # Always disabled due to ML model constraints
        batch_size=args.batch_size,
        enable_validation=not args.no_validation,
        log_level="DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO",
    )

    # Create pipeline
    try:
        if not args.quiet:
            print("Initializing Turkish address normalization pipeline...")

        pipeline = create_pipeline(config)

        if not args.quiet:
            print("Pipeline ready.")

    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    # Process input
    try:
        if args.address:
            # Single address processing
            result = process_single_address(pipeline, args.address, args)

            if result:
                if args.output_file:
                    # Write to file
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        if args.format == "jsonl":
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        elif args.format == "csv":
                            # Convert to AddressOut for CSV export
                            from addrnorm.utils.contracts import AddressOut

                            addr_out = AddressOut(
                                **{
                                    k: v
                                    for k, v in result.items()
                                    if k in AddressOut.model_fields
                                    and k != "_processing"
                                }
                            )
                            import csv

                            writer = csv.writer(f)
                            writer.writerow(AddressOut.csv_headers())
                            writer.writerow(addr_out.to_csv_row())

                    if not args.quiet:
                        print(f"Result written to: {args.output_file}")
                else:
                    # Print to stdout
                    if args.pretty:
                        print(json.dumps(result, ensure_ascii=False, indent=2))
                    else:
                        print(json.dumps(result, ensure_ascii=False))
            else:
                sys.exit(1)

        else:
            # File processing
            stats = process_file_input(pipeline, args)

            if not args.quiet:
                print("\nProcessing complete!")
                print(f"Results written to: {stats['output_file']}")

            # Save metrics if requested
            if args.metrics:
                save_metrics(stats, args.metrics)
                if not args.quiet:
                    print(f"Metrics saved to: {args.metrics}")

            # Show statistics if requested
            if args.stats and not args.quiet:
                print_stats(stats, args.verbose)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


def show_version() -> None:
    """Show version information."""
    print("Turkish Address Normalization Tool")
    print("Version: 2.0.0")
    print("Author: Meren Ertugrul")
    print("Repository: https://github.com/merenert/PreprocessingV2.0")


def main() -> None:
    """Main CLI entry point."""

    parser = create_parser()
    args = parser.parse_args()

    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate arguments
    validate_args(args)

    # Execute command
    if args.command == "normalize":
        run_normalize_command(args)
    elif args.command == "version":
        show_version()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
