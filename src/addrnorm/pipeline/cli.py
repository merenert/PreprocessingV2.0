"""
Command-line interface for Turkish address normalization pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

from .run import AddressNormalizationPipeline, PipelineConfig


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Turkish Address Normalization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single address
  python -m addrnorm.pipeline.cli "Atatürk Mahallesi Cumhuriyet Caddesi No:15"

  # Process from file
  python -m addrnorm.pipeline.cli --input addresses.txt --output results.jsonl

  # Batch processing with custom config
  python -m addrnorm.pipeline.cli --input data.txt \\
      --output out.jsonl --workers 8 --batch-size 50
        """,
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("address", nargs="?", help="Single address to process")
    group.add_argument(
        "--input", "-i", type=str, help="Input file with addresses (one per line)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for results (JSON lines format)"
    )

    # Processing configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--no-multiprocessing", action="store_true", help="Disable multiprocessing"
    )

    # Threshold configuration
    parser.add_argument(
        "--pattern-threshold-high",
        type=float,
        default=0.8,
        help="High confidence threshold for pattern matching (default: 0.8)",
    )
    parser.add_argument(
        "--pattern-threshold-medium",
        type=float,
        default=0.6,
        help="Medium confidence threshold for pattern matching (default: 0.6)",
    )
    parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for ML model (default: 0.7)",
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/turkish_address_ner_improved",
        help="Path to ML model (default: models/turkish_address_ner_improved)",
    )

    # Validation configuration
    parser.add_argument(
        "--no-validation", action="store_true", help="Disable geographic validation"
    )
    parser.add_argument(
        "--geo-data-dir", type=str, help="Directory with geographic data files"
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show processing statistics"
    )

    # Determinism
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic processing (default: 42)",
    )

    return parser.parse_args()


def process_single_address(
    pipeline: AddressNormalizationPipeline, address: str, args
) -> None:
    """Process a single address and output result."""

    result = pipeline.process_single(address)

    if result.success:
        if args.output:
            # Write to file
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.address_out.to_json() + "\n")

            if not args.quiet:
                print(f"Result written to {args.output}")

                if args.stats:
                    print(f"Method: {result.processing_method}")
                    print(f"Confidence: {result.confidence:.3f}")
                    print(f"Processing time: {result.processing_time_ms:.1f}ms")
        else:
            # Print to stdout
            if args.pretty:
                addr_dict = result.address_out.to_dict()
                print(json.dumps(addr_dict, ensure_ascii=False, indent=2))
            else:
                print(result.address_out.to_json())

            if args.stats and not args.quiet:
                print(
                    f"\nStats: {result.processing_method}, "
                    f"confidence={result.confidence:.3f}, "
                    f"time={result.processing_time_ms:.1f}ms",
                    file=sys.stderr,
                )
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def process_file(
    pipeline: AddressNormalizationPipeline, input_file: str, output_file: str, args
) -> None:
    """Process addresses from file."""

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    if not output_file:
        output_file = Path(input_file).with_suffix(".results.jsonl")
        if not args.quiet:
            print(f"Output file not specified, using: {output_file}")

    stats = pipeline.process_file(input_file, output_file)

    if not args.quiet:
        print("Processing complete!")
        print(f"Results written to: {stats['output_file']}")

        if args.stats:
            print("\nStatistics:")
            print(f"  Total processed: {stats['total_processed']}")
            print(f"  Successful: {stats['successful']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Average confidence: {stats['average_confidence']:.3f}")
            print(f"  Average time: {stats['average_time_ms']:.1f}ms")
            print("  Methods used:")
            for method, count in stats["methods_used"].items():
                print(f"    {method}: {count}")


def main():
    """Main CLI entry point."""

    args = parse_args()

    # Create pipeline configuration
    config = PipelineConfig(
        pattern_threshold_high=args.pattern_threshold_high,
        pattern_threshold_medium=args.pattern_threshold_medium,
        ml_confidence_threshold=args.ml_threshold,
        ml_model_path=args.model_path,
        enable_multiprocessing=not args.no_multiprocessing,
        max_workers=args.workers,
        batch_size=args.batch_size,
        enable_validation=not args.no_validation,
        geo_data_dir=args.geo_data_dir,
        random_seed=args.seed,
        log_level="DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO",
    )

    # Create pipeline
    try:
        if not args.quiet:
            print("Initializing pipeline...")

        pipeline = AddressNormalizationPipeline(config)

        if not args.quiet:
            print("Pipeline ready.")

    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    # Process input
    try:
        if args.address:
            # Single address
            process_single_address(pipeline, args.address, args)
        else:
            # File processing
            process_file(pipeline, args.input, args.output, args)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
