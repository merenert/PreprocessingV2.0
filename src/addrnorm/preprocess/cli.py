#!/usr/bin/env python3
"""
CLI interface for the address preprocessing module.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    # Try relative import first (when run as module)
    from ..preprocess import preprocess
    from ..utils.contracts import AddressOut, ExplanationParsed, MethodEnum
except ImportError:
    # Fall back to absolute import (when run directly)
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from addrnorm.preprocess import preprocess
    from addrnorm.utils.contracts import AddressOut, ExplanationParsed, MethodEnum


def process_file(input_file: str, output_file: str = None):
    """Process addresses from input file and save to output file."""
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            addresses = [line.strip() for line in f if line.strip()]

        results = []

        for raw_address in addresses:
            # Preprocess the address
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

        return True

    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Turkish Address Preprocessing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/examples/raw_addresses_tr.txt --output processed.json
  %(prog)s --input addresses.txt
  echo "İstanbul Kadıköy Moda Mah." | %(prog)s --stdin
        """,
    )

    parser.add_argument(
        "--input", "-i", help="Input file containing raw addresses (one per line)"
    )

    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")

    parser.add_argument(
        "--stdin", action="store_true", help="Read from stdin instead of file"
    )

    parser.add_argument("--text", "-t", help="Process a single text string")

    args = parser.parse_args()

    if args.text:
        # Process single text
        processed = preprocess(args.text)
        explanation = ExplanationParsed(
            confidence=0.85, method=MethodEnum.PATTERN, warnings=[]
        )

        result = AddressOut(
            explanation_raw=args.text,
            explanation_parsed=explanation,
            normalized_address=processed["text"],
        )

        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    elif args.stdin:
        # Read from stdin
        addresses = [line.strip() for line in sys.stdin if line.strip()]
        results = []

        for raw_address in addresses:
            processed = preprocess(raw_address)
            explanation = ExplanationParsed(
                confidence=0.85, method=MethodEnum.PATTERN, warnings=[]
            )

            result = AddressOut(
                explanation_raw=raw_address,
                explanation_parsed=explanation,
                normalized_address=processed["text"],
            )

            results.append(result.model_dump())

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))

    elif args.input:
        # Process input file
        success = process_file(args.input, args.output)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
