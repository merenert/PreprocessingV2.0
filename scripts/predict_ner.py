#!/usr/bin/env python3
"""
Predict using trained NER model for Turkish address parsing.

This script provides inference capabilities for the trained NER model.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import spacy

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from addrnorm.preprocess import preprocess


class TurkishAddressPredictor:
    """Predict entities in Turkish addresses using trained NER model."""

    def __init__(self, model_path: str):
        """Initialize predictor with trained model."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from: {model_path}")
        self.nlp = spacy.load(model_path)
        print("✅ Model loaded successfully")

    def predict_single(self, address: str, preprocess_text: bool = True) -> Dict:
        """
        Predict entities in a single address.

        Args:
            address: Raw address text
            preprocess_text: Whether to apply preprocessing

        Returns:
            Dictionary with original text, processed text, and entities
        """
        # Preprocess if requested
        if preprocess_text:
            processed_result = preprocess(address)
            text = processed_result["text"]
        else:
            text = address

        # Predict entities
        doc = self.nlp(text)

        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(
                        ent, "score", 1.0
                    ),  # spaCy doesn't provide scores by default
                }
            )

        return {
            "original_text": address,
            "processed_text": text,
            "entities": entities,
            "entity_count": len(entities),
        }

    def predict_batch(
        self, addresses: List[str], preprocess_text: bool = True
    ) -> List[Dict]:
        """
        Predict entities for multiple addresses.

        Args:
            addresses: List of address texts
            preprocess_text: Whether to apply preprocessing

        Returns:
            List of prediction results
        """
        results = []

        for i, address in enumerate(addresses):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(addresses)} addresses...")

            try:
                result = self.predict_single(address, preprocess_text)
                results.append(result)
            except Exception as e:
                print(f"Error processing address {i}: {e}")
                results.append(
                    {
                        "original_text": address,
                        "processed_text": "",
                        "entities": [],
                        "entity_count": 0,
                        "error": str(e),
                    }
                )

        return results

    def predict_from_file(
        self, input_file: str, output_file: str = None, preprocess_text: bool = True
    ) -> List[Dict]:
        """
        Predict entities for addresses in a file.

        Args:
            input_file: Input file path (one address per line)
            output_file: Output JSON file path (optional)
            preprocess_text: Whether to apply preprocessing

        Returns:
            List of prediction results
        """
        # Read addresses
        addresses = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    addresses.append(line)

        print(f"📖 Read {len(addresses)} addresses from {input_file}")

        # Predict
        results = self.predict_batch(addresses, preprocess_text)

        # Save results if output file specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Results saved to: {output_file}")

        return results

    def interactive_mode(self):
        """Interactive prediction mode."""
        print("\n🤖 Turkish Address NER - Interactive Mode")
        print("Enter addresses to analyze (type 'quit' to exit):")
        print("-" * 50)

        while True:
            try:
                address = input("\nAddress: ").strip()

                if address.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not address:
                    continue

                # Predict
                result = self.predict_single(address)

                # Display results
                print(f"\n📍 Original: {result['original_text']}")
                print(f"🔄 Processed: {result['processed_text']}")
                print(f"🏷️  Entities ({result['entity_count']}):")

                if result["entities"]:
                    for ent in result["entities"]:
                        print(f"   {ent['text']} -> {ent['label']}")
                else:
                    print("   No entities found")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def analyze_performance(self, results: List[Dict]) -> Dict:
        """Analyze prediction performance statistics."""
        total = len(results)

        if total == 0:
            return {}

        # Count entities by type
        entity_counts = {}
        total_entities = 0

        for result in results:
            for entity in result.get("entities", []):
                label = entity["label"]
                entity_counts[label] = entity_counts.get(label, 0) + 1
                total_entities += 1

        # Calculate statistics
        results_with_entities = sum(1 for r in results if r.get("entity_count", 0) > 0)
        avg_entities_per_address = total_entities / total if total > 0 else 0

        return {
            "total_addresses": total,
            "addresses_with_entities": results_with_entities,
            "coverage_percentage": (results_with_entities / total) * 100,
            "total_entities": total_entities,
            "avg_entities_per_address": avg_entities_per_address,
            "entity_type_counts": entity_counts,
        }


def main():
    """Main prediction function."""
    import argparse

    parser = argparse.ArgumentParser(description="Turkish Address NER Prediction")
    parser.add_argument(
        "--model",
        "-m",
        default="models/turkish_address_ner",
        help="Path to trained model",
    )
    parser.add_argument("--input", "-i", help="Input file with addresses")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--text", "-t", help="Single address text to predict")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Skip preprocessing"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show performance statistics"
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("   Train a model first using: python scripts/train_ner.py")
        return

    try:
        # Initialize predictor
        predictor = TurkishAddressPredictor(args.model)

        # Handle different modes
        if args.interactive:
            predictor.interactive_mode()

        elif args.text:
            # Single text prediction
            result = predictor.predict_single(args.text, not args.no_preprocess)

            print("\n📋 Prediction Result:")
            print(f"Original: {result['original_text']}")
            print(f"Processed: {result['processed_text']}")
            print(f"Entities ({result['entity_count']}):")

            for ent in result["entities"]:
                print(f"  {ent['text']} -> {ent['label']}")

        elif args.input:
            # File prediction
            results = predictor.predict_from_file(
                args.input, args.output, not args.no_preprocess
            )

            if args.stats:
                stats = predictor.analyze_performance(results)
                print("\n📊 Performance Statistics:")
                print(f"  Total addresses: {stats['total_addresses']}")
                print(f"  Addresses with entities: {stats['addresses_with_entities']}")
                print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
                print(f"  Total entities: {stats['total_entities']}")
                print(
                    f"  Avg entities/address: {stats['avg_entities_per_address']:.2f}"
                )
                print("  Entity types:")
                for label, count in sorted(stats["entity_type_counts"].items()):
                    print(f"    {label}: {count}")

        else:
            print("❌ Please specify --text, --input, or --interactive mode")
            parser.print_help()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
