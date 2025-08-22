#!/usr/bin/env python3
"""
Convert train_sample.csv to NER training format.

This script converts the existing train_sample.csv data into JSONL format
for NER training with manual entity labeling based on pattern recognition.
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from addrnorm.preprocess import preprocess


class RealDataConverter:
    """Convert real address data to NER training format."""

    def __init__(self):
        """Initialize with Turkish address patterns."""
        # Common patterns for entity recognition
        self.mahalle_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+mah(?:allesi)?\.?\b",
            r"\b(\w+(?:\s+\w+)*)\s+mahallesi\b",
        ]

        self.sokak_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+sok(?:ak)?\.?\b",
            r"\b(\w+(?:\s+\w+)*)\s+sokak\b",
            r"\b(\d+)\.?\s*sok(?:ak)?\.?\b",
        ]

        self.cadde_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+cad(?:desi)?\.?\b",
            r"\b(\w+(?:\s+\w+)*)\s+caddesi\b",
        ]

        self.bulvar_patterns = [r"\b(\w+(?:\s+\w+)*)\s+bulvarı?\b"]

        self.numara_patterns = [
            r"\bno:?\s*(\d+[a-zA-Z]?)\b",
            r"\bnumara:?\s*(\d+[a-zA-Z]?)\b",
        ]

        self.daire_patterns = [r"\bd\.?\s*(\d+)\b", r"\bdaire:?\s*(\d+)\b"]

        self.kat_patterns = [r"\bk\.?\s*(\d+)\b", r"\bkat:?\s*(\d+)\b"]

        # Turkish cities (sample)
        self.iller = {
            "ankara",
            "istanbul",
            "izmir",
            "bursa",
            "antalya",
            "adana",
            "konya",
            "muğla",
            "fethiye",
            "çeşme",
            "dikili",
            "bodrum",
            "marmaris",
        }

        # Turkish districts (sample)
        self.ilceler = {
            "çankaya",
            "beşiktaş",
            "kadıköy",
            "üsküdar",
            "fatih",
            "beyoğlu",
            "karşıyaka",
            "bornova",
            "konak",
            "buca",
            "çiğli",
            "güzelbahçe",
        }

    def find_entities(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find entities in text using pattern matching.

        Args:
            text: Preprocessed address text

        Returns:
            List of (start, end, label) tuples
        """
        entities = []
        text_lower = text.lower()

        # Find mahalle
        for pattern in self.mahalle_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "MAH"))

        # Find sokak
        for pattern in self.sokak_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if match.groups():
                    start = match.start(1)
                    end = match.end(1)
                    entities.append((start, end, "SOKAK"))

        # Find cadde
        for pattern in self.cadde_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "CADDE"))

        # Find bulvar
        for pattern in self.bulvar_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "BULVAR"))

        # Find numbers
        for pattern in self.numara_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "NO"))

        # Find daire
        for pattern in self.daire_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "DAIRE"))

        # Find kat
        for pattern in self.kat_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "KAT"))

        # Find cities/provinces
        words = text_lower.split()
        pos = 0
        for word in words:
            if word in self.iller:
                start = text_lower.find(word, pos)
                end = start + len(word)
                entities.append((start, end, "IL"))
                pos = end
            elif word in self.ilceler:
                start = text_lower.find(word, pos)
                end = start + len(word)
                entities.append((start, end, "ILCE"))
                pos = end
            else:
                pos += len(word) + 1

        # Remove overlapping entities (keep longer ones)
        entities = self.remove_overlapping_entities(entities)

        return entities

    def remove_overlapping_entities(
        self, entities: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """Remove overlapping entities, keeping longer ones."""
        if not entities:
            return entities

        # Sort by start position
        entities.sort(key=lambda x: (x[0], -x[1]))

        filtered = []
        for entity in entities:
            start, end, label = entity

            # Check if this entity overlaps with any in filtered list
            overlaps = False
            for f_start, f_end, f_label in filtered:
                if start < f_end and end > f_start:  # Overlap detected
                    # Keep the longer one
                    if (end - start) <= (f_end - f_start):
                        overlaps = True
                        break
                    else:
                        # Remove the shorter one from filtered
                        filtered = [
                            (s, e, label)
                            for s, e, label in filtered
                            if not (s == f_start and e == f_end)
                        ]

            if not overlaps:
                filtered.append(entity)

        return sorted(filtered, key=lambda x: x[0])

    def convert_csv_to_ner(
        self, csv_path: str, max_samples: int = None
    ) -> List[Tuple[str, Dict]]:
        """
        Convert CSV data to NER format.

        Args:
            csv_path: Path to CSV file
            max_samples: Maximum number of samples to process

        Returns:
            List of (text, annotations) tuples
        """
        training_data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                if i % 500 == 0:
                    print(f"Processed {i} samples...")

                address = row["address"].strip()
                if not address:
                    continue

                # Preprocess the address
                processed = preprocess(address)
                processed_text = processed["text"]

                # Find entities
                entities = self.find_entities(processed_text)

                # Only include samples with entities
                if entities:
                    annotations = {"entities": entities}
                    training_data.append((processed_text, annotations))

        print(f"Converted {len(training_data)} samples with entities")
        return training_data

    def save_to_jsonl(self, data: List[Tuple[str, Dict]], output_path: str):
        """Save data to JSONL format."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for text, annotations in data:
                record = {"text": text, "entities": annotations["entities"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved to: {output_path}")

    def show_sample_data(self, data: List[Tuple[str, Dict]], n_samples: int = 5):
        """Show sample converted data."""
        print(f"\n📋 Sample converted data (first {n_samples}):")
        print("=" * 80)

        for i, (text, annotations) in enumerate(data[:n_samples]):
            print(f"{i+1}. Text: {text}")
            print("   Entities:")
            for start, end, label in annotations["entities"]:
                entity_text = text[start:end]
                print(f"     {entity_text} -> {label} ({start}, {end})")
            print()


def main():
    """Convert train_sample.csv to NER training format."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV to NER training data")
    parser.add_argument(
        "--input", "-i", default="train_sample.csv", help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/train/train_ner.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=5000,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Fraction for test set"
    )

    args = parser.parse_args()

    # Convert data
    converter = RealDataConverter()
    print(f"Converting {args.input} to NER format...")

    data = converter.convert_csv_to_ner(args.input, args.max_samples)

    if not data:
        print("❌ No data converted. Check input file and patterns.")
        return

    # Split into train/test
    import random

    random.shuffle(data)
    split_idx = int(len(data) * (1 - args.test_split))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Save data
    converter.save_to_jsonl(train_data, args.output)

    test_path = Path(args.output).parent / "test_ner.jsonl"
    converter.save_to_jsonl(test_data, str(test_path))

    print(f"✅ Training data: {len(train_data)} samples")
    print(f"✅ Test data: {len(test_data)} samples")

    # Show samples
    converter.show_sample_data(train_data)


if __name__ == "__main__":
    main()
