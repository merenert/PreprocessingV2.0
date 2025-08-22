#!/usr/bin/env python3
"""
Geliştirilmiş veri dönüştürme script'i - Daha akıllı entity etiketleme.

Bu script mahalle ve caddeleri ayrı ayrı etiketler.
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


class ImprovedDataConverter:
    """Geliştirilmiş veri dönüştürücü - daha akıllı entity recognition."""

    def __init__(self):
        """Initialize with improved Turkish address patterns."""
        # Mahalle pattern'leri
        self.mahalle_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+mahallesi\b",
            r"\b(\w+(?:\s+\w+)*)\s+mah\.?\b",
        ]

        # Sokak pattern'leri
        self.sokak_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+sokağı?\b",
            r"\b(\w+(?:\s+\w+)*)\s+sok\.?\b",
            r"\b(\d+)\.?\s*sok(?:ak)?\.?\b",
        ]

        # Cadde pattern'leri
        self.cadde_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+caddesi\b",
            r"\b(\w+(?:\s+\w+)*)\s+cad\.?\b",
        ]

        # Bulvar pattern'leri
        self.bulvar_patterns = [
            r"\b(\w+(?:\s+\w+)*)\s+bulvarı?\b",
            r"\b(\w+(?:\s+\w+)*)\s+bul\.?\b",
        ]

        # Numara pattern'leri
        self.numara_patterns = [
            r"\bno:?\s*(\d+[a-zA-Z]?)\b",
            r"\bnumara:?\s*(\d+[a-zA-Z]?)\b",
            r"\b(\d+)\s*(?=\s|$|[.,])",  # Tek başına sayılar
        ]

        # Daire pattern'leri
        self.daire_patterns = [r"\bd\.?\s*(\d+)\b", r"\bdaire:?\s*(\d+)\b"]

        # Kat pattern'leri
        self.kat_patterns = [r"\bk\.?\s*(\d+)\b", r"\bkat:?\s*(\d+)\b"]

        # Turkish cities (expanded)
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
            "denizli",
            "aydın",
            "manisa",
            "balıkesir",
            "çanakkale",
            "eskişehir",
            "kütahya",
        }

        # Turkish districts (expanded)
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
            "gaziemir",
            "karabağlar",
            "bayraklı",
            "efeler",
            "menteşe",
        }

    def find_entities_improved(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Geliştirilmiş entity bulma - mahalle ve caddeleri ayrı tutar.

        Args:
            text: Preprocessed address text

        Returns:
            List of (start, end, label) tuples
        """
        entities = []
        text_lower = text.lower()

        # 1. ÖNCE MAHALLE BUL
        for pattern in self.mahalle_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                start = match.start()
                end = match.end()
                entities.append((start, end, "MAH"))

        # 2. SONRA CADDE BUL (mahalle alanlarını hariç tut)
        for pattern in self.cadde_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                start = match.start()
                end = match.end()

                # Mahalle ile overlap etmediğini kontrol et
                overlap = False
                for e_start, e_end, e_label in entities:
                    if e_label == "MAH" and not (end <= e_start or start >= e_end):
                        overlap = True
                        break

                if not overlap:
                    entities.append((start, end, "CADDE"))

        # 3. SOKAK BUL
        for pattern in self.sokak_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    start = match.start()
                    end = match.end()

                    # Diğer entities ile overlap kontrolü
                    overlap = False
                    for e_start, e_end, e_label in entities:
                        if not (end <= e_start or start >= e_end):
                            overlap = True
                            break

                    if not overlap:
                        entities.append((start, end, "SOKAK"))

        # 4. BULVAR BUL
        for pattern in self.bulvar_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                start = match.start()
                end = match.end()

                # Overlap kontrolü
                overlap = False
                for e_start, e_end, e_label in entities:
                    if not (end <= e_start or start >= e_end):
                        overlap = True
                        break

                if not overlap:
                    entities.append((start, end, "BULVAR"))

        # 5. NUMARA BUL
        for pattern in self.numara_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    start = match.start(1)
                    end = match.end(1)

                    # Sadece rakam içeren matches
                    if re.search(r"\d", text[start:end]):
                        entities.append((start, end, "NO"))

        # 6. DAIRE BUL
        for pattern in self.daire_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "DAIRE"))

        # 7. KAT BUL
        for pattern in self.kat_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                start = match.start(1)
                end = match.end(1)
                entities.append((start, end, "KAT"))

        # 8. İL BUL
        words = text_lower.split()
        pos = 0
        for word in words:
            if word in self.iller:
                start = text_lower.find(word, pos)
                end = start + len(word)
                entities.append((start, end, "IL"))
                pos = end
            else:
                pos += len(word) + 1

        # 9. İLÇE BUL
        pos = 0
        for word in words:
            if word in self.ilceler:
                start = text_lower.find(word, pos)
                end = start + len(word)
                entities.append((start, end, "ILCE"))
                pos = end
            else:
                pos += len(word) + 1

        # Overlapping entities'i temizle
        entities = self.remove_overlapping_entities(entities)

        return entities

    def remove_overlapping_entities(
        self, entities: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """Remove overlapping entities, keep higher priority ones."""
        if not entities:
            return entities

        # Priority order: MAH > CADDE > SOKAK > BULVAR > NO > DAIRE > KAT > IL > ILCE
        priority = {
            "MAH": 1,
            "CADDE": 2,
            "SOKAK": 3,
            "BULVAR": 4,
            "NO": 5,
            "DAIRE": 6,
            "KAT": 7,
            "IL": 8,
            "ILCE": 9,
        }

        # Sort by position first, then by priority
        entities.sort(key=lambda x: (x[0], priority.get(x[2], 10)))

        filtered = []
        for entity in entities:
            start, end, label = entity

            # Check for overlaps with existing entities
            overlaps = False
            for f_start, f_end, f_label in filtered:
                if not (end <= f_start or start >= f_end):  # Overlap detected
                    # Keep higher priority entity
                    if priority.get(label, 10) < priority.get(f_label, 10):
                        # Remove lower priority entity
                        filtered = [
                            e
                            for e in filtered
                            if not (e[0] == f_start and e[1] == f_end)
                        ]
                    else:
                        overlaps = True
                        break

            if not overlaps:
                filtered.append(entity)

        return sorted(filtered, key=lambda x: x[0])

    def convert_csv_to_ner_improved(
        self, csv_path: str, max_samples: int = None
    ) -> List[Tuple[str, Dict]]:
        """
        Geliştirilmiş CSV to NER dönüştürme.

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

                # Find entities with improved algorithm
                entities = self.find_entities_improved(processed_text)

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
        print(f"\n📋 Sample improved data (first {n_samples}):")
        print("=" * 80)

        for i, (text, annotations) in enumerate(data[:n_samples]):
            print(f"{i+1}. Text: {text}")
            print("   Entities:")
            for start, end, label in annotations["entities"]:
                entity_text = text[start:end]
                print(f"     {entity_text} -> {label} ({start}, {end})")
            print()


def main():
    """Convert train_sample.csv with improved algorithm."""
    import argparse

    parser = argparse.ArgumentParser(description="Improved CSV to NER training data")
    parser.add_argument(
        "--input", "-i", default="train_sample.csv", help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/train/train_ner_improved.jsonl",
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

    # Convert data with improved algorithm
    converter = ImprovedDataConverter()
    print(f"Converting {args.input} with IMPROVED algorithm...")

    data = converter.convert_csv_to_ner_improved(args.input, args.max_samples)

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

    test_path = Path(args.output).parent / "test_ner_improved.jsonl"
    converter.save_to_jsonl(test_data, str(test_path))

    print(f"✅ Improved training data: {len(train_data)} samples")
    print(f"✅ Improved test data: {len(test_data)} samples")

    # Show samples
    converter.show_sample_data(train_data)


if __name__ == "__main__":
    main()
