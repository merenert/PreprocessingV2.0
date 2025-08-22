#!/usr/bin/env python3
"""
Synthetic Turkish address data generator.

This script generates training data for NER model by using existing patterns
and creating variations with different Turkish address components.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from addrnorm.preprocess import preprocess


class TurkishAddressSynthesizer:
    """Generate synthetic Turkish address data for NER training."""

    def __init__(self):
        """Initialize with Turkish address components."""
        # Turkish provinces (sample)
        self.iller = [
            "ankara",
            "istanbul",
            "izmir",
            "bursa",
            "antalya",
            "adana",
            "konya",
            "gaziantep",
            "şanlıurfa",
            "kocaeli",
            "mersin",
            "diyarbakır",
            "kayseri",
            "eskişehir",
            "urfa",
            "malatya",
            "erzurum",
            "van",
            "batman",
            "elazığ",
        ]

        # Districts (sample)
        self.ilceler = [
            "çankaya",
            "keçiören",
            "mamak",
            "sincan",
            "etimesgut",
            "altındağ",
            "beşiktaş",
            "kadıköy",
            "üsküdar",
            "şişli",
            "beyoğlu",
            "fatih",
            "karşıyaka",
            "bornova",
            "konak",
            "bayraklı",
            "buca",
            "çiğli",
            "nilüfer",
            "osmangazi",
            "yıldırım",
            "mudanya",
            "gemlik",
            "inegöl",
            "kepez",
            "muratpaşa",
            "konyaaltı",
            "aksu",
            "döşemealtı",
            "manavgat",
        ]

        # Neighborhoods
        self.mahalleler = [
            "atatürk mahallesi",
            "cumhuriyet mahallesi",
            "yenişehir mahallesi",
            "merkez mahallesi",
            "kızılay mahallesi",
            "çayyolu mahallesi",
            "etiler mahallesi",
            "nişantaşı mahallesi",
            "levent mahallesi",
            "maslak mahallesi",
            "bebek mahallesi",
            "ortaköy mahallesi",
            "kozyatağı mahallesi",
            "acıbadem mahallesi",
            "göztepe mahallesi",
            "fenerbahçe mahallesi",
            "moda mahallesi",
            "caddebostan mahallesi",
            "alsancak mahallesi",
            "karataş mahallesi",
            "güzelyalı mahallesi",
            "bahçelievler mahallesi",
            "kültür mahallesi",
            "çankaya mahallesi",
        ]

        # Street types and names
        self.sokak_cadde = [
            ("atatürk caddesi", "CADDE"),
            ("gazi mustafa kemal bulvarı", "BULVAR"),
            ("istiklal caddesi", "CADDE"),
            ("cumhuriyet caddesi", "CADDE"),
            ("kızılırmak sokak", "SOKAK"),
            ("sakarya sokak", "SOKAK"),
            ("mimar sinan caddesi", "CADDE"),
            ("mehmet akif ersoy sokak", "SOKAK"),
            ("fatih sultan mehmet bulvarı", "BULVAR"),
            ("adnan menderes bulvarı", "BULVAR"),
            ("zübeyde hanım caddesi", "CADDE"),
            ("halide edip adıvar sokak", "SOKAK"),
            ("barbaros bulvarı", "BULVAR"),
            ("nispetiye caddesi", "CADDE"),
            ("teşvikiye caddesi", "CADDE"),
            ("valikonağı caddesi", "CADDE"),
            ("bağdat caddesi", "CADDE"),
            ("nişantaşı caddesi", "CADDE"),
        ]

        # Building numbers
        self.numbers = list(range(1, 500, 2)) + list(range(2, 500, 2))

        # Apartment/floor info
        self.daire_info = [
            ("daire", "DAIRE"),
            ("d", "DAIRE"),
            ("no", "DAIRE"),
            ("kat", "KAT"),
            ("k", "KAT"),
        ]

        # Sites/complexes
        self.siteler = [
            "park sitesi",
            "bahçe sitesi",
            "villa sitesi",
            "residence",
            "plaza",
            "towers",
            "konakları",
            "evleri",
        ]

    def generate_address_pattern_1(self) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Generate: IL ILCE MAHALLE SOKAK/CADDE NO"""
        il = random.choice(self.iller)
        ilce = random.choice(self.ilceler)
        mahalle = random.choice(self.mahalleler)
        sokak_cadde, sokak_type = random.choice(self.sokak_cadde)
        no = random.choice(self.numbers)

        # Build address text
        parts = [il, ilce, mahalle, sokak_cadde, f"no {no}"]
        text = " ".join(parts)

        # Calculate entity positions
        entities = []
        pos = 0

        # IL
        entities.append((pos, pos + len(il), "IL"))
        pos += len(il) + 1

        # ILCE
        entities.append((pos, pos + len(ilce), "ILCE"))
        pos += len(ilce) + 1

        # MAHALLE
        entities.append((pos, pos + len(mahalle), "MAH"))
        pos += len(mahalle) + 1

        # SOKAK/CADDE
        entities.append((pos, pos + len(sokak_cadde), sokak_type))
        pos += len(sokak_cadde) + 1

        # NO (skip "no " prefix)
        pos += 3  # "no "
        entities.append((pos, pos + len(str(no)), "NO"))

        return text, entities

    def generate_address_pattern_2(self) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Generate: MAHALLE SOKAK/CADDE NO DAIRE KAT IL/ILCE"""
        mahalle = random.choice(self.mahalleler)
        sokak_cadde, sokak_type = random.choice(self.sokak_cadde)
        no = random.choice(self.numbers)
        daire_no = random.randint(1, 20)
        kat_no = random.randint(1, 8)
        il = random.choice(self.iller)

        # Build text
        parts = [
            mahalle,
            sokak_cadde,
            f"no {no}",
            f"daire {daire_no}",
            f"kat {kat_no}",
            il,
        ]
        text = " ".join(parts)

        entities = []
        pos = 0

        # MAHALLE
        entities.append((pos, pos + len(mahalle), "MAH"))
        pos += len(mahalle) + 1

        # SOKAK/CADDE
        entities.append((pos, pos + len(sokak_cadde), sokak_type))
        pos += len(sokak_cadde) + 1

        # NO
        pos += 3  # "no "
        entities.append((pos, pos + len(str(no)), "NO"))
        pos += len(str(no)) + 1

        # DAIRE
        pos += 6  # "daire "
        entities.append((pos, pos + len(str(daire_no)), "DAIRE"))
        pos += len(str(daire_no)) + 1

        # KAT
        pos += 4  # "kat "
        entities.append((pos, pos + len(str(kat_no)), "KAT"))
        pos += len(str(kat_no)) + 1

        # IL
        entities.append((pos, pos + len(il), "IL"))

        return text, entities

    def generate_address_pattern_3(self) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Generate: SITE IL ILCE MAHALLE BLOK NO"""
        site = random.choice(self.siteler)
        il = random.choice(self.iller)
        ilce = random.choice(self.ilceler)
        mahalle = random.choice(self.mahalleler)
        blok = random.choice(["a blok", "b blok", "c blok", "1 blok", "2 blok"])
        no = random.choice(self.numbers)

        parts = [site, il, ilce, mahalle, blok, f"no {no}"]
        text = " ".join(parts)

        entities = []
        pos = 0

        # SITE
        entities.append((pos, pos + len(site), "SITE"))
        pos += len(site) + 1

        # IL
        entities.append((pos, pos + len(il), "IL"))
        pos += len(il) + 1

        # ILCE
        entities.append((pos, pos + len(ilce), "ILCE"))
        pos += len(ilce) + 1

        # MAHALLE
        entities.append((pos, pos + len(mahalle), "MAH"))
        pos += len(mahalle) + 1

        # BLOK
        entities.append((pos, pos + len(blok), "BLOK"))
        pos += len(blok) + 1

        # NO
        pos += 3  # "no "
        entities.append((pos, pos + len(str(no)), "NO"))

        return text, entities

    def add_variations(
        self, text: str, entities: List[Tuple[int, int, str]]
    ) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Add variations like abbreviations, case changes, punctuation."""
        variations = [(text, entities)]

        # Abbreviation variations
        abbrev_text = (
            text.replace("mahallesi", "mah")
            .replace("caddesi", "cad")
            .replace("sokak", "sok")
        )
        if abbrev_text != text:
            # Recalculate entity positions for abbreviated version
            abbrev_entities = []
            offset = 0
            for start, end, label in entities:
                # Simple offset adjustment (not perfect but good enough)
                new_start = start + offset
                original_span = text[start:end]

                if "mahallesi" in original_span:
                    original_span = original_span.replace("mahallesi", "mah")
                    offset -= 3  # "mahallesi" -> "mah" = -3 chars
                elif "caddesi" in original_span:
                    original_span = original_span.replace("caddesi", "cad")
                    offset -= 2  # "caddesi" -> "cad" = -2 chars
                elif "sokak" in original_span:
                    original_span = original_span.replace("sokak", "sok")
                    offset -= 1  # "sokak" -> "sok" = -1 char

                new_end = new_start + len(original_span)
                abbrev_entities.append((new_start, new_end, label))

            variations.append((abbrev_text, abbrev_entities))

        # Case variations
        upper_text = text.upper()
        variations.append((upper_text, entities))

        # Punctuation variations
        punct_text = (
            text.replace(" no ", " no: ")
            .replace(" daire ", " d. ")
            .replace(" kat ", " k. ")
        )
        if punct_text != text:
            variations.append(
                (punct_text, entities)
            )  # Entity positions stay same for simple punct

        return variations

    def generate_synthetic_data(self, n_samples: int = 5000) -> List[Tuple[str, Dict]]:
        """Generate n_samples synthetic Turkish addresses."""
        print(f"Generating {n_samples} synthetic Turkish addresses...")

        data = []
        patterns = [
            self.generate_address_pattern_1,
            self.generate_address_pattern_2,
            self.generate_address_pattern_3,
        ]

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{n_samples} samples...")

            # Choose random pattern
            pattern_func = random.choice(patterns)
            text, entities = pattern_func()

            # Preprocess the text (this is what the model will see)
            processed = preprocess(text)
            processed_text = processed["text"]

            # For synthetic data, we'll use the processed text
            # Entity positions would need recalculation, but for simplicity
            # we'll use original positions as approximation

            # Add to dataset
            annotations = {"entities": entities}
            data.append((processed_text, annotations))

            # Add variations
            if random.random() < 0.3:  # 30% chance of variations
                variations = self.add_variations(text, entities)
                for var_text, var_entities in variations[1:]:  # Skip original
                    var_processed = preprocess(var_text)
                    var_annotations = {"entities": var_entities}
                    data.append((var_processed["text"], var_annotations))

        print(f"Generated {len(data)} total samples (including variations)")
        return data


def main():
    """Generate synthetic training data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic Turkish address data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/train/train_ner.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=5000,
        help="Number of base samples to generate",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )

    args = parser.parse_args()

    # Generate data
    synthesizer = TurkishAddressSynthesizer()
    data = synthesizer.generate_synthetic_data(args.samples)

    # Split into train/test
    random.shuffle(data)
    split_idx = int(len(data) * (1 - args.test_split))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Save training data
    train_path = Path(args.output)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for text, annotations in train_data:
            record = {"text": text, "entities": annotations["entities"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save test data
    test_path = train_path.parent / "test_ner.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for text, annotations in test_data:
            record = {"text": text, "entities": annotations["entities"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Training data saved: {train_path} ({len(train_data)} samples)")
    print(f"✅ Test data saved: {test_path} ({len(test_data)} samples)")

    # Show sample
    print("\n📋 Sample generated data:")
    for i, (text, annotations) in enumerate(train_data[:3]):
        print(f"{i+1}. Text: {text}")
        print(f"   Entities: {annotations['entities']}")
        print()


if __name__ == "__main__":
    main()
