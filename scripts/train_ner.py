#!/usr/bin/env python3
"""
Train NER model for Turkish address parsing.

This script trains a spaCy NER model using the converted training data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from spacy.training.example import Example
from spacy.util import compounding, minibatch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from addrnorm.ml.ner_baseline import TurkishAddressNER


def load_training_data(jsonl_path: str) -> List[Tuple[str, Dict]]:
    """Load training data from JSONL file."""
    data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                text = record["text"]
                entities = record["entities"]
                annotations = {"entities": entities}
                data.append((text, annotations))

    print(f"Loaded {len(data)} training samples")
    return data


def create_training_examples(
    nlp, training_data: List[Tuple[str, Dict]]
) -> List[Example]:
    """Create spaCy training examples."""
    examples = []

    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    return examples


def train_model(train_data_path: str, model_output_path: str, n_iter: int = 30):
    """Train the NER model."""
    print("🚀 Starting NER model training...")
    print(f"Training data: {train_data_path}")
    print(f"Output model: {model_output_path}")
    print(f"Iterations: {n_iter}")
    print("-" * 60)

    # Load training data
    training_data = load_training_data(train_data_path)

    if not training_data:
        print("❌ No training data found!")
        return

    # Initialize Turkish NER model
    ner_model = TurkishAddressNER()
    nlp = ner_model.create_blank_model()

    if nlp is None:
        print("❌ Failed to create spaCy model!")
        return

    # Add the NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels
    labels = set()
    for _, annotations in training_data:
        for start, end, label in annotations["entities"]:
            labels.add(label)

    for label in labels:
        ner.add_label(label)

    print(f"📝 Entity labels: {sorted(labels)}")

    # Create training examples
    examples = create_training_examples(nlp, training_data)

    # Training configuration
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # Start training
    print(f"\n🎯 Training with {len(examples)} examples...")

    with nlp.disable_pipes(*other_pipes):
        nlp.begin_training()

        for iteration in range(n_iter):
            print(f"Iteration {iteration + 1}/{n_iter}")

            # Shuffle training data
            import random

            random.shuffle(examples)

            losses = {}

            # Create mini-batches
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                nlp.update(batch, drop=0.2, losses=losses)

            print(f"  Loss: {losses.get('ner', 0):.4f}")

    # Save the model
    model_path = Path(model_output_path)
    model_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_path)

    print(f"\n✅ Model saved to: {model_path}")

    # Test the trained model
    print("\n🧪 Testing trained model:")
    test_texts = [
        "atatürk mahallesi cumhuriyet caddesi no 15 kat 3 daire 5 ankara",
        "konak mahallesi alsancak bulvarı numara 23 izmir",
        "bağdat caddesi no 125 kadıköy istanbul",
    ]

    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: {text}")
        print("Entities:")
        for ent in doc.ents:
            print(f"  {ent.text} -> {ent.label_}")


def evaluate_model(model_path: str, test_data_path: str):
    """Evaluate the trained model."""
    print("\n📊 Evaluating model on test data...")

    # Load model
    nlp = spacy.load(model_path)

    # Load test data
    test_data = load_training_data(test_data_path)

    if not test_data:
        print("❌ No test data found!")
        return

    # Evaluate
    examples = create_training_examples(nlp, test_data)
    scores = nlp.evaluate(examples)

    print("✅ Evaluation Results:")
    print(f"  Token accuracy: {scores['token_acc']:.4f}")
    print(f"  Entity F-score: {scores['ents_f']:.4f}")
    print(f"  Entity precision: {scores['ents_p']:.4f}")
    print(f"  Entity recall: {scores['ents_r']:.4f}")

    # Show per-label scores
    if "ents_per_type" in scores:
        print("\n📋 Per-entity scores:")
        for label, metrics in scores["ents_per_type"].items():
            print(f"  {label}:")
            print(f"    F-score: {metrics['f']:.4f}")
            print(f"    Precision: {metrics['p']:.4f}")
            print(f"    Recall: {metrics['r']:.4f}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Turkish Address NER model")
    parser.add_argument(
        "--train-data",
        "-t",
        default="data/train/train_ner.jsonl",
        help="Training data JSONL file",
    )
    parser.add_argument(
        "--test-data", default="data/train/test_ner.jsonl", help="Test data JSONL file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/turkish_address_ner",
        help="Output model directory",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=30, help="Number of training iterations"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on test data after training"
    )

    args = parser.parse_args()

    # Train model
    train_model(args.train_data, args.output, args.iterations)

    # Evaluate if requested
    if args.evaluate and Path(args.test_data).exists():
        evaluate_model(args.output, args.test_data)
    else:
        print("\n💡 To evaluate: python scripts/train_ner.py --evaluate")


if __name__ == "__main__":
    main()
