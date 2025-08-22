"""
NER (Named Entity Recognition) baseline for Turkish address parsing.

This module provides a spaCy-based NER system to extract address components
when pattern matching fails or has low confidence.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.training.example import Example
from spacy.util import compounding, minibatch


class TurkishAddressNER:
    """spaCy-based NER model for Turkish addresses."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize NER model.

        Args:
            model_path: Path to saved model. If None, creates blank model.
        """
        self.model_path = model_path
        self.nlp = None
        self.entity_labels = [
            "IL",  # İl/Province
            "ILCE",  # İlçe/District
            "MAH",  # Mahalle/Neighborhood
            "SOKAK",  # Sokak/Street
            "CADDE",  # Cadde/Avenue
            "BULVAR",  # Bulvar/Boulevard
            "NO",  # Numara/Number
            "DAIRE",  # Daire/Apartment
            "KAT",  # Kat/Floor
            "BLOK",  # Blok/Block
            "SITE",  # Site/Complex
            "POSTA",  # Posta Kodu/Postal Code
        ]

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.create_blank_model()

    def create_blank_model(self):
        """Create a blank spaCy model with NER component."""
        # Create blank Turkish model (or use 'tr_core_news_sm' if available)
        try:
            self.nlp = spacy.load("tr_core_news_sm")
        except OSError:
            # Fallback to blank model
            self.nlp = spacy.blank("tr")

        # Add NER component if not present
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")

        # Add entity labels
        for label in self.entity_labels:
            ner.add_label(label)

        return self.nlp

    def load_model(self, model_path: str):
        """Load trained model from disk."""
        self.nlp = spacy.load(model_path)
        print(f"Model loaded from: {model_path}")

    def save_model(self, model_path: str):
        """Save trained model to disk."""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(model_path)
        print(f"Model saved to: {model_path}")

    def train(
        self,
        train_data: List[Tuple[str, Dict]],
        n_iter: int = 30,
        dropout: float = 0.2,
        learn_rate: float = 0.001,
    ):
        """
        Train the NER model.

        Args:
            train_data: List of (text, {"entities": [(start, end, label)]})
            n_iter: Number of training iterations
            dropout: Dropout rate for training
            learn_rate: Learning rate
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        # Prepare training examples
        examples = []
        for text, annotations in train_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        # Get NER component
        self.nlp.get_pipe("ner")

        # Train the model
        optimizer = self.nlp.create_optimizer()

        print(f"Training NER model with {len(examples)} examples...")

        for i in range(n_iter):
            losses = {}

            # Create minibatches with increasing size
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                self.nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)

            if i % 5 == 0:
                print(f"Iteration {i}: Loss = {losses.get('ner', 0):.4f}")

        print("Training completed!")

    def predict(self, text: str) -> Dict[str, Union[str, List[Dict]]]:
        """
        Extract entities from text.

        Args:
            text: Input address text

        Returns:
            Dictionary with original text and extracted entities
        """
        if not self.nlp:
            raise ValueError("Model not initialized")

        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": (
                        float(ent._.get("confidence", 0.0))
                        if hasattr(ent._, "confidence")
                        else 0.0
                    ),
                }
            )

        return {
            "text": text,
            "entities": entities,
            "tokens": [token.text for token in doc],
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict entities for multiple texts."""
        return [self.predict(text) for text in texts]

    def evaluate(self, test_data: List[Tuple[str, Dict]]) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            test_data: Test data in same format as training data

        Returns:
            Dictionary with precision, recall, f1 scores
        """
        if not test_data:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        total_predicted = 0
        total_correct = 0
        total_gold = 0

        for text, gold_annotations in test_data:
            # Get predictions
            prediction = self.predict(text)
            predicted_entities = prediction["entities"]

            # Get gold entities
            gold_entities = gold_annotations.get("entities", [])

            # Convert to sets for comparison
            predicted_spans = {
                (ent["start"], ent["end"], ent["label"]) for ent in predicted_entities
            }
            gold_spans = {(start, end, label) for start, end, label in gold_entities}

            # Calculate metrics
            total_predicted += len(predicted_spans)
            total_gold += len(gold_spans)
            total_correct += len(predicted_spans & gold_spans)

        # Calculate precision, recall, F1
        precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        recall = total_correct / total_gold if total_gold > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_predicted": total_predicted,
            "total_correct": total_correct,
            "total_gold": total_gold,
        }

    def format_bio_output(self, text: str) -> List[Tuple[str, str]]:
        """
        Format prediction output in BIO tagging scheme.

        Args:
            text: Input text

        Returns:
            List of (token, bio_tag) tuples
        """
        doc = self.nlp(text)

        # Initialize all tokens as 'O' (Outside)
        bio_tags = ["O"] * len(doc)

        # Process entities
        for ent in doc.ents:
            start_token = None
            end_token = None

            # Find token indices for entity span
            for i, token in enumerate(doc):
                if token.idx >= ent.start_char and start_token is None:
                    start_token = i
                if token.idx + len(token.text) <= ent.end_char:
                    end_token = i

            # Apply BIO tagging
            if start_token is not None:
                if end_token is None:
                    end_token = start_token

                # First token gets B- prefix
                bio_tags[start_token] = f"B-{ent.label_}"

                # Subsequent tokens get I- prefix
                for i in range(start_token + 1, end_token + 1):
                    if i < len(bio_tags):
                        bio_tags[i] = f"I-{ent.label_}"

        return list(zip([token.text for token in doc], bio_tags))


def load_training_data(file_path: str) -> List[Tuple[str, Dict]]:
    """
    Load training data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of (text, annotations) tuples
    """
    training_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data["text"]
                entities = data["entities"]
                annotations = {"entities": entities}
                training_data.append((text, annotations))

    return training_data


def save_training_data(data: List[Tuple[str, Dict]], file_path: str):
    """Save training data to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for text, annotations in data:
            record = {"text": text, "entities": annotations["entities"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """Demo function for NER baseline."""
    # Create sample training data
    sample_data = [
        (
            "ankara çankaya atatürk mahallesi kızılırmak sokak no 15",
            {
                "entities": [
                    (0, 6, "IL"),
                    (7, 14, "ILCE"),
                    (15, 32, "MAH"),
                    (33, 47, "SOKAK"),
                    (51, 53, "NO"),
                ]
            },
        ),
        (
            "istanbul beşiktaş etiler mahallesi nispetiye caddesi 142",
            {
                "entities": [
                    (0, 8, "IL"),
                    (9, 17, "ILCE"),
                    (18, 33, "MAH"),
                    (34, 52, "CADDE"),
                    (53, 56, "NO"),
                ]
            },
        ),
    ]

    # Initialize and train model
    ner_model = TurkishAddressNER()

    print("Training NER model...")
    ner_model.train(sample_data, n_iter=10)

    # Test prediction
    test_text = "bursa nilüfer görükle mahallesi atatürk caddesi 45"
    result = ner_model.predict(test_text)

    print(f"\nInput: {test_text}")
    print("Extracted entities:")
    for entity in result["entities"]:
        print(f"  {entity['text']} -> {entity['label']}")

    # Test BIO format
    bio_output = ner_model.format_bio_output(test_text)
    print("\nBIO Format:")
    for token, tag in bio_output:
        print(f"  {token}\t{tag}")


if __name__ == "__main__":
    main()
