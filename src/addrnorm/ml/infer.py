"""
Integration module for ML-based address normalization.

This module provides the interface between the pattern-based normalization
and the ML fallback system.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.append(str(src_dir))

try:
    import spacy

    from addrnorm.preprocess import preprocess

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class MLAddressNormalizer:
    """ML-based address normalization fallback system."""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize ML normalizer.

        Args:
            model_path: Path to trained NER model
            confidence_threshold: Minimum confidence for pattern matching
        """
        self.model_path = model_path or "models/turkish_address_ner"
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self._model_loaded = False

        # Try to load model on initialization
        self._load_model()

    def _load_model(self) -> bool:
        """Load the trained NER model."""
        if not SPACY_AVAILABLE:
            print("⚠️  spaCy not available. ML fallback disabled.")
            return False

        if self._model_loaded:
            return True

        model_path = Path(self.model_path)
        if not model_path.exists():
            print(f"⚠️  ML model not found: {model_path}")
            print("   Train a model using: python scripts/train_ner.py")
            return False

        try:
            self.nlp = spacy.load(self.model_path)
            self._model_loaded = True
            print(f"✅ ML model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if ML fallback is available."""
        return SPACY_AVAILABLE and self._model_loaded

    def extract_entities_ml(self, address: str) -> Dict:
        """
        Extract entities using ML model.

        Args:
            address: Address text to process

        Returns:
            Dictionary with extracted entities and metadata
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "ML model not available",
                "entities": [],
                "method": "ml_fallback",
            }

        try:
            # Preprocess address
            processed_result = preprocess(address)
            text = processed_result["text"]

            # Run NER prediction
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
                        "confidence": getattr(ent, "score", 0.8),  # Default confidence
                    }
                )

            return {
                "success": True,
                "original_text": address,
                "processed_text": text,
                "entities": entities,
                "entity_count": len(entities),
                "method": "ml_fallback",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "method": "ml_fallback",
            }

    def normalize_with_ml_fallback(
        self,
        address: str,
        pattern_result: Dict = None,
        use_ml_if_low_confidence: bool = True,
    ) -> Dict:
        """
        Normalize address with ML fallback.

        Args:
            address: Raw address text
            pattern_result: Result from pattern-based normalization
            use_ml_if_low_confidence: Use ML if pattern confidence is low

        Returns:
            Normalized address with method information
        """
        result = {
            "original_address": address,
            "normalized": {},
            "method": "unknown",
            "confidence": 0.0,
            "fallback_used": False,
        }

        # If we have pattern result, check its confidence
        if pattern_result:
            pattern_confidence = pattern_result.get("confidence", 0.0)

            if pattern_confidence >= self.confidence_threshold:
                # Use pattern result
                result.update(
                    {
                        "normalized": pattern_result.get("normalized", {}),
                        "method": "pattern_matching",
                        "confidence": pattern_confidence,
                        "fallback_used": False,
                    }
                )
                return result

        # Use ML fallback if available and needed
        if use_ml_if_low_confidence and self.is_available():
            ml_result = self.extract_entities_ml(address)

            if ml_result.get("success", False):
                # Convert ML entities to normalized format
                normalized = self._convert_ml_entities_to_normalized(
                    ml_result["entities"]
                )

                result.update(
                    {
                        "normalized": normalized,
                        "method": "ml_fallback",
                        "confidence": self._calculate_ml_confidence(
                            ml_result["entities"]
                        ),
                        "fallback_used": True,
                        "ml_details": ml_result,
                    }
                )
            else:
                # ML failed, use pattern result if available
                if pattern_result:
                    result.update(
                        {
                            "normalized": pattern_result.get("normalized", {}),
                            "method": "pattern_matching_fallback",
                            "confidence": pattern_result.get("confidence", 0.0),
                            "fallback_used": False,
                            "ml_error": ml_result.get("error", "Unknown error"),
                        }
                    )
        else:
            # No ML available, use pattern result
            if pattern_result:
                result.update(
                    {
                        "normalized": pattern_result.get("normalized", {}),
                        "method": "pattern_matching_only",
                        "confidence": pattern_result.get("confidence", 0.0),
                        "fallback_used": False,
                    }
                )

        return result

    def _convert_ml_entities_to_normalized(self, entities: List[Dict]) -> Dict:
        """Convert ML entities to normalized address format."""
        normalized = {}

        for entity in entities:
            label = entity["label"]
            text = entity["text"]

            # Map entity labels to normalized keys
            if label == "IL":
                normalized["il"] = text
            elif label == "ILCE":
                normalized["ilce"] = text
            elif label == "MAH":
                normalized["mahalle"] = text
            elif label == "SOKAK":
                normalized["sokak"] = text
            elif label == "CADDE":
                normalized["cadde"] = text
            elif label == "BULVAR":
                normalized["bulvar"] = text
            elif label == "NO":
                normalized["numara"] = text
            elif label == "DAIRE":
                normalized["daire"] = text
            elif label == "KAT":
                normalized["kat"] = text

        return normalized

    def _calculate_ml_confidence(self, entities: List[Dict]) -> float:
        """Calculate overall confidence from ML entities."""
        if not entities:
            return 0.0

        # Average confidence of all entities
        confidences = [ent.get("confidence", 0.5) for ent in entities]
        return sum(confidences) / len(confidences)

    def get_stats(self) -> Dict:
        """Get ML system statistics."""
        return {
            "spacy_available": SPACY_AVAILABLE,
            "model_loaded": self._model_loaded,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "is_available": self.is_available(),
        }


# Global ML normalizer instance
_ml_normalizer = None


def get_ml_normalizer(
    model_path: str = None, confidence_threshold: float = 0.5
) -> MLAddressNormalizer:
    """Get or create ML normalizer instance."""
    global _ml_normalizer

    if _ml_normalizer is None:
        _ml_normalizer = MLAddressNormalizer(model_path, confidence_threshold)

    return _ml_normalizer


def normalize_with_ml_fallback(
    address: str,
    pattern_result: Dict = None,
    model_path: str = None,
    confidence_threshold: float = 0.5,
) -> Dict:
    """
    Convenience function for ML-backed address normalization.

    Args:
        address: Address to normalize
        pattern_result: Optional pattern matching result
        model_path: Path to ML model
        confidence_threshold: Minimum confidence for patterns

    Returns:
        Normalization result with method information
    """
    normalizer = get_ml_normalizer(model_path, confidence_threshold)
    return normalizer.normalize_with_ml_fallback(address, pattern_result)


def is_ml_available() -> bool:
    """Check if ML fallback is available."""
    normalizer = get_ml_normalizer()
    return normalizer.is_available()
