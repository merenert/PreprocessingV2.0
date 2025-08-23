"""
Spatial relations and directional keywords for Turkish address explanations.
Handles spatial relationship extraction between landmarks and addresses.
"""

import re
from typing import Dict, List, Tuple, Optional
from .models import SpatialRelation


class SpatialRelationExtractor:
    """Extracts spatial relationships from Turkish text."""

    def __init__(self):
        """Initialize spatial relation patterns and keywords."""
        self.spatial_keywords = {
            # Primary directional relations
            "karşısı": ["karşısı", "karşısında", "karşı"],
            "yanı": ["yanı", "yanında", "yan", "kenarı", "kenarında"],
            "arkası": ["arkası", "arkasında", "arka", "gerisinde", "geri"],
            "önü": ["önü", "önünde", "ön", "önde"],
            "üstü": ["üstü", "üstünde", "üst", "üzerinde", "yukarısında"],
            "altı": ["altı", "altında", "alt", "aşağısında"],
            "bitişiği": ["bitişiği", "bitişik", "bitişiğinde", "yanında"],
            # Secondary proximity relations
            "yakını": ["yakını", "yakınında", "civarda", "civarında", "etrafında"],
            "çevresinde": ["çevresinde", "çevresi", "etrafında", "etrafı"],
            "içinde": ["içinde", "içi", "iç", "dahilinde"],
            "dışında": ["dışında", "dışı", "haricinde", "dış"],
            "arasında": ["arasında", "arası", "ortasında", "orta"],
        }

        # Compile regex patterns for each relation
        self.relation_patterns = self._compile_patterns()

        # Confidence weights for different relation types
        self.confidence_weights = {
            "karşısı": 0.95,
            "yanı": 0.90,
            "önü": 0.90,
            "arkası": 0.85,
            "bitişiği": 0.85,
            "üstü": 0.80,
            "altı": 0.80,
            "yakını": 0.75,
            "çevresinde": 0.70,
            "içinde": 0.65,
            "dışında": 0.60,
            "arasında": 0.55,
        }

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for spatial relations."""
        patterns = {}

        for relation, keywords in self.spatial_keywords.items():
            # Create pattern that matches any of the keywords
            keyword_pattern = "|".join(re.escape(kw) for kw in keywords)
            # Add word boundaries to avoid partial matches
            pattern = rf"\b(?:{keyword_pattern})\b"
            patterns[relation] = re.compile(pattern, re.IGNORECASE | re.UNICODE)

        return patterns

    def extract_relations(self, text: str) -> List[SpatialRelation]:
        """
        Extract spatial relations from text.

        Args:
            text: Input text to analyze

        Returns:
            List of detected spatial relations with confidence scores
        """
        relations = []
        text_lower = text.lower().strip()

        # Track matches to avoid duplicates
        matched_positions = set()

        for relation_type, pattern in self.relation_patterns.items():
            matches = list(pattern.finditer(text_lower))

            for match in matches:
                start, end = match.span()

                # Skip if this position was already matched by a higher-confidence relation
                if any(
                    start <= pos < end or pos <= start < pos + 10
                    for pos in matched_positions
                ):
                    continue

                confidence = self.confidence_weights.get(relation_type, 0.5)

                # Adjust confidence based on context
                confidence = self._adjust_confidence(
                    text_lower, match, relation_type, confidence
                )

                if confidence >= 0.3:  # Minimum threshold
                    relations.append(
                        SpatialRelation(relation=relation_type, confidence=confidence)
                    )

                    # Mark this position as used
                    for pos in range(start, end):
                        matched_positions.add(pos)

        # Sort by confidence and return top matches
        relations.sort(key=lambda x: x.confidence, reverse=True)
        return relations[:3]  # Return top 3 matches

    def _adjust_confidence(
        self, text: str, match: re.Match, relation_type: str, base_confidence: float
    ) -> float:
        """
        Adjust confidence based on context and text quality.

        Args:
            text: Full text being analyzed
            match: Regex match object
            relation_type: Type of spatial relation
            base_confidence: Base confidence score

        Returns:
            Adjusted confidence score
        """
        confidence = base_confidence
        matched_text = match.group()

        # Boost confidence for exact keyword matches
        primary_keywords = self.spatial_keywords[relation_type][
            :2
        ]  # First 2 are primary
        if matched_text in primary_keywords:
            confidence += 0.1

        # Context analysis
        start_pos = max(0, match.start() - 10)
        end_pos = min(len(text), match.end() + 10)
        context = text[start_pos:end_pos]

        # Boost if surrounded by landmark indicators
        landmark_indicators = [
            "hotel",
            "market",
            "hastane",
            "okul",
            "cami",
            "park",
            "banka",
        ]
        if any(indicator in context for indicator in landmark_indicators):
            confidence += 0.05

        # Penalize if in a very short text (likely incomplete)
        if len(text.strip()) < 10:
            confidence -= 0.2

        # Penalize if surrounded by numbers (might be address numbers)
        if re.search(r"\d+.*" + re.escape(matched_text) + r".*\d+", context):
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def get_best_relation(self, text: str) -> Optional[SpatialRelation]:
        """
        Get the best spatial relation from text.

        Args:
            text: Input text

        Returns:
            Best spatial relation or None if none found
        """
        relations = self.extract_relations(text)
        return relations[0] if relations else None

    def get_supported_relations(self) -> List[str]:
        """Get list of all supported spatial relations."""
        return list(self.spatial_keywords.keys())

    def get_relation_keywords(self, relation_type: str) -> List[str]:
        """Get keywords for a specific relation type."""
        return self.spatial_keywords.get(relation_type, [])
