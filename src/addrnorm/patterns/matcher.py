"""
Pattern matcher for Turkish addresses.

Matches preprocessed text against compiled patterns and calculates confidence scores.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .compiler import CompiledPattern, PatternCompiler
from .thresholds import ThresholdManager


@dataclass
class MatchResult:
    """Result of pattern matching."""

    pattern_id: str
    pattern_priority: int
    slots: Dict[str, str]  # slot_name -> matched_value
    confidence: float
    raw_text: str
    matched_text: str

    def __str__(self):
        return (
            f"Match(id={self.pattern_id}, conf={self.confidence:.3f}, "
            f"slots={len(self.slots)})"
        )


class PatternMatcher:
    """Matches text against compiled patterns and calculates confidence scores."""

    def __init__(
        self,
        compiler: Optional[PatternCompiler] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        """Initialize matcher with compiled patterns."""
        self.compiler = compiler or PatternCompiler()
        self.patterns = self.compiler.compile_all_patterns()
        self.keywords = self.compiler.get_keywords()
        self.scoring_config = self.compiler.get_scoring_config()

        # Initialize threshold manager
        self.threshold_manager = threshold_manager or ThresholdManager()

        # Cache commonly used config values
        self.slot_weights = self.scoring_config.get("slot_weights", {})
        self.min_confidence = self.scoring_config.get("min_confidence_threshold", 0.3)
        self.good_match_threshold = self.scoring_config.get("good_match_threshold", 0.7)

    def match_text(self, text: str, max_matches: int = 5) -> List[MatchResult]:
        """
        Match text against all patterns and return best matches.

        Args:
            text: Preprocessed text to match
            max_matches: Maximum number of matches to return

        Returns:
            List of match results sorted by confidence (highest first)
        """
        if not text or not text.strip():
            return []

        matches = []

        # Try each pattern in priority order
        for pattern in self.patterns:
            match_result = self._try_pattern(pattern, text)
            if match_result:
                # Get dynamic threshold for this pattern
                threshold = self.threshold_manager.get_threshold(pattern.id)

                # Only include matches that meet the dynamic threshold
                if match_result.confidence >= threshold:
                    matches.append(match_result)

        # Sort by confidence, then by priority
        matches.sort(key=lambda m: (-m.confidence, -m.pattern_priority))

        return matches[:max_matches]

    def get_best_match(self, text: str) -> Optional[MatchResult]:
        """Get the single best match for the given text."""
        matches = self.match_text(text, max_matches=1)
        return matches[0] if matches else None

    def provide_feedback(self, pattern_id: str, was_successful: bool) -> None:
        """
        Provide feedback on pattern matching success.

        Args:
            pattern_id: ID of the pattern that was used
            was_successful: Whether the pattern matching was successful
        """
        self.threshold_manager.update_success(pattern_id, was_successful)

    def get_threshold_info(self, pattern_id: str) -> Dict[str, Any]:
        """Get threshold information for a pattern."""
        return self.threshold_manager.get_threshold_info(pattern_id)

    def get_current_threshold(self, pattern_id: str) -> float:
        """Get current threshold for a pattern."""
        return self.threshold_manager.get_threshold(pattern_id)

    def _try_pattern(
        self, pattern: CompiledPattern, text: str
    ) -> Optional[MatchResult]:
        """Try to match a single pattern against text."""
        # Search for pattern match (not just from start)
        match = pattern.regex.search(text)
        if not match:
            return None

        # Extract slot values
        slots = {}
        for slot in pattern.slots:
            group_value = match.group(slot.group_index)
            if group_value is not None:
                slots[slot.name] = group_value.strip()

        # Calculate confidence score
        confidence = self._calculate_confidence(
            pattern=pattern, slots=slots, matched_text=match.group(0), full_text=text
        )

        return MatchResult(
            pattern_id=pattern.id,
            pattern_priority=pattern.priority,
            slots=slots,
            confidence=confidence,
            raw_text=text,
            matched_text=match.group(0),
        )

    def _calculate_confidence(
        self,
        pattern: CompiledPattern,
        slots: Dict[str, str],
        matched_text: str,
        full_text: str,
    ) -> float:
        """
        Calculate confidence score for a pattern match.

        Uses multiple factors:
        - Pattern coverage (how much of the text was matched)
        - Slot quality (presence of required slots, slot weights)
        - Token overlap between pattern and text
        - Keyword presence
        - Edit distance normalization
        """
        scores = []

        # 1. Pattern Coverage Score
        coverage_score = len(matched_text) / len(full_text)
        scores.append(
            (
                "coverage",
                coverage_score
                * self.scoring_config.get("pattern_coverage_weight", 0.9),
            )
        )

        # 2. Slot Quality Score
        slot_score = self._calculate_slot_score(pattern, slots)
        scores.append(("slots", slot_score))

        # 3. Token Overlap Score
        token_overlap = self._calculate_token_overlap(matched_text, full_text)
        scores.append(
            (
                "token_overlap",
                token_overlap * self.scoring_config.get("token_overlap_weight", 0.8),
            )
        )

        # 4. Keyword Presence Score
        keyword_score = self._calculate_keyword_score(full_text)
        scores.append(
            (
                "keywords",
                keyword_score * self.scoring_config.get("keyword_presence_weight", 0.6),
            )
        )

        # 5. Edit Distance Score (Levenshtein normalization)
        edit_distance_score = self._calculate_edit_distance_score(
            matched_text, full_text
        )
        scores.append(("edit_distance", edit_distance_score))

        # Weighted average of all scores
        total_score = sum(score for name, score in scores)
        confidence = total_score / len(scores)

        # Apply penalties
        confidence = self._apply_penalties(
            confidence, pattern, slots, matched_text, full_text
        )

        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

    def _calculate_slot_score(
        self, pattern: CompiledPattern, slots: Dict[str, str]
    ) -> float:
        """Calculate score based on slot quality and completeness."""
        if not pattern.slots:
            return 1.0

        total_weight = 0.0
        matched_weight = 0.0

        for slot in pattern.slots:
            slot_weight = slot.weight
            total_weight += slot_weight

            if slot.name in slots and slots[slot.name]:
                # Slot is filled
                value = slots[slot.name]

                # Quality bonus for slot type
                if slot.slot_type == "number" and value.isdigit():
                    quality_bonus = 1.0
                elif slot.slot_type == "named" and len(value.split()) <= 3:
                    quality_bonus = 1.0  # Good named slots are short
                elif slot.slot_type == "text":
                    quality_bonus = 0.8  # Text slots are less reliable
                else:
                    quality_bonus = 0.9

                matched_weight += slot_weight * quality_bonus
            elif slot.is_optional:
                # Optional slot not filled - no penalty
                continue
            else:
                # Required slot not filled - penalty
                penalty = self.scoring_config.get("missing_required_slot_penalty", 0.3)
                matched_weight -= slot_weight * penalty

        return matched_weight / total_weight if total_weight > 0 else 0.0

    def _calculate_token_overlap(self, matched_text: str, full_text: str) -> float:
        """Calculate token overlap ratio between matched and full text."""
        matched_tokens = set(matched_text.lower().split())
        full_tokens = set(full_text.lower().split())

        if not full_tokens:
            return 0.0

        overlap = len(matched_tokens.intersection(full_tokens))
        return overlap / len(full_tokens)

    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate score based on presence of important keywords."""
        text_lower = text.lower()

        # Count location indicators (positive)
        location_keywords = self.keywords.get("location_indicators", [])
        location_count = sum(
            1 for keyword in location_keywords if keyword in text_lower
        )
        location_score = min(1.0, location_count / 3)  # Normalize to max 3 keywords

        # Count low importance words (slight negative)
        low_importance = self.keywords.get("low_importance", [])
        low_count = sum(1 for keyword in low_importance if keyword in text_lower)
        low_penalty = min(0.2, low_count * 0.05)  # Small penalty for city names etc

        return max(0.0, location_score - low_penalty)

    def _calculate_edit_distance_score(
        self, matched_text: str, full_text: str
    ) -> float:
        """Calculate normalized edit distance score."""
        if not matched_text or not full_text:
            return 0.0

        # Use Levenshtein distance
        distance = self._levenshtein_distance(matched_text.lower(), full_text.lower())
        max_length = max(len(matched_text), len(full_text))

        if max_length == 0:
            return 1.0

        # Convert to similarity score (1 = identical, 0 = completely different)
        similarity = 1.0 - (distance / max_length)
        return max(0.0, similarity)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _apply_penalties(
        self,
        base_confidence: float,
        pattern: CompiledPattern,
        slots: Dict[str, str],
        matched_text: str,
        full_text: str,
    ) -> float:
        """Apply various penalties to the base confidence score."""
        confidence = base_confidence

        # Penalty for extra unmatched text
        extra_text_ratio = (len(full_text) - len(matched_text)) / len(full_text)
        extra_penalty = extra_text_ratio * self.scoring_config.get(
            "extra_text_penalty", 0.1
        )
        confidence -= extra_penalty

        # Penalty for very short matches
        if len(matched_text.split()) < 2:
            confidence *= 0.8

        # Bonus for exact keyword matches
        exact_match_weight = self.scoring_config.get("exact_match_weight", 1.0)
        if matched_text.lower() == full_text.lower():
            confidence *= exact_match_weight

        return confidence

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns."""
        return {
            "total_patterns": len(self.patterns),
            "patterns_by_priority": {p.priority: p.id for p in self.patterns},
            "slot_types": list(
                set(
                    slot.slot_type
                    for pattern in self.patterns
                    for slot in pattern.slots
                )
            ),
            "keywords": {
                category: len(keywords) for category, keywords in self.keywords.items()
            },
        }


def main():
    """Demo function to show pattern matching."""
    from ..preprocess import preprocess

    matcher = PatternMatcher()

    # Test with real data from train_sample.csv
    test_addresses = [
        "Akarca Mah. Adnan Menderes Cad. 864.Sok. No:15 D.1 K.2",
        "İsmet inönü mahallesi 2001 sokak no:2 Çeşme belediyesi",
        "Bitez mahallesi Adnan Menderes caddesi 1410.sokak no 90A",
        "Dedebaşı mahallesi 6100 sokak no 10 Kat 7 daire 25",
        "Yeni sanayi mahallesi 515 sokak no 53 c3 blok",
    ]

    print("Pattern Matching Demo")
    print("=" * 60)

    for address in test_addresses:
        print(f"\nOriginal: {address}")

        # Preprocess first
        preprocessed = preprocess(address)
        preprocessed_text = preprocessed["text"]
        print(f"Preprocessed: {preprocessed_text}")

        # Find matches
        matches = matcher.match_text(preprocessed_text, max_matches=2)

        if matches:
            for i, match in enumerate(matches, 1):
                print(f"  Match {i}: {match}")
                print(f"    Slots: {match.slots}")
        else:
            print("  No matches found")

        print("-" * 40)


if __name__ == "__main__":
    main()
