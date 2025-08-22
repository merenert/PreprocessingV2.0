"""
Tests for explanation parsing functionality.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from addrnorm.explainer.rules import (
    ExplanationParser,
    TurkishSuffixProcessor,
    load_test_explanations,
    parse_explanation,
)


class TestTurkishSuffixProcessor:
    """Test Turkish suffix processing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.processor = TurkishSuffixProcessor()

    def test_genitive_suffix_removal(self):
        """Test removal of genitive case suffixes."""
        assert self.processor.strip_suffixes("marketin") == "market"
        assert self.processor.strip_suffixes("okulun") == "okul"
        assert self.processor.strip_suffixes("parkın") == "park"
        assert self.processor.strip_suffixes("caminin") == "cami"

    def test_possessive_suffix_removal(self):
        """Test removal of possessive suffixes."""
        assert self.processor.strip_suffixes("hastanesi") == "hastane"
        assert self.processor.strip_suffixes("bankası") == "banka"
        assert self.processor.strip_suffixes("oteli") == "otel"

    def test_no_oversuffixing(self):
        """Test that words don't become too short."""
        # Should not remove if word becomes too short
        assert len(self.processor.strip_suffixes("an")) >= 2
        assert len(self.processor.strip_suffixes("un")) >= 2

    def test_compound_suffixes(self):
        """Test removal of multiple suffixes."""
        # "çarşının" → "çarşı" + "nın" → "çarşı"
        assert self.processor.strip_suffixes("çarşının") == "çarşı"


class TestExplanationParser:
    """Test core explanation parsing logic."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = ExplanationParser()

    def test_clean_text(self):
        """Test text cleaning functionality."""
        assert self.parser._clean_text("  Hotel Karşısı!  ") == "hotel karşısı"
        assert self.parser._clean_text("Market'in yanı...") == "market in yanı"

    def test_relation_extraction(self):
        """Test spatial relation extraction."""
        relation, landmark = self.parser._extract_relation_and_landmark("hotel karşısı")
        assert relation == "across_from"
        assert landmark == "hotel"

        relation, landmark = self.parser._extract_relation_and_landmark("marketin yanı")
        assert relation == "next_to"
        assert landmark == "market"

    def test_complex_landmarks(self):
        """Test multi-word landmark extraction."""
        relation, landmark = self.parser._extract_relation_and_landmark(
            "amorium hotel karşısı"
        )
        assert relation == "across_from"
        assert landmark == "amorium hotel"

    def test_confidence_calculation(self):
        """Test confidence scoring."""
        # High confidence for known landmark types
        confidence = self.parser._calculate_confidence(
            "across_from", "hotel", "hotel karşısı"
        )
        assert confidence >= 0.8

        # Lower confidence for unknown landmarks
        confidence = self.parser._calculate_confidence("next_to", "xyz", "xyz yanı")
        assert confidence < 0.8


class TestParseExplanation:
    """Test main parse_explanation function."""

    def test_basic_parsing(self):
        """Test basic explanation parsing."""
        result = parse_explanation("Amorium Hotel karşısı")
        assert result is not None
        assert result.type == "landmark"
        assert result.name == "amorium hotel"
        assert result.relation == "across_from"
        assert result.confidence > 0.7
        assert result.original_text == "Amorium Hotel karşısı"

    def test_genitive_case_handling(self):
        """Test handling of genitive case."""
        result = parse_explanation("marketin yanı")
        assert result is not None
        assert result.name == "market"
        assert result.relation == "next_to"

    def test_various_relations(self):
        """Test different spatial relations."""
        test_cases = [
            ("otel karşısı", "across_from"),
            ("park yanı", "next_to"),
            ("okul arkası", "behind"),
            ("hastane önü", "in_front"),
            ("banka köşesi", "corner"),
            ("terminal girişi", "entrance"),
        ]

        for text, expected_relation in test_cases:
            result = parse_explanation(text)
            assert result is not None, f"Failed to parse: {text}"
            assert result.relation == expected_relation, f"Wrong relation for: {text}"

    def test_invalid_inputs(self):
        """Test invalid or empty inputs."""
        assert parse_explanation("") is None
        assert parse_explanation("   ") is None
        assert parse_explanation("sadece tek kelime") is None
        assert parse_explanation("kelimeler relation yok") is None

    def test_confidence_scores(self):
        """Test confidence scoring system."""
        # Known landmark should have high confidence
        result = parse_explanation("hotel karşısı")
        assert result.confidence >= 0.8

        # Unknown landmark should have lower confidence
        result = parse_explanation("bilinmeyenyer yanı")
        assert result.confidence < 0.8


class TestRealWorldExamples:
    """Test with real-world explanation examples."""

    def test_hotel_examples(self):
        """Test hotel-related explanations."""
        examples = [
            "Amorium Hotel karşısı",
            "Grand Otel yanında",
            "Hilton Hotel arkasında",
        ]

        for example in examples:
            result = parse_explanation(example)
            assert result is not None, f"Failed: {example}"
            assert result.type == "landmark"
            assert "hotel" in result.name or "otel" in result.name

    def test_market_examples(self):
        """Test market-related explanations."""
        examples = ["marketin yanı", "Migros karşısında", "BİM arkası"]

        for example in examples:
            result = parse_explanation(example)
            assert result is not None, f"Failed: {example}"
            assert result.type == "landmark"

    def test_possessive_forms(self):
        """Test various possessive forms."""
        examples = [
            ("caminin karşısı", "cami"),
            ("okulun önü", "okul"),
            ("hastanesi yanı", "hastane"),
            ("çarşının içi", "çarşı"),
        ]

        for text, expected_root in examples:
            result = parse_explanation(text)
            assert result is not None, f"Failed: {text}"
            assert (
                expected_root in result.name
            ), f"Expected {expected_root} in {result.name}"


class TestEndToEndAccuracy:
    """End-to-end accuracy testing with test dataset."""

    def test_accuracy_on_test_set(self):
        """Test accuracy on explanations.txt dataset."""
        # Load test examples
        test_file = (
            Path(__file__).parent.parent / "data" / "examples" / "explanations.txt"
        )

        if not test_file.exists():
            pytest.skip("Test data file not found")

        explanations = load_test_explanations(str(test_file))

        # Filter out comments and empty lines
        valid_explanations = [
            line for line in explanations if line and not line.startswith("#")
        ]

        if len(valid_explanations) < 10:
            pytest.skip("Not enough test data")

        # Test parsing success rate
        successful_parses = 0

        for explanation in valid_explanations:
            result = parse_explanation(explanation)
            if result and result.confidence >= 0.5:
                successful_parses += 1

        # Calculate accuracy
        accuracy = successful_parses / len(valid_explanations)
        print(
            f"\nAccuracy: {accuracy:.2%} "
            f"({successful_parses}/{len(valid_explanations)})"
        )

        # Requirement: ≥90% accuracy on small test pool
        assert accuracy >= 0.90, f"Accuracy {accuracy:.2%} below required 90%"

    def test_specific_challenging_cases(self):
        """Test specific challenging cases."""
        challenging_cases = [
            "köprünün altı",  # Complex suffix + spatial relation
            "tünelin girişi",  # Genitive + entrance
            "alışveriş merkezi yanı",  # Multi-word compound
            "otobüs durağı karşısı",  # Multi-word with spaces
        ]

        successful = 0
        for case in challenging_cases:
            result = parse_explanation(case)
            if result and result.confidence >= 0.5:
                successful += 1
                print(f"✅ {case} → {result.name} ({result.relation})")
            else:
                print(f"❌ {case} → Failed")

        accuracy = successful / len(challenging_cases)
        assert accuracy >= 0.75, f"Challenging cases accuracy {accuracy:.2%} too low"


def main():
    """Run all tests manually."""
    print("🧪 Running Explanation Parser Tests")
    print("=" * 50)

    # Run a few key tests manually
    test_cases = [
        "Amorium Hotel karşısı",
        "marketin yanı",
        "parkın arkası",
        "hastane önü",
        "caminin karşısında",
        "çarşının içi",
    ]

    successful = 0
    for case in test_cases:
        result = parse_explanation(case)
        if result:
            print(f"✅ '{case}'")
            print(f"   → {result.name} | {result.relation} | {result.confidence:.3f}")
            successful += 1
        else:
            print(f"❌ '{case}' - Failed to parse")

    accuracy = successful / len(test_cases)
    print(f"\nManual Test Accuracy: {accuracy:.2%}")
    print(f"Required: ≥90% → {'✅ PASS' if accuracy >= 0.9 else '❌ FAIL'}")


if __name__ == "__main__":
    main()
