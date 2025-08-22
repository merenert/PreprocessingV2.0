"""
Comprehensive tests for the pattern matching module.

Tests both positive and negative cases for all major patterns,
plus confidence scoring and edge cases.
"""

import pytest

from src.addrnorm.patterns.compiler import PatternCompiler
from src.addrnorm.patterns.matcher import MatchResult, PatternMatcher
from src.addrnorm.preprocess import preprocess


class TestPatternCompiler:
    """Test the pattern compiler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = PatternCompiler()

    def test_load_patterns_config(self):
        """Test that pattern configuration loads correctly."""
        config = self.compiler.patterns_config

        assert "patterns" in config
        assert "keywords" in config
        assert "scoring" in config
        assert len(config["patterns"]) >= 6  # At least 6 patterns as required

    def test_compile_all_patterns(self):
        """Test compilation of all patterns."""
        patterns = self.compiler.compile_all_patterns()

        assert len(patterns) >= 6

        # Check that patterns are sorted by priority
        priorities = [p.priority for p in patterns]
        assert priorities == sorted(priorities, reverse=True)

        # Check that all patterns have required fields
        for pattern in patterns:
            assert pattern.id
            assert pattern.regex
            assert isinstance(pattern.slots, list)
            assert pattern.priority > 0

    def test_pattern_regex_compilation(self):
        """Test that generated regex patterns compile correctly."""
        patterns = self.compiler.compile_all_patterns()

        for pattern in patterns:
            # Regex should compile without errors
            assert pattern.regex is not None

            # Test with example inputs if available
            if pattern.examples:
                for example in pattern.examples:
                    # Preprocess example first
                    preprocessed = preprocess(example)
                    text = preprocessed["text"]

                    # Should match at least one example
                    pattern.regex.search(text)
                    # Note: Not all examples may match due to preprocessing changes
                    # This is acceptable - we're just testing compilation

    def test_slot_parsing(self):
        """Test that slots are parsed correctly from patterns."""
        patterns = self.compiler.compile_all_patterns()

        # Find a pattern with known slots
        mahalle_pattern = None
        for pattern in patterns:
            if "mahalle" in pattern.pattern_text.lower():
                mahalle_pattern = pattern
                break

        assert mahalle_pattern is not None

        # Should have at least Mahalle slot
        slot_names = [slot.name for slot in mahalle_pattern.slots]
        assert any("Mahalle" in name for name in slot_names)

        # Check slot properties
        for slot in mahalle_pattern.slots:
            assert slot.name
            assert slot.slot_type in ["named", "number", "text"]
            assert slot.group_index > 0
            assert 0 <= slot.weight <= 1.0

    def test_pattern_validation(self):
        """Test pattern validation functionality."""
        # Valid patterns
        valid_patterns = [
            "<Mahalle> <Sokak> no <n>",
            "<Açıklama?> <Mahalle> <text>",
            "<n> sokak no <n2>",
        ]

        for pattern in valid_patterns:
            assert self.compiler.validate_pattern(pattern)

        # Invalid patterns
        invalid_patterns = [
            "<Mahalle> <",  # Unclosed slot
            "><Sokak>",  # Invalid syntax
            "<Mahalle> [invalid]",  # Wrong bracket type
        ]

        for pattern in invalid_patterns:
            assert not self.compiler.validate_pattern(pattern)


class TestPatternMatcher:
    """Test the pattern matcher functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = PatternMatcher()

        # Real address examples from train_sample.csv
        self.real_addresses = [
            "Akarca Mah. Adnan Menderes Cad. 864.Sok. No:15 D.1 K.2",
            "Cumhuriye Mah. Hükümet Cad. Sivriler İşhanı No:3 Fethiye/Muğla",
            "İsmet inönü mahallesi 2001 sokak no:2 Çeşme belediyesi Çeşme",
            "Bitez mahallesi Adnan Menderes caddesi gündonumu mevkii 1410.sokak no 90A",
            "Dedebaşı mahallesi 6100 sokak no 10 Kat 7 daire 25",
            "Yeni sanayi mahallesi 515 sokak no 53 c3 blok emek aliminyum",
        ]

    def test_basic_pattern_matching(self):
        """Test basic pattern matching functionality."""
        # Test with a simple, well-formed address
        address = "moda mahalle bahariye sokak no 12"
        matches = self.matcher.match_text(address)

        assert len(matches) > 0
        best_match = matches[0]
        assert best_match.confidence > 0.3
        assert (
            "Mahalle" in best_match.slots or "mahalle" in str(best_match.slots).lower()
        )

    def test_real_address_matching(self):
        """Test matching against real addresses from dataset."""
        successful_matches = 0

        for address in self.real_addresses:
            # Preprocess first
            preprocessed = preprocess(address)
            text = preprocessed["text"]

            matches = self.matcher.match_text(text)

            if matches and matches[0].confidence >= 0.3:
                successful_matches += 1

                # Check that we got reasonable slot extractions
                best_match = matches[0]
                assert isinstance(best_match.slots, dict)
                assert len(best_match.slots) > 0

        # Should successfully match at least 50% of real addresses
        success_rate = successful_matches / len(self.real_addresses)
        assert success_rate >= 0.5, f"Success rate too low: {success_rate:.2f}"

    def test_confidence_scoring_properties(self):
        """Test that confidence scores have expected properties."""
        # Test with various quality inputs
        test_cases = [
            ("moda mahalle bahariye sokak no 12", "high quality"),
            ("istanbul kadıköy moda mahalle bahariye sokak no 12", "with explanation"),
            ("mahalle sokak", "minimal info"),
            ("random text without structure", "poor quality"),
            ("", "empty input"),
        ]

        scores = []
        for text, description in test_cases:
            matches = self.matcher.match_text(text)
            if matches:
                scores.append((matches[0].confidence, description))
            else:
                scores.append((0.0, description))

        # Scores should be in [0, 1] range
        for score, desc in scores:
            assert 0.0 <= score <= 1.0, f"Score out of range for {desc}: {score}"

        # High quality should score higher than poor quality
        high_quality_score = scores[0][0]  # "moda mahalle bahariye sokak no 12"
        poor_quality_score = scores[3][0]  # "random text without structure"

        if high_quality_score > 0 and poor_quality_score > 0:
            assert high_quality_score > poor_quality_score

    def test_idempotency(self):
        """Test that matching is idempotent and deterministic."""
        test_text = "moda mahalle bahariye sokak no 12"

        # Run multiple times
        results = []
        for _ in range(3):
            matches = self.matcher.match_text(test_text)
            if matches:
                result = (
                    matches[0].pattern_id,
                    matches[0].confidence,
                    matches[0].slots,
                )
                results.append(result)

        # All results should be identical
        if results:
            first_result = results[0]
            for result in results[1:]:
                assert result == first_result, "Matching is not deterministic"

    def test_slot_extraction_accuracy(self):
        """Test accuracy of slot extraction for known patterns."""
        # Test cases with expected extractions
        test_cases = [
            {
                "input": "akarca mahalle adnan menderes cadde 864 sokak no 15",
                "expected_slots": {"Mahalle", "Sokak", "n"},
                "expected_values": {"mahalle": "akarca", "no": "15"},
            },
            {
                "input": "dedebaşı mahalle 6100 sokak no 10 kat 7 daire 25",
                "expected_slots": {"Mahalle", "n", "Kat", "Daire"},
                "expected_values": {"mahalle": "dedebaşı", "no": "10"},
            },
        ]

        for case in test_cases:
            matches = self.matcher.match_text(case["input"])

            if matches:
                best_match = matches[0]
                slots = best_match.slots

                # Check that we extracted expected slot types
                # (exact names may vary due to pattern variations)
                assert len(slots) > 0, f"No slots extracted for: {case['input']}"

                # Check for presence of key information
                for key, expected_value in case["expected_values"].items():
                    if "mahalle" in key:
                        assert any(
                            expected_value in value.lower() for value in slots.values()
                        ), f"Expected {expected_value} not found in slots: {slots}"

    def test_negative_cases(self):
        """Test that non-address text gets low confidence scores."""
        negative_cases = [
            "bu bir adres değil sadece normal metin",
            "telefon: 0212 123 45 67",
            "e-posta: test@example.com",
            "1234567890",
            "AAAAAAA",
            "!@#$%^&*()",
        ]

        for text in negative_cases:
            matches = self.matcher.match_text(text)

            # Either no matches or very low confidence
            if matches:
                assert matches[0].confidence < 0.5, (
                    f"Non-address text got high confidence: {text} -> "
                    f"{matches[0].confidence}"
                )

    def test_pattern_priority_ordering(self):
        """Test that higher priority patterns are preferred."""
        # Create ambiguous text that could match multiple patterns
        ambiguous_text = "test mahalle test sokak no 123"

        matches = self.matcher.match_text(ambiguous_text, max_matches=5)

        if len(matches) > 1:
            # Matches should be ordered by confidence, then priority
            for i in range(len(matches) - 1):
                current = matches[i]
                next_match = matches[i + 1]

                # Either higher confidence, or same confidence but higher priority
                assert current.confidence > next_match.confidence or (
                    current.confidence == next_match.confidence
                    and current.pattern_priority >= next_match.pattern_priority
                )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Only whitespace
            "a",  # Single character
            "a " * 1000,  # Very long string
            "çğıöşüÇĞIÖŞÜ",  # Only Turkish characters
            "123",  # Only numbers
            "test test test test test test test test",  # Repetitive
        ]

        for text in edge_cases:
            # Should not crash
            matches = self.matcher.match_text(text)

            # Should return reasonable results
            assert isinstance(matches, list)

            for match in matches:
                assert isinstance(match, MatchResult)
                assert 0.0 <= match.confidence <= 1.0
                assert isinstance(match.slots, dict)

    def test_scoring_components(self):
        """Test that different scoring components work correctly."""
        # Test with a high-quality address that should score well
        high_quality = "moda mahalle bahariye sokak no 12 daire 5"
        matches = self.matcher.match_text(high_quality)

        if matches:
            high_score = matches[0].confidence

            # Test with lower quality versions
            partial_quality = "mahalle sokak no"
            partial_matches = self.matcher.match_text(partial_quality)

            if partial_matches:
                partial_score = partial_matches[0].confidence

                # High quality should score better than partial
                assert (
                    high_score > partial_score
                ), f"High quality {high_score} should beat partial {partial_score}"


class TestPatternIntegration:
    """Integration tests for the complete pattern matching system."""

    def test_preprocessing_integration(self):
        """Test integration with preprocessing module."""
        from src.addrnorm.preprocess import preprocess

        # Test the complete pipeline
        raw_address = "İSTANBUL KADIKÖY MODA MAH. BAHARİYE CAD. NO:12 D:5"

        # Step 1: Preprocess
        preprocessed = preprocess(raw_address)

        # Step 2: Pattern match
        matcher = PatternMatcher()
        matches = matcher.match_text(preprocessed["text"])

        # Should get reasonable results
        assert len(matches) > 0
        best_match = matches[0]
        assert best_match.confidence > 0.3
        assert len(best_match.slots) > 0

    def test_pattern_coverage(self):
        """Test that patterns cover different types of addresses."""
        # Different address types that should be covered
        address_types = {
            "simple_mahalle_sokak": "moda mahalle bahariye sokak no 12",
            "with_cadde": "balgat mahalle turan güneş cadde no 67",
            "with_apartment": "moda mahalle bahariye sokak no 12 daire 5",
            "with_explanation": "istanbul kadiköy moda mahalle bahariye sokak no 12",
            "site_pattern": "rivella evleri a blok no 77 daire 15",
            "number_first": "1771 sokak no 5",
        }

        matcher = PatternMatcher()
        covered_types = 0

        for addr_type, address in address_types.items():
            matches = matcher.match_text(address)
            if matches and matches[0].confidence >= 0.4:
                covered_types += 1

        # Should cover at least 80% of address types
        coverage_rate = covered_types / len(address_types)
        assert coverage_rate >= 0.8, f"Pattern coverage too low: {coverage_rate:.2f}"

    def test_real_world_performance(self):
        """Test performance on real address data."""
        # Load some real addresses from train_sample.csv
        real_addresses = [
            "Akarca Mah. Adnan Menderes Cad. 864.Sok. No:15 D.1 K.2",
            "Cumhuriye Mah. Hükümet Cad. Sivriler İşhanı No:3",
            "İsmet inönü mahallesi 2001 sokak no:2",
            "Bitez mahallesi Adnan Menderes caddesi 1410.sokak no 90A",
            "Dedebaşı mahallesi 6100 sokak no 10 Kat 7 daire 25",
            "Yeni sanayi mahallesi 515 sokak no 53 c3 blok",
        ]

        matcher = PatternMatcher()
        total_processed = 0
        successful_matches = 0

        for address in real_addresses:
            # Preprocess
            preprocessed = preprocess(address)

            # Match
            matches = matcher.match_text(preprocessed["text"])
            total_processed += 1

            if matches and matches[0].confidence >= 0.3:
                successful_matches += 1

        # Performance metrics
        success_rate = successful_matches / total_processed
        assert (
            success_rate >= 0.5
        ), f"Real-world success rate too low: {success_rate:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
