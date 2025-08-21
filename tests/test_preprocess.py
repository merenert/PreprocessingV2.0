"""
Comprehensive tests for the preprocess module.
"""

import pytest

from src.addrnorm.preprocess import (
    clean_punctuation,
    expand_abbreviations,
    load_abbreviations,
    normalize_case,
    normalize_unicode,
    preprocess,
    tokenize,
)


class TestNormalizeCase:
    """Test case normalization functionality."""

    def test_basic_lowercase(self):
        """Test basic lowercase conversion."""
        assert normalize_case("ISTANBUL") == "istanbul"
        assert normalize_case("Ankara") == "ankara"

    def test_turkish_characters(self):
        """Test Turkish character normalization."""
        assert normalize_case("İSTANBUL") == "istanbul"
        assert normalize_case("ĞÜŞÖÇ") == "ğüşöç"
        # Note: We don't convert all I to ı, only specifically İ to i
        assert normalize_case("IĞDIR") == "iğdir"

    def test_mixed_case(self):
        """Test mixed case text."""
        assert normalize_case("İstanbul Kadıköy") == "istanbul kadıköy"
        assert normalize_case("ÇANKAYA mah.") == "çankaya mah."

    def test_empty_and_none(self):
        """Test edge cases."""
        assert normalize_case("") == ""
        assert normalize_case(None) is None


class TestNormalizeUnicode:
    """Test Unicode normalization functionality."""

    def test_nfc_normalization(self):
        """Test NFC normalization."""
        # Test composed vs decomposed characters
        text_composed = "çğşüöı"
        text_decomposed = "c\u0327g\u0306s\u0327u\u0308o\u0308ı"

        result = normalize_unicode(text_decomposed)
        assert result == text_composed

    def test_turkish_dotted_i(self):
        """Test Turkish dotted i variations."""
        assert normalize_unicode("i̇stanbul") == "istanbul"
        assert normalize_unicode("İstanbul") == "istanbul"

    def test_combining_characters(self):
        """Test combining character removal."""
        text_with_combining = "test\u0307"  # with combining dot above
        result = normalize_unicode(text_with_combining)
        assert "\u0307" not in result

    def test_empty_input(self):
        """Test empty input."""
        assert normalize_unicode("") == ""
        assert normalize_unicode(None) is None


class TestLoadAbbreviations:
    """Test abbreviation loading functionality."""

    def test_load_default_abbreviations(self):
        """Test loading default abbreviations."""
        abbrevs = load_abbreviations()
        assert isinstance(abbrevs, dict)
        assert "mh" in abbrevs
        assert abbrevs["mh"] == "mahalle"
        assert "cd" in abbrevs
        assert abbrevs["cd"] == "cadde"

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        abbrevs = load_abbreviations("nonexistent.yaml")
        assert abbrevs == {}


class TestExpandAbbreviations:
    """Test abbreviation expansion functionality."""

    @pytest.fixture
    def sample_mapping(self):
        """Sample abbreviation mapping for testing."""
        return {
            "mh": "mahalle",
            "cd": "cadde",
            "no": "numara",
            "n:": "numara",
            "apt": "apartman",
        }

    def test_basic_expansion(self, sample_mapping):
        """Test basic abbreviation expansion."""
        text = "Moda mh. Bahariye cd. no 12"
        result = expand_abbreviations(text, sample_mapping)
        assert result == "Moda mahalle. Bahariye cadde. numara 12"

    def test_case_insensitive_expansion(self, sample_mapping):
        """Test case-insensitive expansion."""
        text = "MODA MH. bahariye CD."
        result = expand_abbreviations(text, sample_mapping)
        assert result == "MODA mahalle. bahariye cadde."

    def test_special_punctuation_expansion(self, sample_mapping):
        """Test expansion with special punctuation like 'n:'."""
        text = "Ev n: 15 apt"
        result = expand_abbreviations(text, sample_mapping)
        assert result == "Ev numara 15 apartman"

    def test_word_boundary_respect(self, sample_mapping):
        """Test that word boundaries are respected."""
        text = "Mehmet cd."  # 'mh' shouldn't be expanded here
        result = expand_abbreviations(text, sample_mapping)
        assert result == "Mehmet cadde."

    def test_longest_match_first(self):
        """Test that longest abbreviations are matched first."""
        mapping = {"cd": "cadde", "cad": "cadde_full"}
        text = "test cad"
        result = expand_abbreviations(text, mapping)
        assert result == "test cadde_full"

    def test_empty_input(self, sample_mapping):
        """Test empty input."""
        assert expand_abbreviations("", sample_mapping) == ""
        assert expand_abbreviations(None, sample_mapping) is None


class TestCleanPunctuation:
    """Test punctuation cleaning functionality."""

    def test_basic_punctuation_removal(self):
        """Test basic punctuation replacement."""
        text = "Moda,mah./Bahariye-cd."
        result = clean_punctuation(text)
        expected = "Moda mah Bahariye cd"
        assert result == expected

    def test_multiple_spaces_cleanup(self):
        """Test multiple spaces are cleaned up."""
        text = "test   multiple    spaces"
        result = clean_punctuation(text)
        assert result == "test multiple spaces"

    def test_leading_trailing_spaces(self):
        """Test leading and trailing spaces are removed."""
        text = "  test  "
        result = clean_punctuation(text)
        assert result == "test"

    def test_brackets_and_parentheses(self):
        """Test bracket and parentheses removal."""
        text = "test (inside) [brackets] {braces}"
        result = clean_punctuation(text)
        assert result == "test inside brackets braces"


class TestTokenize:
    """Test tokenization functionality."""

    def test_basic_tokenization(self):
        """Test basic space-separated tokenization."""
        text = "istanbul kadıköy moda"
        result = tokenize(text)
        assert result == ["istanbul", "kadıköy", "moda"]

    def test_multi_separator_tokenization(self):
        """Test tokenization with multiple separators."""
        text = "istanbul,kadıköy/moda-mahalle"
        result = tokenize(text)
        assert result == ["istanbul", "kadıköy", "moda", "mahalle"]

    def test_empty_tokens_filtered(self):
        """Test that empty tokens are filtered out."""
        text = "istanbul  kadıköy   moda"
        result = tokenize(text)
        assert result == ["istanbul", "kadıköy", "moda"]

    def test_idempotency(self):
        """Test that tokenization is idempotent."""
        text = "istanbul kadıköy moda"
        tokens1 = tokenize(text)
        tokens2 = tokenize(" ".join(tokens1))
        assert tokens1 == tokens2

    def test_empty_input(self):
        """Test empty input."""
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestPreprocess:
    """Test complete preprocessing pipeline."""

    def test_complete_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "İSTANBUL, KADIKÖY MH. BAHARİYE CD. NO:12 APT."
        result = preprocess(text)

        assert "text" in result
        assert "tokens" in result
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) > 0

    def test_preprocessing_steps(self):
        """Test individual preprocessing steps."""
        text = "MODA MH. BAHARİYE CD."
        result = preprocess(text)

        # Should be lowercase
        assert result["text"].islower()
        # Should expand abbreviations
        assert "mahalle" in result["text"]
        assert "cadde" in result["text"]

    def test_selective_preprocessing(self):
        """Test selective preprocessing with kwargs."""
        text = "MODA MH."

        # Disable abbreviation expansion
        result = preprocess(text, expand_abbreviations=False)
        assert "mh" in result["text"]
        assert "mahalle" not in result["text"]

        # Disable case normalization
        result = preprocess(text, normalize_case=False)
        assert "MODA" in result["text"]

    def test_custom_abbreviation_mapping(self):
        """Test preprocessing with custom abbreviation mapping."""
        text = "test xyz"
        custom_mapping = {"xyz": "custom"}

        result = preprocess(text, abbreviation_mapping=custom_mapping)
        assert "custom" in result["text"]

    def test_idempotency_assertion(self):
        """Test that preprocessing ensures tokenization idempotency."""
        text = "istanbul kadıköy moda"
        result = preprocess(text)

        # Should not raise assertion error
        assert result is not None


class TestRealWorldExamples:
    """Test with real-world Turkish address examples."""

    def test_example_1(self):
        """Test: İstanbul, Kadıköy, Moda Mah., Bahariye Cad. No:12 D:5"""
        text = "İstanbul, Kadıköy, Moda Mah., Bahariye Cad. No:12 D:5"
        result = preprocess(text)

        assert "istanbul" in result["text"]
        assert "kadıköy" in result["text"]
        assert "mahalle" in result["text"]
        assert "cadde" in result["text"]
        assert "numara" in result["text"]
        assert "daire" in result["text"]

    def test_example_2(self):
        """Test: Ankara, Çankaya, Kızılay Mah., Atatürk Bulvarı No:100"""
        text = "Ankara, Çankaya, Kızılay Mah., Atatürk Bulvarı No:100"
        result = preprocess(text)

        assert "ankara" in result["text"]
        assert "çankaya" in result["text"]
        assert "kızılay" in result["text"]
        assert "mahalle" in result["text"]
        assert "atatürk" in result["text"]
        assert "bulvar" in result["text"]
        assert "numara" in result["text"]

    def test_example_3(self):
        """Test: İzmir, Konak, Alsancak Mah., Kıbrıs Şehitleri Cad. No:45"""
        text = "İzmir, Konak, Alsancak Mah., Kıbrıs Şehitleri Cad. No:45"
        result = preprocess(text)

        assert "izmir" in result["text"]
        assert "konak" in result["text"]
        assert "alsancak" in result["text"]
        assert "mahalle" in result["text"]
        assert "kıbrıs" in result["text"]
        assert "şehitleri" in result["text"]
        assert "cadde" in result["text"]
        assert "numara" in result["text"]

    def test_abbreviation_edge_cases(self):
        """Test edge cases with abbreviations."""
        cases = [
            ("mh.", "mahalle"),
            ("n:", "numara"),
            ("cd.", "cadde"),
            ("sk.", "sokak"),
            ("blv.", "bulvar"),
            ("apt.", "apartman"),
            ("d:", "daire"),
        ]

        for abbrev, expected in cases:
            text = f"test {abbrev} something"
            result = preprocess(text)
            assert (
                expected in result["text"]
            ), f"Failed to expand {abbrev} to {expected}"

    def test_complex_address_with_punctuation(self):
        """Test complex address with lots of punctuation."""
        text = "İstanbul/Kadıköy-Moda Mh.(Bahariye Cd.) N:12/5 Apt.Blok:A"
        result = preprocess(text)

        # Should be cleaned and expanded
        assert "istanbul" in result["text"]
        assert "kadıköy" in result["text"]
        assert "moda" in result["text"]
        assert "mahalle" in result["text"]
        assert "bahariye" in result["text"]
        assert "cadde" in result["text"]
        assert "numara" in result["text"]
        assert "apartman" in result["text"]
        assert "blok" in result["text"]

    def test_unicode_normalization_real_cases(self):
        """Test Unicode normalization with real Turkish text."""
        # Text with various Unicode representations
        text = "İç Çeşme Mevkii, Büyükçekmece"
        result = preprocess(text)

        assert "iç" in result["text"]
        assert "çeşme" in result["text"]
        # "mevkii" gets normalized to "mevki" per our abbreviations
        assert "mevki" in result["text"]
        assert "büyükçekmece" in result["text"]

    def test_numbers_preservation(self):
        """Test that numbers are preserved in preprocessing."""
        text = "Sokak No:123 Blok:45 Daire:67"
        result = preprocess(text)

        assert "123" in result["text"]
        assert "45" in result["text"]
        assert "67" in result["text"]

    def test_tokenization_quality(self):
        """Test tokenization produces meaningful tokens."""
        text = "İstanbul Kadıköy Moda Mahalle Bahariye Cadde Numara 12"
        result = preprocess(text)

        tokens = result["tokens"]

        # Should have meaningful tokens
        assert "istanbul" in tokens
        assert "kadıköy" in tokens
        assert "moda" in tokens
        assert "mahalle" in tokens
        assert "bahariye" in tokens
        assert "cadde" in tokens
        assert "numara" in tokens
        assert "12" in tokens

        # Should not have empty or meaningless tokens
        assert "" not in tokens
        assert " " not in tokens
