"""
Tests for simplified explanation processing.
"""

import pytest
from addrnorm.explanation import process_explanation, parse_explanation


class TestExplanationProcessing:
    """Test explanation text processing."""

    def test_process_explanation_basic(self):
        """Test basic explanation processing."""
        result = process_explanation("Migros yanı")
        assert result == "Migros yanı"

    def test_process_explanation_whitespace(self):
        """Test whitespace cleaning."""
        result = process_explanation("  Migros   yanı  ")
        assert result == "Migros yanı"

    def test_process_explanation_empty(self):
        """Test empty input."""
        assert process_explanation("") == ""
        assert process_explanation(None) == ""
        assert process_explanation("   ") == ""

    def test_process_explanation_complex(self):
        """Test complex explanation."""
        result = process_explanation("Amorium Hotel   karşısında  ")
        assert result == "Amorium Hotel karşısında"

    def test_parse_explanation_compatibility(self):
        """Test backward compatibility function."""
        result = parse_explanation("Şekerbank ATM yanı")
        assert result == "Şekerbank ATM yanı"

    def test_process_explanation_special_chars(self):
        """Test with Turkish characters."""
        result = process_explanation("İstanbul Üniversitesi önü")
        assert result == "İstanbul Üniversitesi önü"

    def test_process_explanation_multiple_spaces(self):
        """Test multiple consecutive spaces."""
        result = process_explanation("McDonald's     karşısı")
        assert result == "McDonald's karşısı"

    def test_process_explanation_non_string(self):
        """Test non-string input."""
        assert process_explanation(123) == ""
        assert process_explanation([]) == ""
        assert process_explanation({}) == ""
