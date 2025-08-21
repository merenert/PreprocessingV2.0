"""
Integration tests for the complete preprocessing pipeline.
"""

import json

import pytest

from src.addrnorm.preprocess import preprocess
from src.addrnorm.preprocess.core import (
    clean_punctuation,
    expand_abbreviations,
    normalize_case,
    normalize_unicode,
    tokenize,
)
from src.addrnorm.utils.contracts import AddressOut, ExplanationParsed, MethodEnum


class TestPreprocessingIntegration:
    """Integration tests for the complete preprocessing workflow."""

    def test_complete_pipeline_real_addresses(self):
        """Test the complete pipeline with real Turkish addresses."""
        real_addresses = [
            "İSTANBUL KADIKÖY MODA MAH. PROF. DR. ŞÜKRÜ KAYA CAD. N:12 D:5",
            "ankara çankaya balgat mh. turan güneş blv. no:45/7",
            "İzmir Konak Alsancak Mah. Cumhuriyet Blv. No:123 Kat:3 Daire:8",
            "Antalya Muratpaşa Lara Mah. Barış Manço Cad. N:67 D:12",
            "BURSA NİLÜFER GÖRÜKLE MAH. UNIVERSITY CAD. NO:34/2",
        ]

        for address in real_addresses:
            result = preprocess(address)

            # Verify result structure
            assert isinstance(result, dict)
            assert "text" in result
            assert "tokens" in result

            # Verify text is cleaned
            assert result["text"].strip() == result["text"]
            assert len(result["text"]) > 0

            # Verify tokens
            assert isinstance(result["tokens"], list)
            assert len(result["tokens"]) > 0

            # Verify all tokens are strings
            for token in result["tokens"]:
                assert isinstance(token, str)
                assert len(token) > 0

    def test_pipeline_with_contracts(self):
        """Test integration with data contracts."""
        test_address = "İSTANBUL KADIKÖY MODA MAH. PROF. DR. CAD. N:12"

        # Process address
        processed = preprocess(test_address)

        # Create contracts
        explanation = ExplanationParsed(
            confidence=0.9, method=MethodEnum.PATTERN, warnings=[]
        )

        address_out = AddressOut(
            explanation_raw=test_address,
            explanation_parsed=explanation,
            normalized_address=processed["text"],
        )

        # Verify serialization
        json_output = address_out.to_json()
        parsed_json = json.loads(json_output)

        assert parsed_json["explanation_raw"] == test_address
        assert parsed_json["normalized_address"] == processed["text"]
        assert parsed_json["explanation_parsed"]["confidence"] == 0.9

        # Verify CSV serialization
        csv_row = address_out.to_csv_row()
        assert len(csv_row) == 18  # Should match number of CSV headers
        assert csv_row[13] == test_address  # explanation_raw at index 13
        assert csv_row[14] == processed["text"]  # normalized_address at index 14

    def test_edge_cases_pipeline(self):
        """Test pipeline with edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Only whitespace
            "123",  # Only numbers
            "!!!",  # Only punctuation
            "A",  # Single character
            "İ" * 1000,  # Very long string
            "normal text",  # No Turkish characters
            "çğıöşüÇĞIÖŞÜ",  # All Turkish characters
        ]

        for case in edge_cases:
            result = preprocess(case)

            # Should not crash
            assert isinstance(result, dict)
            assert "text" in result
            assert "tokens" in result

            # Empty input should produce empty output
            if not case.strip():
                assert result["text"] == ""
                assert result["tokens"] == []

    def test_batch_processing(self):
        """Test processing multiple addresses in batch."""
        addresses = [
            "İstanbul Kadıköy",
            "Ankara Çankaya",
            "İzmir Konak",
            "Bursa Nilüfer",
            "Antalya Muratpaşa",
        ]

        results = []
        for address in addresses:
            result = preprocess(address)
            results.append(result)

        # Verify all processed successfully
        assert len(results) == len(addresses)

        # Verify each result is valid
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert len(result["text"]) > 0
            assert len(result["tokens"]) > 0

            # Original should be preserved in some form
            original_lower = addresses[i].lower().replace("i̇", "i")  # Handle Turkish i
            result_lower = result["text"].lower()

            # Should contain key parts of original
            original_words = original_lower.split()
            for word in original_words:
                if len(word) > 2:  # Skip very short words
                    # Word should appear or be abbreviated
                    assert (
                        word in result_lower
                        or any(
                            word.startswith(token.lower()) for token in result["tokens"]
                        )
                        or any(
                            token.lower().startswith(word) for token in result["tokens"]
                        )
                    )

    def test_idempotency(self):
        """Test that processing the same input multiple times gives same result."""
        test_address = "İSTANBUL KADIKÖY MODA MAH. PROF. DR. CAD. N:12"

        result1 = preprocess(test_address)
        result2 = preprocess(test_address)
        result3 = preprocess(test_address)

        # All results should be identical
        assert result1 == result2 == result3

        # Test with already processed text
        processed_text = result1["text"]
        result4 = preprocess(processed_text)

        # Should be stable (processing processed text should change little)
        assert (
            result4["text"] == processed_text
            or len(result4["text"]) >= len(processed_text) * 0.8
        )

    def test_performance_baseline(self):
        """Test that processing is reasonably fast."""
        import time

        test_address = (
            "İSTANBUL KADIKÖY MODA MAH. PROF. DR. ŞÜKRÜ KAYA CAD. N:12 D:5 KAT:3"
        )

        # Measure processing time
        start_time = time.time()
        for _ in range(100):
            preprocess(test_address)
        end_time = time.time()

        # Should process 100 addresses in reasonable time
        total_time = end_time - start_time
        assert (
            total_time < 5.0
        ), f"Processing too slow: {total_time:.2f}s for 100 addresses"

        avg_time = total_time / 100
        assert (
            avg_time < 0.05
        ), f"Average processing time too slow: {avg_time:.4f}s per address"

    def test_memory_usage(self):
        """Test that processing doesn't leak memory."""
        import gc

        test_address = "İSTANBUL KADIKÖY MODA MAH. PROF. DR. CAD. N:12"

        # Get initial memory usage (rough estimate)
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process many addresses
        for _ in range(1000):
            result = preprocess(test_address)
            del result

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Too many objects created: {object_growth}"

    def test_unicode_robustness(self):
        """Test handling of various Unicode characters."""
        unicode_addresses = [
            "İstanbul Kadıköy",  # Turkish
            "Москва Центр",  # Cyrillic
            "北京市朝阳区",  # Chinese
            "العنوان الجديد",  # Arabic
            "עברית כתובת",  # Hebrew
            "Διεύθυνση Ελλάδα",  # Greek
            "مکان جدید",  # Persian
        ]

        for address in unicode_addresses:
            result = preprocess(address)

            # Should not crash
            assert isinstance(result, dict)
            assert "text" in result

            # Result should be valid UTF-8
            result_encoded = result["text"].encode("utf-8")
            result_decoded = result_encoded.decode("utf-8")
            assert result_decoded == result["text"]


class TestModuleStepByStep:
    """Test each step of the preprocessing pipeline individually."""

    def test_step_by_step_processing(self):
        """Test each preprocessing step individually."""
        original = "İSTANBUL KADIKÖY MODA MAH. PROF. DR. CAD. N:12 D:5"

        # Step 1: Case normalization
        step1 = normalize_case(original)
        assert step1 != original  # Should be different
        assert step1.lower() == step1  # Should be lowercase

        # Step 2: Unicode normalization
        step2 = normalize_unicode(step1)
        # Should be valid Unicode
        assert isinstance(step2, str)

        # Step 3: Abbreviation expansion
        step3 = expand_abbreviations(step2)
        # Should expand abbreviations
        assert "mah" not in step3.lower() or "mahalle" in step3.lower()

        # Step 4: Punctuation cleaning
        step4 = clean_punctuation(step3)
        # Should remove/normalize punctuation
        assert ":" not in step4 or step4.count(":") <= step3.count(":")

        # Step 5: Tokenization
        tokens = tokenize(step4)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Full pipeline
        full_result = preprocess(original)
        assert full_result["text"] == step4
        assert full_result["tokens"] == tokens

    def test_individual_functions_with_edge_cases(self):
        """Test individual functions with edge cases."""

        # Test normalize_case with edge cases
        assert normalize_case("") == ""
        assert normalize_case("ABC") == "abc"
        assert normalize_case("İSTANBUL") == "istanbul"

        # Test normalize_unicode with edge cases
        assert normalize_unicode("") == ""
        assert normalize_unicode("test") == "test"

        # Test expand_abbreviations with edge cases
        assert expand_abbreviations("") == ""
        assert expand_abbreviations("mah.") != "mah."

        # Test clean_punctuation with edge cases
        assert clean_punctuation("") == ""
        assert clean_punctuation("a.b.c") != "a.b.c"

        # Test tokenize with edge cases
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize("a b c") == ["a", "b", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
