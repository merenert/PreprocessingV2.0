"""
Text preprocessing module for Turkish address normalization.
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import yaml

from ..utils.config import config


def normalize_case(text: str) -> str:
    """
    Normalize text case for Turkish addresses.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with proper Turkish casing
    """
    if not text:
        return text

    # Convert to lowercase first
    text = text.lower()

    # Handle Turkish specific characters - but preserve ı and i correctly
    turkish_replacements = {
        "i̇": "i",  # Remove combining dot above
        "İ": "i",  # Capital İ to lowercase i
        # Don't convert I to ı in all cases - that's too aggressive
        "Ğ": "ğ",
        "Ü": "ü",
        "Ş": "ş",
        "Ö": "ö",
        "Ç": "ç",
    }

    for tr_char, replacement in turkish_replacements.items():
        text = text.replace(tr_char, replacement)

    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters for consistent processing.

    Args:
        text: Input text to normalize

    Returns:
        Unicode normalized text
    """
    if not text:
        return text

    # Normalize to NFD (decomposed) then to NFC (composed)
    text = unicodedata.normalize("NFD", text)
    text = unicodedata.normalize("NFC", text)

    # Handle common Turkish character variations
    char_mappings = {
        "ı̇": "i",  # dotted ı
        "i̇": "i",  # i with combining dot
        "İ": "i",  # capital I with dot
        "I": "ı",  # capital I without dot
    }

    for char, replacement in char_mappings.items():
        text = text.replace(char, replacement)

    return text


def load_abbreviations(file_path: str = None) -> Dict[str, str]:
    """
    Load abbreviation mappings from YAML file.

    Args:
        file_path: Path to abbreviations file

    Returns:
        Dictionary mapping abbreviations to full forms
    """
    if file_path is None:
        file_path = config.get("resources.abbreviations_file")

    try:
        # Handle both absolute and relative paths
        if not Path(file_path).is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            file_path = project_root / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Ensure all values are strings (YAML might load some as other types)
        result = {}
        for key, value in data.items():
            if isinstance(key, str) and value is not None:
                result[key] = str(value)

        return result
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def expand_abbreviations(text: str, mapping: Dict[str, str] = None) -> str:
    """
    Expand abbreviations in text using provided mapping.

    Args:
        text: Input text containing abbreviations
        mapping: Dictionary mapping abbreviations to full forms

    Returns:
        Text with expanded abbreviations
    """
    if not text:
        return text

    if mapping is None:
        mapping = load_abbreviations()

    if not mapping:
        return text

    # Filter out non-string keys and sort abbreviations by length (longest first)
    valid_abbrevs = {k: v for k, v in mapping.items() if isinstance(k, str) and k}
    sorted_abbrevs = sorted(valid_abbrevs.keys(), key=len, reverse=True)

    result = text

    for abbrev in sorted_abbrevs:
        if not abbrev:
            continue

        full_form = valid_abbrevs[abbrev]

        # Create pattern for word boundaries
        # Handle special cases like "n:" which might be followed by numbers
        if abbrev.endswith(":"):
            # For colon-ending abbreviations, match the abbreviation followed by colon
            # and optionally followed by whitespace or word boundary
            pattern = r"\b" + re.escape(abbrev[:-1]) + r":(?=\s|\w|$)"
            replacement = full_form
        elif abbrev.endswith("."):
            # For dot-ending abbreviations
            pattern = re.escape(abbrev) + r"(?=\s|$)"
            replacement = full_form
        else:
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            replacement = full_form

        # Case-insensitive replacement
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def clean_punctuation(text: str) -> str:
    """
    Clean and standardize punctuation in text.

    Args:
        text: Input text to clean

    Returns:
        Text with cleaned punctuation
    """
    if not text:
        return text

    punctuation_mapping = config.get("preprocess.punctuation_mapping", {})

    result = text

    # Apply punctuation mappings
    for punct, replacement in punctuation_mapping.items():
        result = result.replace(punct, replacement)

    # Remove extra whitespace
    result = re.sub(r"\s+", " ", result)

    return result.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into meaningful components.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    if not text:
        return []

    separators = config.get("preprocess.token_separators", [" "])

    # Create regex pattern from separators
    separator_pattern = "|".join(re.escape(sep) for sep in separators)

    # Split on separators and filter empty tokens
    tokens = re.split(f"[{separator_pattern}]+", text)
    tokens = [token.strip() for token in tokens if token.strip()]

    return tokens


def clean_address(text: str) -> str:
    """
    Clean and standardize address text.

    Args:
        text: Raw address text

    Returns:
        Cleaned address text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Basic cleaning
    result = preprocess(text)

    return result["text"]


def preprocess(text: str, **kwargs) -> Dict[str, any]:
    """
    Complete preprocessing pipeline for Turkish address text.

    Args:
        text: Input text to preprocess
        **kwargs: Override configuration options

    Returns:
        Dictionary containing processed text and tokens
    """
    if not text:
        return {"text": "", "tokens": []}

    result = text

    # Apply preprocessing steps based on configuration
    if kwargs.get("normalize_case", config.get("preprocess.normalize_case", True)):
        result = normalize_case(result)

    if kwargs.get(
        "normalize_unicode", config.get("preprocess.normalize_unicode", True)
    ):
        result = normalize_unicode(result)

    # Expand abbreviations BEFORE cleaning punctuation to preserve patterns like "N:"
    if kwargs.get(
        "expand_abbreviations", config.get("preprocess.expand_abbreviations", True)
    ):
        abbrev_mapping = kwargs.get("abbreviation_mapping") or load_abbreviations()
        result = expand_abbreviations(result, abbrev_mapping)

    if kwargs.get(
        "clean_punctuation", config.get("preprocess.clean_punctuation", True)
    ):
        result = clean_punctuation(result)

    # Tokenize the processed text
    tokens = tokenize(result)

    # Ensure idempotency - running tokenize again should give same result
    tokens_check = tokenize(" ".join(tokens))
    assert tokens == tokens_check, "Tokenization is not idempotent"

    return {"text": result, "tokens": tokens}
