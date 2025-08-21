"""
Tests for data contracts and serialization.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from src.addrnorm.utils.contracts import (
    AddressOut,
    ExplanationParsed,
    MethodEnum,
    RelationEnum,
)


@pytest.fixture
def json_schema():
    """Load the JSON schema."""
    schema_path = Path(__file__).parent.parent / "schemas" / "output.address.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_address():
    """Create a sample address for testing."""
    explanation = ExplanationParsed(
        confidence=0.95, method=MethodEnum.ML, warnings=["Test warning"]
    )

    return AddressOut(
        country="TR",
        city="İstanbul",
        district="Kadıköy",
        neighborhood="Moda Mah.",
        street="Bahariye Cad.",
        building="Test Plaza",
        block="A",
        number="12",
        entrance="2",
        floor="5",
        apartment="15",
        postcode="34710",
        relation=RelationEnum.KARSISI,
        explanation_raw="İstanbul, Kadıköy, Moda Mah., Bahariye Cad. No:12 D:5",
        explanation_parsed=explanation,
        normalized_address=(
            "TR, İstanbul, Kadıköy, Moda Mah., Bahariye Cad., Test Plaza, "
            "A Blok, No:12, Giriş:2, Kat:5, Daire:15, 34710"
        ),
    )


def test_json_schema_validation(json_schema, sample_address):
    """Test that Pydantic model matches JSON schema."""
    # Convert pydantic model to dict
    address_dict = sample_address.model_dump()

    # Validate against JSON schema
    jsonschema.validate(address_dict, json_schema)


def test_pydantic_model_validation():
    """Test Pydantic model validation."""
    explanation = ExplanationParsed(
        confidence=0.87, method=MethodEnum.PATTERN, warnings=[]
    )

    address = AddressOut(
        explanation_raw="Test raw address",
        explanation_parsed=explanation,
        normalized_address="Test normalized address",
    )

    assert address.explanation_parsed.confidence == 0.87
    assert address.explanation_parsed.method == MethodEnum.PATTERN
    assert address.explanation_parsed.warnings == []


def test_pydantic_validation_errors():
    """Test Pydantic validation catches errors."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExplanationParsed(
            confidence=1.5, method=MethodEnum.ML, warnings=[]  # Invalid: > 1.0
        )


def test_country_code_validation():
    """Test country code validation."""
    from pydantic import ValidationError

    explanation = ExplanationParsed(confidence=0.95, method=MethodEnum.ML, warnings=[])

    # Valid country code
    address = AddressOut(
        country="TR",
        explanation_raw="Test",
        explanation_parsed=explanation,
        normalized_address="Test",
    )
    assert address.country == "TR"

    # Invalid country code should raise validation error
    with pytest.raises(ValidationError):
        AddressOut(
            country="INVALID",
            explanation_raw="Test",
            explanation_parsed=explanation,
            normalized_address="Test",
        )


def test_postcode_validation():
    """Test postcode validation."""
    from pydantic import ValidationError

    explanation = ExplanationParsed(confidence=0.95, method=MethodEnum.ML, warnings=[])

    # Valid postcode
    address = AddressOut(
        postcode="34710",
        explanation_raw="Test",
        explanation_parsed=explanation,
        normalized_address="Test",
    )
    assert address.postcode == "34710"

    # Invalid postcode should raise validation error
    with pytest.raises(ValidationError):
        AddressOut(
            postcode="123",  # Too short
            explanation_raw="Test",
            explanation_parsed=explanation,
            normalized_address="Test",
        )


def test_to_json_serialization(sample_address):
    """Test JSON serialization."""
    json_str = sample_address.to_json()
    parsed = json.loads(json_str)

    assert parsed["country"] == "TR"
    assert parsed["city"] == "İstanbul"
    assert parsed["explanation_parsed"]["confidence"] == 0.95
    assert parsed["explanation_parsed"]["method"] == "ml"


def test_to_csv_row_serialization(sample_address):
    """Test CSV row serialization."""
    csv_row = sample_address.to_csv_row()

    expected_length = 18  # Number of fields
    assert len(csv_row) == expected_length

    assert csv_row[0] == "TR"  # country
    assert csv_row[1] == "İstanbul"  # city
    assert csv_row[12] == "karsisi"  # relation
    assert (
        csv_row[13] == "İstanbul, Kadıköy, Moda Mah., Bahariye Cad. No:12 D:5"
    )  # explanation_raw
    assert csv_row[15] == "0.95"  # confidence
    assert csv_row[16] == "ml"  # method
    assert json.loads(csv_row[17]) == ["Test warning"]  # warnings


def test_csv_headers():
    """Test CSV headers."""
    headers = AddressOut.csv_headers()
    expected_headers = [
        "country",
        "city",
        "district",
        "neighborhood",
        "street",
        "building",
        "block",
        "number",
        "entrance",
        "floor",
        "apartment",
        "postcode",
        "relation",
        "explanation_raw",
        "normalized_address",
        "confidence",
        "method",
        "warnings",
    ]

    assert headers == expected_headers


def test_to_csv_string(sample_address):
    """Test CSV string generation."""
    addresses = [sample_address]
    csv_string = AddressOut.to_csv_string(addresses)

    lines = csv_string.strip().split("\n")
    assert len(lines) == 2  # Header + 1 data row

    # Check header
    header_fields = lines[0].split(",")
    assert header_fields[0] == "country"
    assert header_fields[1] == "city"

    # Check data row starts correctly
    assert lines[1].startswith("TR,İstanbul")


def test_minimal_address():
    """Test address with only required fields."""
    explanation = ExplanationParsed(
        confidence=0.5, method=MethodEnum.FALLBACK, warnings=["Minimal data"]
    )

    address = AddressOut(
        explanation_raw="Minimal test address",
        explanation_parsed=explanation,
        normalized_address="Minimal normalized",
    )

    # Should serialize without errors
    json_str = address.to_json()
    csv_row = address.to_csv_row()

    assert json_str is not None
    assert len(csv_row) == 18

    # Most fields should be empty strings
    assert csv_row[0] == ""  # country
    assert csv_row[1] == ""  # city
    assert csv_row[13] == "Minimal test address"  # explanation_raw
    assert csv_row[14] == "Minimal normalized"  # normalized_address


def test_json_schema_completeness(json_schema):
    """Test that JSON schema covers all expected fields."""
    properties = json_schema["properties"]

    # Check all expected fields are present
    expected_fields = {
        "country",
        "city",
        "district",
        "neighborhood",
        "street",
        "building",
        "block",
        "number",
        "entrance",
        "floor",
        "apartment",
        "postcode",
        "relation",
        "explanation_raw",
        "explanation_parsed",
        "normalized_address",
    }

    schema_fields = set(properties.keys())
    assert schema_fields == expected_fields

    # Check required fields
    required_fields = set(json_schema["required"])
    expected_required = {"explanation_raw", "explanation_parsed", "normalized_address"}
    assert required_fields == expected_required
