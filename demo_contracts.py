#!/usr/bin/env python3
"""
Demonstration script for the address normalization contracts.
"""

import json

from src.addrnorm.utils.contracts import (
    AddressOut,
    ExplanationParsed,
    MethodEnum,
    RelationEnum,
)


def demo_contracts():
    """Demonstrate the usage of data contracts."""
    print("=== Address Normalization Contract Demo ===\n")

    # Create sample explanation
    explanation = ExplanationParsed(
        confidence=0.95, method=MethodEnum.ML, warnings=["Posta kodu eksik"]
    )

    # Create sample address
    address = AddressOut(
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

    print("1. JSON Serialization:")
    json_output = address.to_json(indent=2)
    print(json_output)
    print()

    print("2. CSV Row Serialization:")
    csv_row = address.to_csv_row()
    headers = AddressOut.csv_headers()

    print("Headers:")
    print(", ".join(headers))
    print("\nData:")
    print(", ".join([f'"{field}"' if "," in field else field for field in csv_row]))
    print()

    print("3. CSV String (multiple addresses):")
    minimal_explanation = ExplanationParsed(
        confidence=0.6, method=MethodEnum.FALLBACK, warnings=[]
    )

    minimal_address = AddressOut(
        city="Ankara",
        explanation_raw="Ankara, basit adres",
        explanation_parsed=minimal_explanation,
        normalized_address="Ankara, basit adres",
    )

    csv_string = AddressOut.to_csv_string([address, minimal_address])
    print(csv_string)

    print("4. Schema Validation:")
    from pathlib import Path

    import jsonschema

    schema_path = Path(__file__).parent / "schemas" / "output.address.json"
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        address_dict = address.model_dump()
        jsonschema.validate(address_dict, schema)
        print("✅ Address validates against JSON schema")
    except Exception as e:
        print(f"❌ Schema validation error: {e}")


if __name__ == "__main__":
    demo_contracts()
