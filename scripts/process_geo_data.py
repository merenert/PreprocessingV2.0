#!/usr/bin/env python3
"""
Process Turkish cities and districts data to lowercase ASCII format.
"""

import csv
import json
import os
import unicodedata


def to_lowercase_ascii(text):
    """Convert Turkish text to lowercase ASCII."""
    if not text:
        return text

    # Normalize unicode
    text = unicodedata.normalize("NFD", text)
    # Remove accents
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    # Convert to lowercase
    text = text.lower()

    # Replace Turkish specific characters
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }

    for tr_char, ascii_char in replacements.items():
        text = text.replace(tr_char, ascii_char)

    return text


def process_cities():
    """Process cities from iller.csv to cities_tr.csv."""

    cities = []

    with open("iller.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city_name = row["name"].strip()
            ascii_name = to_lowercase_ascii(city_name)
            cities.append(ascii_name)

    # Sort cities
    cities.sort()

    # Write to resources
    os.makedirs("data/resources", exist_ok=True)

    with open("data/resources/cities_tr.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for city in cities:
            writer.writerow([city])

    print(f"✅ Processed {len(cities)} cities")
    print(f"Sample cities: {cities[:5]}")

    return cities


def process_districts():
    """Process districts from ilceler.csv to districts_tr.json."""

    districts_by_city = {}

    with open("ilceler.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = to_lowercase_ascii(row["il"].strip())
            district = to_lowercase_ascii(row["ilçe"].strip())

            if city not in districts_by_city:
                districts_by_city[city] = []

            districts_by_city[city].append(district)

    # Sort districts for each city
    for city in districts_by_city:
        districts_by_city[city].sort()

    # Write to resources
    with open("data/resources/districts_tr.json", "w", encoding="utf-8") as f:
        json.dump(districts_by_city, f, ensure_ascii=False, indent=2)

    total_districts = sum(len(districts) for districts in districts_by_city.values())
    print(
        f"✅ Processed {len(districts_by_city)} cities with {total_districts} districts"
    )
    print(f"Sample: {list(districts_by_city.keys())[:5]}")

    return districts_by_city


def validate_data(cities, districts_by_city):
    """Validate the processed data."""

    print("\n📊 Data Validation:")
    print(f"Cities count: {len(cities)}")
    print("Expected: 81 (Turkish provinces)")

    if len(cities) == 81:
        print("✅ City count is correct!")
    else:
        print(f"⚠️ City count mismatch: expected 81, got {len(cities)}")

    # Check some major cities
    major_cities = ["istanbul", "ankara", "izmir", "bursa", "antalya"]
    missing_major = [city for city in major_cities if city not in cities]

    if not missing_major:
        print("✅ All major cities present")
    else:
        print(f"⚠️ Missing major cities: {missing_major}")

    # Check districts
    cities_with_districts = len(districts_by_city)
    print(f"Cities with districts: {cities_with_districts}")

    # Sample district counts
    for city in ["istanbul", "ankara", "izmir"]:
        if city in districts_by_city:
            count = len(districts_by_city[city])
            print(f"  {city}: {count} districts")


if __name__ == "__main__":
    print("🔄 Processing Turkish geographic data...")

    cities = process_cities()
    districts_by_city = process_districts()
    validate_data(cities, districts_by_city)

    print("\n✅ Data processing complete!")
    print("📁 Files created:")
    print("  - data/resources/cities_tr.csv")
    print("  - data/resources/districts_tr.json")
