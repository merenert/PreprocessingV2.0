"""
Geographic validation module for Turkish addresses.
Provides city/district standardization, fuzzy matching, and consistency checks.
"""

import csv
import difflib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ValidationResult:
    """Result of geographic validation."""

    is_valid: bool
    standardized_value: Optional[str]
    confidence: float
    suggestions: List[str]
    warnings: List[str]


@dataclass
class ConsistencyResult:
    """Result of city-district consistency check."""

    is_consistent: bool
    standardized_city: Optional[str]
    standardized_district: Optional[str]
    suggestions: List[Tuple[str, str]]  # (city, district) pairs
    warnings: List[str]


class TurkishGeoValidator:
    """
    Geographic validator for Turkish cities and districts.
    Provides fuzzy matching and consistency validation.
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize validator with geographic data.

        Args:
            data_dir: Directory containing cities_tr.csv and districts_tr.json
        """
        if data_dir is None:
            # Default to data/resources relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "..", "..", "data", "resources")

        self.data_dir = data_dir
        self.cities: Set[str] = set()
        self.districts_by_city: Dict[str, List[str]] = {}
        self.all_districts: Set[str] = set()

        self._load_data()

    def _load_data(self):
        """Load cities and districts data from files."""

        # Load cities
        cities_file = os.path.join(self.data_dir, "cities_tr.csv")
        if os.path.exists(cities_file):
            with open(cities_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    city = row["name"].strip().lower()
                    self.cities.add(city)

        # Load districts
        districts_file = os.path.join(self.data_dir, "districts_tr.json")
        if os.path.exists(districts_file):
            with open(districts_file, "r", encoding="utf-8") as f:
                self.districts_by_city = json.load(f)

            # Build set of all districts
            for districts in self.districts_by_city.values():
                self.all_districts.update(districts)

        print(
            f"📊 Loaded {len(self.cities)} cities and "
            f"{len(self.all_districts)} districts"
        )

    def validate_city(
        self, city: str, fuzzy_threshold: float = 0.6
    ) -> ValidationResult:
        """
        Validate and standardize a city name.

        Args:
            city: City name to validate
            fuzzy_threshold: Minimum similarity for fuzzy matching (0.0-1.0)

        Returns:
            ValidationResult with validation outcome
        """

        if not city or not city.strip():
            return ValidationResult(
                is_valid=False,
                standardized_value=None,
                confidence=0.0,
                suggestions=[],
                warnings=["Empty city name"],
            )

        # Normalize input
        normalized_city = self._normalize_text(city)

        # Exact match
        if normalized_city in self.cities:
            return ValidationResult(
                is_valid=True,
                standardized_value=normalized_city,
                confidence=1.0,
                suggestions=[],
                warnings=[],
            )

        # Fuzzy matching
        matches = difflib.get_close_matches(
            normalized_city, self.cities, n=5, cutoff=fuzzy_threshold
        )

        if matches:
            best_match = matches[0]
            similarity = difflib.SequenceMatcher(
                None, normalized_city, best_match
            ).ratio()

            if similarity >= 0.8:  # High confidence match
                return ValidationResult(
                    is_valid=True,
                    standardized_value=best_match,
                    confidence=similarity,
                    suggestions=matches[1:],  # Other suggestions
                    warnings=[f"Corrected '{city}' to '{best_match}'"],
                )
            else:  # Lower confidence - suggest but don't auto-correct
                return ValidationResult(
                    is_valid=False,
                    standardized_value=None,
                    confidence=similarity,
                    suggestions=matches,
                    warnings=[
                        f"'{city}' not found, did you mean: {', '.join(matches[:3])}?"
                    ],
                )

        # No matches found
        return ValidationResult(
            is_valid=False,
            standardized_value=None,
            confidence=0.0,
            suggestions=[],
            warnings=[f"City '{city}' not found in Turkish provinces"],
        )

    def validate_district(
        self, district: str, city: str = None, fuzzy_threshold: float = 0.6
    ) -> ValidationResult:
        """
        Validate and standardize a district name.

        Args:
            district: District name to validate
            city: Optional city context for validation
            fuzzy_threshold: Minimum similarity for fuzzy matching

        Returns:
            ValidationResult with validation outcome
        """

        if not district or not district.strip():
            return ValidationResult(
                is_valid=False,
                standardized_value=None,
                confidence=0.0,
                suggestions=[],
                warnings=["Empty district name"],
            )

        # Normalize input
        normalized_district = self._normalize_text(district)
        normalized_city = self._normalize_text(city) if city else None

        # If city is provided, check within that city first
        if normalized_city and normalized_city in self.districts_by_city:
            city_districts = self.districts_by_city[normalized_city]

            # Exact match within city
            if normalized_district in city_districts:
                return ValidationResult(
                    is_valid=True,
                    standardized_value=normalized_district,
                    confidence=1.0,
                    suggestions=[],
                    warnings=[],
                )

            # Fuzzy match within city
            matches = difflib.get_close_matches(
                normalized_district, city_districts, n=3, cutoff=fuzzy_threshold
            )

            if matches:
                best_match = matches[0]
                similarity = difflib.SequenceMatcher(
                    None, normalized_district, best_match
                ).ratio()

                return ValidationResult(
                    is_valid=similarity >= 0.8,
                    standardized_value=best_match if similarity >= 0.8 else None,
                    confidence=similarity,
                    suggestions=matches,
                    warnings=(
                        [f"District '{district}' corrected to '{best_match}' in {city}"]
                        if similarity >= 0.8
                        else [
                            f"District '{district}' not found in {city}, "
                            f"suggestions: {', '.join(matches)}"
                        ]
                    ),
                )

        # Check across all districts
        if normalized_district in self.all_districts:
            return ValidationResult(
                is_valid=True,
                standardized_value=normalized_district,
                confidence=1.0,
                suggestions=[],
                warnings=(
                    ["District found but city context unclear"] if not city else []
                ),
            )

        # Fuzzy match across all districts
        matches = difflib.get_close_matches(
            normalized_district, self.all_districts, n=5, cutoff=fuzzy_threshold
        )

        if matches:
            best_match = matches[0]
            similarity = difflib.SequenceMatcher(
                None, normalized_district, best_match
            ).ratio()

            return ValidationResult(
                is_valid=similarity >= 0.8,
                standardized_value=best_match if similarity >= 0.8 else None,
                confidence=similarity,
                suggestions=matches,
                warnings=(
                    [f"District '{district}' corrected to '{best_match}'"]
                    if similarity >= 0.8
                    else [
                        f"District '{district}' not found, "
                        f"suggestions: {', '.join(matches[:3])}"
                    ]
                ),
            )

        return ValidationResult(
            is_valid=False,
            standardized_value=None,
            confidence=0.0,
            suggestions=[],
            warnings=[f"District '{district}' not found in Turkish districts"],
        )

    def validate_city_district_consistency(
        self, city: str, district: str
    ) -> ConsistencyResult:
        """
        Validate that a city-district combination is consistent.

        Args:
            city: City name
            district: District name

        Returns:
            ConsistencyResult with consistency check outcome
        """

        # First validate city and district individually
        city_result = self.validate_city(city)
        district_result = self.validate_district(district)

        warnings = []
        warnings.extend(city_result.warnings)
        warnings.extend(district_result.warnings)

        # Use standardized values if available
        std_city = city_result.standardized_value or self._normalize_text(city)
        std_district = district_result.standardized_value or self._normalize_text(
            district
        )

        # Check consistency
        if std_city and std_city in self.districts_by_city:
            city_districts = self.districts_by_city[std_city]

            if std_district in city_districts:
                return ConsistencyResult(
                    is_consistent=True,
                    standardized_city=std_city,
                    standardized_district=std_district,
                    suggestions=[],
                    warnings=warnings,
                )
            else:
                # Find alternative cities for this district
                suggestions = []
                for c, districts in self.districts_by_city.items():
                    if std_district in districts:
                        suggestions.append((c, std_district))

                warnings.append(f"District '{district}' is not in city '{city}'")

                return ConsistencyResult(
                    is_consistent=False,
                    standardized_city=std_city,
                    standardized_district=std_district,
                    suggestions=suggestions[:5],  # Top 5 suggestions
                    warnings=warnings,
                )

        return ConsistencyResult(
            is_consistent=False,
            standardized_city=std_city,
            standardized_district=std_district,
            suggestions=[],
            warnings=warnings + ["Unable to verify city-district consistency"],
        )

    def validate_postcode(self, postcode: str) -> ValidationResult:
        """
        Validate Turkish postal code format and range.

        Args:
            postcode: Postal code to validate

        Returns:
            ValidationResult with validation outcome
        """

        if not postcode or not postcode.strip():
            return ValidationResult(
                is_valid=False,
                standardized_value=None,
                confidence=0.0,
                suggestions=[],
                warnings=["Empty postal code"],
            )

        # Clean postcode
        clean_postcode = re.sub(r"\D", "", postcode.strip())

        # Check length
        if len(clean_postcode) != 5:
            return ValidationResult(
                is_valid=False,
                standardized_value=None,
                confidence=0.0,
                suggestions=[],
                warnings=[f"Postal code must be 5 digits, got {len(clean_postcode)}"],
            )

        # Check range (Turkish postal codes: 01000-81999)
        try:
            code_num = int(clean_postcode)
            if 1000 <= code_num <= 81999:
                return ValidationResult(
                    is_valid=True,
                    standardized_value=clean_postcode,
                    confidence=1.0,
                    suggestions=[],
                    warnings=[],
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    standardized_value=None,
                    confidence=0.0,
                    suggestions=[],
                    warnings=[
                        f"Postal code {clean_postcode} out of valid range (01000-81999)"
                    ],
                )
        except ValueError:
            return ValidationResult(
                is_valid=False,
                standardized_value=None,
                confidence=0.0,
                suggestions=[],
                warnings=[f"Invalid postal code format: {postcode}"],
            )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        import unicodedata

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

        return text.strip()

    def get_stats(self) -> Dict[str, any]:
        """Get validator statistics."""

        return {
            "cities_count": len(self.cities),
            "districts_count": len(self.all_districts),
            "cities_with_districts": len(self.districts_by_city),
            "avg_districts_per_city": (
                len(self.all_districts) / len(self.districts_by_city)
                if self.districts_by_city
                else 0
            ),
            "data_source": "Official Turkish geographic data",
            "supported_validations": [
                "city_validation",
                "district_validation",
                "city_district_consistency",
                "postal_code_validation",
                "fuzzy_matching",
            ],
        }


def create_geo_validator(data_dir: str = None) -> TurkishGeoValidator:
    """Factory function to create a TurkishGeoValidator instance."""
    return TurkishGeoValidator(data_dir)
