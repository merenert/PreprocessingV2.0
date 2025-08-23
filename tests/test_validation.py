"""
Validation module tests for coverage increase.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from addrnorm.validate.geo import TurkishGeoValidator, create_geo_validator


def test_create_geo_validator():
    """Test geo validator creation."""
    validator = create_geo_validator()
    assert validator is not None
    assert isinstance(validator, TurkishGeoValidator)


def test_geo_validator_basic():
    """Test basic geo validation functionality."""
    validator = create_geo_validator()

    # Test city validation
    result = validator.validate_city("İstanbul")
    assert result.is_valid

    result = validator.validate_city("Ankara")
    assert result.is_valid

    result = validator.validate_city("İzmir")
    assert result.is_valid

    # Test invalid city
    result = validator.validate_city("InvalidCity")
    assert not result.is_valid


def test_geo_validator_districts():
    """Test district validation."""
    validator = create_geo_validator()

    # Test district validation
    result = validator.validate_district("Beşiktaş")
    assert result.is_valid

    result = validator.validate_district("Çankaya")
    assert result.is_valid

    result = validator.validate_district("Konak")
    assert result.is_valid

    # Test invalid district
    result = validator.validate_district("InvalidDistrict")
    assert not result.is_valid


def test_geo_validator_city_district_pair():
    """Test city-district pair validation."""
    validator = create_geo_validator()

    # Test consistency validation
    result = validator.validate_city_district_consistency("İstanbul", "Beşiktaş")
    assert result.is_consistent

    result = validator.validate_city_district_consistency("Ankara", "Çankaya")
    assert result.is_consistent


def test_geo_validator_stats():
    """Test geo validator statistics."""
    validator = create_geo_validator()

    if hasattr(validator, "get_stats"):
        stats = validator.get_stats()
        assert isinstance(stats, dict)
        assert "cities_count" in stats


def test_geo_validator_normalization():
    """Test geo name normalization."""
    validator = create_geo_validator()

    # Test case insensitive matching
    result = validator.validate_city("istanbul")
    assert result.is_valid

    result = validator.validate_city("ANKARA")
    assert result.is_valid

    result = validator.validate_district("besiktas")
    assert result.is_valid
