"""
Fuzz testing for Turkish address normalization using Hypothesis.
Generates random variations and tests system stability and schema compliance.
"""

import json
import string
import time
from typing import Any, Dict

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, initialize, rule

from addrnorm.pipeline import PipelineConfig, create_pipeline
from addrnorm.utils.contracts import AddressOut

# Address component generators
turkish_chars = "çğıöşüÇĞIÖŞÜ"
ascii_chars = string.ascii_letters
numbers = string.digits
common_separators = " ,-./:"

# Turkish cities (subset for testing)
TURKISH_CITIES = [
    "İstanbul",
    "Ankara",
    "İzmir",
    "Bursa",
    "Antalya",
    "Adana",
    "Konya",
    "Gaziantep",
    "Mersin",
    "Diyarbakır",
    "Kayseri",
    "Eskişehir",
    "Urfa",
    "Malatya",
    "Erzurum",
    "Van",
    "Batman",
    "Elazığ",
    "Iğdır",
    "Muş",
]

# Common Turkish districts
TURKISH_DISTRICTS = [
    "Merkez",
    "Çankaya",
    "Beşiktaş",
    "Kadıköy",
    "Şişli",
    "Fatih",
    "Beyoğlu",
    "Üsküdar",
    "Maltepe",
    "Pendik",
    "Kartal",
    "Ataşehir",
    "Bakırköy",
]

# Common Turkish neighborhoods
TURKISH_NEIGHBORHOODS = [
    "Merkez",
    "Yeni",
    "Eski",
    "Çarşı",
    "Pazar",
    "Kale",
    "Bahçe",
    "Orman",
    "Levent",
    "Etiler",
    "Maslak",
    "Mecidiyeköy",
    "Gayrettepe",
    "Nişantaşı",
]

# Street types
STREET_TYPES = ["Caddesi", "Sokağı", "Bulvarı", "Meydanı", "Yolu", "Geçidi", "Çıkmazı"]

# Building types
BUILDING_TYPES = ["Plaza", "AVM", "Hastanesi", "Okulu", "Camii", "Parkı", "İstasyonu"]


class AddressComponentStrategy:
    """Strategy for generating address components."""

    @staticmethod
    def city():
        """Generate city names."""
        return st.one_of(
            st.sampled_from(TURKISH_CITIES),
            st.text(min_size=3, max_size=15, alphabet=ascii_chars + turkish_chars),
        )

    @staticmethod
    def district():
        """Generate district names."""
        return st.one_of(
            st.sampled_from(TURKISH_DISTRICTS),
            st.text(min_size=3, max_size=20, alphabet=ascii_chars + turkish_chars),
        )

    @staticmethod
    def neighborhood():
        """Generate neighborhood names."""
        return st.one_of(
            st.sampled_from(TURKISH_NEIGHBORHOODS),
            st.text(min_size=3, max_size=20, alphabet=ascii_chars + turkish_chars),
        )

    @staticmethod
    def street():
        """Generate street names."""
        base_name = st.text(
            min_size=3, max_size=20, alphabet=ascii_chars + turkish_chars
        )
        street_type = st.sampled_from(STREET_TYPES)
        return st.builds(lambda name, stype: f"{name} {stype}", base_name, street_type)

    @staticmethod
    def building():
        """Generate building names."""
        base_name = st.text(
            min_size=3, max_size=20, alphabet=ascii_chars + turkish_chars
        )
        building_type = st.sampled_from(BUILDING_TYPES)
        return st.builds(
            lambda name, btype: f"{name} {btype}", base_name, building_type
        )

    @staticmethod
    def number():
        """Generate building numbers."""
        return st.one_of(
            st.integers(min_value=1, max_value=9999).map(str),
            st.builds(
                lambda a, b: f"{a}/{b}",
                st.integers(min_value=1, max_value=999),
                st.integers(min_value=1, max_value=99),
            ),
        )

    @staticmethod
    def floor():
        """Generate floor numbers."""
        return st.one_of(
            st.integers(min_value=-2, max_value=50).map(str),
            st.sampled_from(["Zemin", "Bodrum", "Çatı"]),
        )

    @staticmethod
    def apartment():
        """Generate apartment numbers."""
        return st.one_of(
            st.integers(min_value=1, max_value=999).map(str),
            st.builds(
                lambda a, b: f"{a}{b}",
                st.integers(min_value=1, max_value=99),
                st.sampled_from(["A", "B", "C", "D"]),
            ),
        )


def generate_random_address():
    """Strategy for generating complete random addresses."""

    # Core components
    city = AddressComponentStrategy.city()
    district = AddressComponentStrategy.district()
    neighborhood = st.one_of(st.none(), AddressComponentStrategy.neighborhood())
    street = st.one_of(st.none(), AddressComponentStrategy.street())
    building = st.one_of(st.none(), AddressComponentStrategy.building())
    number = st.one_of(st.none(), AddressComponentStrategy.number())
    floor = st.one_of(st.none(), AddressComponentStrategy.floor())
    apartment = st.one_of(st.none(), AddressComponentStrategy.apartment())

    # Separators and case variations
    separator = st.sampled_from([" ", ", ", " - ", "/"])
    case_transform = st.sampled_from(["upper", "lower", "title", "mixed", "none"])

    def build_address(
        city_val,
        district_val,
        neighborhood_val,
        street_val,
        building_val,
        number_val,
        floor_val,
        apartment_val,
        sep,
        case_type,
    ):
        """Build address string from components."""

        components = []

        # Add components in various orders
        if city_val:
            components.append(city_val)
        if district_val:
            components.append(district_val)
        if neighborhood_val:
            components.append(neighborhood_val)
        if street_val:
            components.append(street_val)
        if building_val:
            components.append(building_val)
        if number_val:
            components.append(f"No:{number_val}")
        if floor_val:
            components.append(f"Kat:{floor_val}")
        if apartment_val:
            components.append(f"Daire:{apartment_val}")

        if not components:
            components = ["Test"]

        address = sep.join(components)

        # Apply case transformations
        if case_type == "upper":
            address = address.upper()
        elif case_type == "lower":
            address = address.lower()
        elif case_type == "title":
            address = address.title()
        elif case_type == "mixed":
            # Random case for each character
            import random

            address = "".join(
                c.upper() if random.random() > 0.5 else c.lower() for c in address
            )

        return address

    return st.builds(
        build_address,
        city,
        district,
        neighborhood,
        street,
        building,
        number,
        floor,
        apartment,
        separator,
        case_transform,
    )


class FuzzTestRunner:
    """Runner for fuzz tests with stability and schema compliance checks."""

    def __init__(self):
        """Initialize the fuzz test runner."""
        config = PipelineConfig(
            enable_validation=True,
            log_level="ERROR",  # Minimize noise during fuzz testing
        )
        self.pipeline = create_pipeline(config)
        self.results = []

    def test_address_stability(self, address: str) -> Dict[str, Any]:
        """Test if address processing is stable (doesn't crash)."""

        start_time = time.time()

        try:
            result = self.pipeline.process_single(address)
            processing_time = (time.time() - start_time) * 1000

            return {
                "input": address,
                "status": "stable",
                "success": result.success,
                "error": result.error,
                "processing_time_ms": processing_time,
                "method": (
                    result.processing_method
                    if hasattr(result, "processing_method")
                    else None
                ),
                "confidence": (
                    result.confidence if hasattr(result, "confidence") else None
                ),
            }

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                "input": address,
                "status": "crashed",
                "error": str(e),
                "processing_time_ms": processing_time,
            }

    def test_schema_compliance(self, address: str) -> Dict[str, Any]:
        """Test if output complies with AddressOut schema."""

        try:
            result = self.pipeline.process_single(address)

            if not result.success or not result.address_out:
                return {
                    "input": address,
                    "schema_valid": True,  # No output to validate
                    "validation_error": None,
                }

            # Try to serialize/deserialize to validate schema
            try:
                json_str = result.address_out.to_json()
                data = json.loads(json_str)

                # Try to recreate AddressOut from dict
                AddressOut(**data)

                return {
                    "input": address,
                    "schema_valid": True,
                    "validation_error": None,
                }

            except Exception as schema_error:
                return {
                    "input": address,
                    "schema_valid": False,
                    "validation_error": str(schema_error),
                }

        except Exception as e:
            return {
                "input": address,
                "schema_valid": False,
                "validation_error": f"Processing error: {str(e)}",
            }


# Hypothesis-based fuzz tests
@given(generate_random_address())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_stability(address):
    """Test system stability with random addresses."""
    runner = FuzzTestRunner()
    result = runner.test_address_stability(address)

    # System should never crash
    assert result["status"] != "crashed", f"System crashed on input: {address}"

    # Processing time should be reasonable (< 10 seconds)
    assert (
        result["processing_time_ms"] < 10000
    ), f"Processing too slow: {result['processing_time_ms']}ms"


@given(generate_random_address())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_schema_compliance(address):
    """Test schema compliance with random addresses."""
    runner = FuzzTestRunner()
    result = runner.test_schema_compliance(address)

    # Output should always comply with schema if successful
    if not result["schema_valid"]:
        print(f"Schema violation for input: {address}")
        print(f"Error: {result['validation_error']}")

    assert result["schema_valid"], f"Schema violation: {result['validation_error']}"


# Edge case testing
@given(st.text(min_size=0, max_size=1000, alphabet=string.printable))
@settings(max_examples=50)
def test_fuzz_arbitrary_text(text):
    """Test with completely arbitrary text input."""
    runner = FuzzTestRunner()
    result = runner.test_address_stability(text)

    # Should never crash, even with garbage input
    assert result["status"] != "crashed", f"System crashed on arbitrary text: {text}"


# Special character testing
@given(st.text(min_size=1, max_size=100, alphabet=turkish_chars + "çğıöşüÇĞIÖŞÜ"))
@settings(max_examples=30)
def test_fuzz_turkish_characters(text):
    """Test with Turkish character variations."""
    runner = FuzzTestRunner()
    result = runner.test_address_stability(text)

    assert result["status"] != "crashed", f"System crashed on Turkish text: {text}"


# Numeric edge cases
@given(
    st.builds(
        lambda nums: " ".join(map(str, nums)),
        st.lists(
            st.integers(min_value=-999999, max_value=999999), min_size=1, max_size=10
        ),
    )
)
@settings(max_examples=30)
def test_fuzz_numbers(number_text):
    """Test with numeric inputs."""
    runner = FuzzTestRunner()
    result = runner.test_address_stability(number_text)

    assert result["status"] != "crashed", f"System crashed on numbers: {number_text}"


class AddressFuzzStateMachine(RuleBasedStateMachine):
    """Stateful testing of address normalization pipeline."""

    addresses = Bundle("addresses")

    def __init__(self):
        super().__init__()
        config = PipelineConfig(log_level="ERROR")
        self.pipeline = create_pipeline(config)

    @initialize()
    def setup(self):
        """Setup initial state."""
        pass

    @rule(target=addresses, address=generate_random_address())
    def create_address(self, address):
        """Create a new address to test."""
        return address

    @rule(address=addresses)
    def process_address(self, address):
        """Process an address and check stability."""
        try:
            result = self.pipeline.process_single(address)
            # Should complete without exception
            assert result is not None

            if result.success and result.address_out:
                # If successful, output should be valid
                json_str = result.address_out.to_json()
                data = json.loads(json_str)
                AddressOut(**data)  # Should not raise exception

        except Exception as e:
            # Should not crash
            assert False, f"Processing crashed for address '{address}': {e}"

    @rule(address=addresses)
    def process_address_twice(self, address):
        """Test deterministic behavior - same input should give same output."""
        try:
            result1 = self.pipeline.process_single(address)
            result2 = self.pipeline.process_single(address)

            # Results should be consistent
            assert result1.success == result2.success

            if result1.success and result2.success:
                if result1.address_out and result2.address_out:
                    # Normalized addresses should be identical
                    assert (
                        result1.address_out.normalized_address
                        == result2.address_out.normalized_address
                    )

        except Exception as e:
            assert False, f"Determinism test failed for '{address}': {e}"


# Run stateful tests
TestAddressFuzzStateMachine = AddressFuzzStateMachine.TestCase
TestAddressFuzzStateMachine.settings = settings(
    max_examples=50, stateful_step_count=10, deadline=None
)


def run_comprehensive_fuzz_tests():
    """Run comprehensive fuzz testing suite."""

    print("🔀 Starting comprehensive fuzz testing...")

    # Run basic stability tests
    print("  Testing stability with random addresses...")
    runner = FuzzTestRunner()

    stable_count = 0
    crashed_count = 0
    schema_violations = 0

    # Generate test cases
    for i in range(100):
        # Generate random address
        address = generate_random_address().example()

        # Test stability
        stability_result = runner.test_address_stability(address)
        if stability_result["status"] == "stable":
            stable_count += 1
        else:
            crashed_count += 1
            print(f"    CRASH: {address} -> {stability_result['error']}")

        # Test schema compliance
        if stability_result["status"] == "stable":
            schema_result = runner.test_schema_compliance(address)
            if not schema_result["schema_valid"]:
                schema_violations += 1
                print(f"    SCHEMA: {address} -> {schema_result['validation_error']}")

    print(f"  Stability: {stable_count}/100 ({stable_count}%)")
    print(f"  Crashes: {crashed_count}")
    print(f"  Schema violations: {schema_violations}")

    results = {
        "total_tests": 100,
        "stable": stable_count,
        "crashed": crashed_count,
        "schema_violations": schema_violations,
        "stability_rate": stable_count / 100,
        "schema_compliance_rate": (
            (stable_count - schema_violations) / stable_count if stable_count > 0 else 0
        ),
    }

    # Save results
    with open("fuzz_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("💾 Fuzz test results saved to: fuzz_test_results.json")

    return results


if __name__ == "__main__":
    # Run standalone fuzz tests
    results = run_comprehensive_fuzz_tests()
    print("\n📊 Fuzz Test Summary:")
    print(f"Stability Rate: {results['stability_rate']:.1%}")
    print(f"Schema Compliance: {results['schema_compliance_rate']:.1%}")
