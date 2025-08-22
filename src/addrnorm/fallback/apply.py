"""
Fallback normalization application module.
Converts raw address dictionaries to AddressOut objects using rule-based approach.
"""

from typing import Any, Dict, List

from ..utils.contracts import AddressOut, ExplanationParsed, MethodEnum
from .rules import TurkishAddressRules


class FallbackNormalizer:
    """
    Applies rule-based fallback normalization to convert address dictionaries
    to AddressOut objects when pattern and ML methods fail.
    """

    def __init__(self):
        self.rules = TurkishAddressRules()

    def normalize(self, address_dict: Dict[str, Any], raw_input: str) -> AddressOut:
        """
        Convert address dictionary to AddressOut using fallback rules.

        Args:
            address_dict: Dictionary with address components (may be incomplete)
            raw_input: Original raw address string

        Returns:
            AddressOut object with normalized address
        """

        # Extract components using rules
        rule_result = self.rules.normalize_patterns(raw_input)

        # Apply heuristics for missing components
        heuristic_result = self.rules.apply_heuristics(
            raw_input, rule_result["components"]
        )

        # Merge rule-based and heuristic components
        final_components = self._merge_components(
            address_dict, rule_result["components"], heuristic_result["components"]
        )

        # Calculate confidence and collect warnings
        confidence = self._calculate_confidence(rule_result, heuristic_result)
        warnings = self._collect_warnings(
            rule_result, heuristic_result, final_components
        )

        # Build AddressOut object
        return self._build_address_out(
            final_components=final_components,
            raw_input=raw_input,
            confidence=confidence,
            warnings=warnings,
            rule_result=rule_result,
            heuristic_result=heuristic_result,
        )

    def _merge_components(
        self,
        original: Dict[str, Any],
        rule_based: Dict[str, str],
        heuristic: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Merge components from different sources with priority:
        1. Original dictionary (highest priority)
        2. Rule-based extraction
        3. Heuristic assignment (lowest priority)
        """

        merged = {}

        # Mapping from our internal names to AddressOut field names
        field_mapping = {
            "city": "city",
            "district": "district",
            "neighborhood": "neighborhood",
            "street": "street",
            "building": "building",
            "block": "block",
            "number": "number",
            "entrance": "entrance",
            "floor": "floor",
            "apartment": "apartment",
            "postcode": "postcode",
        }

        # Start with original dictionary
        for key, value in original.items():
            if key in field_mapping and value and str(value).strip():
                mapped_key = field_mapping[key]
                merged[mapped_key] = str(value).strip()

        # Add rule-based components if not already present
        for key, value in rule_based.items():
            if key in field_mapping and value and str(value).strip():
                mapped_key = field_mapping[key]
                if mapped_key not in merged:
                    merged[mapped_key] = str(value).strip()

        # Add heuristic components if not already present
        for key, value in heuristic.items():
            if key in field_mapping and value and str(value).strip():
                mapped_key = field_mapping[key]
                if mapped_key not in merged:
                    merged[mapped_key] = str(value).strip()

        return merged

    def _calculate_confidence(self, rule_result: Dict, heuristic_result: Dict) -> float:
        """Calculate overall confidence based on rule and heuristic results."""

        rule_confidence = rule_result.get("confidence", 0.0)
        heuristic_confidence = heuristic_result.get("confidence", 0.0)

        # Weight rule-based confidence higher than heuristics
        if rule_confidence > 0:
            # If we have rule-based matches, use them with minor heuristic boost
            return min(rule_confidence * 0.8 + heuristic_confidence * 0.2, 1.0)
        else:
            # If only heuristics, very low confidence
            return min(heuristic_confidence * 0.3, 0.3)

    def _collect_warnings(
        self,
        rule_result: Dict,
        heuristic_result: Dict,
        final_components: Dict[str, str],
    ) -> List[str]:
        """Collect warnings from processing."""

        warnings = []

        # Add rule-based warnings
        warnings.extend(rule_result.get("warnings", []))

        # Add heuristic warnings
        warnings.extend(heuristic_result.get("warnings", []))

        # Add specific warnings based on component availability
        if len(final_components) == 0:
            warnings.append("No address components could be extracted")
        elif len(final_components) < 3:
            warnings.append(
                "Very few address components extracted - low quality result"
            )

        if "city" not in final_components and "district" not in final_components:
            warnings.append("No city or district information found")

        if "street" not in final_components and "neighborhood" not in final_components:
            warnings.append("No street or neighborhood information found")

        # Add fallback method warning
        warnings.append("Processed using fallback rules - lower confidence")

        return warnings

    def _build_address_out(
        self,
        final_components: Dict[str, str],
        raw_input: str,
        confidence: float,
        warnings: List[str],
        rule_result: Dict,
        heuristic_result: Dict,
    ) -> AddressOut:
        """Build the final AddressOut object."""

        # Create explanation parsed object
        explanation_parsed = ExplanationParsed(
            confidence=confidence, method=MethodEnum.FALLBACK, warnings=warnings
        )

        # Build normalized address string
        normalized_address = self._build_normalized_string(final_components)

        # Create AddressOut object - only include fields that have values
        address_kwargs = {
            "explanation_raw": raw_input,
            "explanation_parsed": explanation_parsed,
            "normalized_address": normalized_address,
        }

        # Add components only if they have values (never add null/empty strings)
        for field, value in final_components.items():
            if value and str(value).strip():
                address_kwargs[field] = str(value).strip()

        return AddressOut(**address_kwargs)

    def _build_normalized_string(self, components: Dict[str, str]) -> str:
        """Build a normalized address string from components."""

        # Define the order of components for the normalized string
        component_order = [
            "neighborhood",
            "street",
            "building",
            "block",
            "number",
            "entrance",
            "floor",
            "apartment",
            "district",
            "city",
            "postcode",
        ]

        parts = []

        for field in component_order:
            if field in components and components[field]:
                value = components[field]

                # Add appropriate suffixes for Turkish address components
                if field == "neighborhood" and not value.lower().endswith(
                    ("mahalle", "mahallesi")
                ):
                    value += " Mahallesi"
                elif field == "street":
                    # Street names might already have proper endings from rules
                    pass
                elif field == "block" and not value.lower().startswith("blok"):
                    value = f"{value} Blok"
                elif field == "number" and not value.lower().startswith("no"):
                    value = f"No: {value}"
                elif field == "floor" and not value.lower().startswith("kat"):
                    value = f"Kat: {value}"
                elif field == "apartment" and not value.lower().startswith("daire"):
                    value = f"Daire: {value}"

                parts.append(value)

        # Join with appropriate separators
        if parts:
            return ", ".join(parts)
        else:
            return "Adres bilgisi eksik"

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the fallback normalizer."""

        rule_stats = self.rules.get_pattern_stats()

        return {
            "normalizer_type": "fallback",
            "rule_engine": "TurkishAddressRules",
            "pattern_stats": rule_stats,
            "supported_fields": [
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
            ],
            "confidence_range": "0.1 - 0.8 (very low to medium)",
            "method": "FALLBACK",
        }


def create_fallback_normalizer() -> FallbackNormalizer:
    """Factory function to create a FallbackNormalizer instance."""
    return FallbackNormalizer()


# Convenience function for direct use
def normalize_address_fallback(
    address_dict: Dict[str, Any], raw_input: str
) -> AddressOut:
    """
    Convenience function to normalize an address using fallback rules.

    Args:
        address_dict: Dictionary with address components
        raw_input: Original raw address string

    Returns:
        AddressOut object with normalized address
    """
    normalizer = create_fallback_normalizer()
    return normalizer.normalize(address_dict, raw_input)
