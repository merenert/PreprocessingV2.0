"""
Rule-based fallback normalization for Turkish addresses.
Handles common patterns and provides heuristic field assignment.
"""

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NormalizationRule:
    """Represents a normalization rule."""

    pattern: str
    replacement: str
    field: str
    confidence: float
    description: str


class TurkishAddressRules:
    """Rule-based normalization for Turkish addresses."""

    def __init__(self):
        self._compile_rules()
        self._compile_field_keywords()

    def _compile_rules(self):
        """Compile all normalization rules."""

        # Number patterns
        self.number_rules = [
            NormalizationRule(
                pattern=r"\b(?:no|num|numara|numero)\s*[:\s]*\s*(\d+[a-zA-Z]?)\b",
                replacement=r"\1",
                field="number",
                confidence=0.9,
                description="Number extraction: no 12, num:7, numara 45",
            ),
            NormalizationRule(
                pattern=r"\b(\d+)\s*(?:nolu|numarali|sayili)\b",
                replacement=r"\1",
                field="number",
                confidence=0.85,
                description="Number with suffix: 12 nolu, 7 numarali",
            ),
        ]

        # Block patterns
        self.block_rules = [
            NormalizationRule(
                pattern=r"\b([a-zA-Z])\s*blok\b",
                replacement=r"\1",
                field="block",
                confidence=0.9,
                description="Block pattern: a blok, B blok",
            ),
            NormalizationRule(
                pattern=r"\bblok\s*([a-zA-Z])\b",
                replacement=r"\1",
                field="block",
                confidence=0.9,
                description="Block pattern: blok a, blok B",
            ),
        ]

        # Floor patterns
        self.floor_rules = [
            NormalizationRule(
                pattern=r"\bkat\s*[:\s]*\s*(\d+)\b",
                replacement=r"\1",
                field="floor",
                confidence=0.9,
                description="Floor pattern: kat 3, kat:5",
            ),
            NormalizationRule(
                pattern=r"\b(\d+)\s*\.?\s*kat\b",
                replacement=r"\1",
                field="floor",
                confidence=0.9,
                description="Floor pattern: 3 kat, 5. kat",
            ),
            NormalizationRule(
                pattern=r"\bk(\d+)\b",
                replacement=r"\1",
                field="floor",
                confidence=0.7,
                description="Floor abbreviation: k3, k5",
            ),
        ]

        # Apartment/Door patterns
        self.apartment_rules = [
            NormalizationRule(
                pattern=r"\bdaire\s*[:\s]*\s*(\d+[a-zA-Z]?)\b",
                replacement=r"\1",
                field="apartment",
                confidence=0.9,
                description="Apartment pattern: daire 5, daire:12a",
            ),
            NormalizationRule(
                pattern=r"\b(\d+[a-zA-Z]?)\s*(?:nolu\s*)?daire\b",
                replacement=r"\1",
                field="apartment",
                confidence=0.9,
                description="Apartment pattern: 5 daire, 12a nolu daire",
            ),
            NormalizationRule(
                pattern=r"\bd(\d+[a-zA-Z]?)\b",
                replacement=r"\1",
                field="apartment",
                confidence=0.75,
                description="Apartment abbreviation: d5, d12a",
            ),
            NormalizationRule(
                pattern=r"\bkapi\s*[:\s]*\s*(\d+[a-zA-Z]?)\b",
                replacement=r"\1",
                field="apartment",
                confidence=0.8,
                description="Door pattern: kapi 5, kapi:12",
            ),
        ]

        # Street patterns
        self.street_rules = [
            NormalizationRule(
                pattern=r"\b(\d+)\s*(?:\.?\s*)?sokak\b",
                replacement=r"\1 Sokak",
                field="street",
                confidence=0.85,
                description="Street pattern: 123 sokak, 456. sokak",
            ),
            NormalizationRule(
                pattern=r"\b(\d+)\s*sk\b",
                replacement=r"\1 Sokak",
                field="street",
                confidence=0.8,
                description="Street abbreviation: 123 sk",
            ),
        ]

        # Boulevard patterns
        self.boulevard_rules = [
            NormalizationRule(
                pattern=r"\b(.+?)\s+bulvari?\b",
                replacement=r"\1 Bulvarı",
                field="street",
                confidence=0.9,
                description="Boulevard pattern: atatürk bulvarı",
            ),
            NormalizationRule(
                pattern=r"\b(.+?)\s+bul\b",
                replacement=r"\1 Bulvarı",
                field="street",
                confidence=0.8,
                description="Boulevard abbreviation: atatürk bul",
            ),
        ]

        # Avenue patterns
        self.avenue_rules = [
            NormalizationRule(
                pattern=r"\b(.+?)\s+caddesi?\b",
                replacement=r"\1 Caddesi",
                field="street",
                confidence=0.9,
                description="Avenue pattern: bagdat caddesi",
            ),
            NormalizationRule(
                pattern=r"\b(.+?)\s+cad\b",
                replacement=r"\1 Caddesi",
                field="street",
                confidence=0.8,
                description="Avenue abbreviation: bagdat cad",
            ),
        ]

    def _compile_field_keywords(self):
        """Compile keywords for field identification."""

        # Turkish city keywords (major cities)
        self.city_keywords = {
            "istanbul",
            "ankara",
            "izmir",
            "bursa",
            "antalya",
            "adana",
            "konya",
            "şanlıurfa",
            "gaziantep",
            "kocaeli",
            "mersin",
            "diyarbakır",
            "hatay",
            "manisa",
            "kayseri",
            "samsun",
            "balıkesir",
            "kahramanmaraş",
            "van",
            "aydın",
            "denizli",
            "muğla",
            "tekirdağ",
            "trabzon",
            "eskişehir",
            "malatya",
            "erzurum",
            "ordu",
            "afyonkarahisar",
            "tokat",
            "çorum",
            "kırıkkale",
            "uşak",
            "düzce",
            "osmaniye",
            "çanakkale",
            "kırşehir",
        }

        # District keywords (major districts)
        self.district_keywords = {
            "merkez",
            "center",
            "şehir",
            "belediye",
            "buca",
            "konak",
            "bornova",
            "karşıyaka",
            "alsancak",
            "çiğli",
            "gaziemir",
            "balçova",
            "narlıdere",
            "bayraklı",
            "güzelbahçe",
            "menemen",
            "urla",
            "seferihisar",
            "torbalı",
            "kemalpaşa",
            "foça",
            "selçuk",
            "tire",
            "ödemiş",
            "kiraz",
            "beydağ",
            "karaburun",
            "çeşme",
            "dikili",
            "bergama",
            "aliağa",
            "bayındır",
            "eşek",
            "doğanbey",
            "dağıstanhan",
            "akhisar",
            "soma",
            "kırkağaç",
            "yunusemre",
            "şehzadeler",
            "turgutlu",
            "alaşehir",
            "salihli",
            "sarıgöl",
            "gördes",
            "demirci",
            "köprübaşı",
            "ahmetli",
            "gölmarmara",
            "selendi",
        }

        # Neighborhood keywords
        self.neighborhood_keywords = {
            "mahalle",
            "mahallesi",
            "mah",
            "neighborhood",
            "quarter",
            "district",
        }

        # Street type keywords
        self.street_keywords = {
            "sokak",
            "sokağı",
            "sk",
            "street",
            "caddesi",
            "cadde",
            "cad",
            "avenue",
            "bulvarı",
            "bulvar",
            "bul",
            "boulevard",
            "yolu",
            "yol",
        }

    def normalize_patterns(self, text: str) -> Dict[str, any]:
        """
        Apply rule-based normalization to extract and normalize address components.

        Args:
            text: Input address text

        Returns:
            Dictionary with normalized components and metadata
        """

        result = {
            "components": {},
            "applied_rules": [],
            "confidence": 0.0,
            "warnings": [],
        }

        # Clean and normalize text
        normalized_text = self._clean_text(text)

        # Apply all rule categories
        rule_categories = [
            ("number", self.number_rules),
            ("block", self.block_rules),
            ("floor", self.floor_rules),
            ("apartment", self.apartment_rules),
            ("street", self.street_rules + self.boulevard_rules + self.avenue_rules),
        ]

        total_confidence = 0.0
        rule_count = 0

        for category, rules in rule_categories:
            for rule in rules:
                matches = re.finditer(rule.pattern, normalized_text, re.IGNORECASE)

                for match in matches:
                    extracted_value = re.sub(
                        rule.pattern,
                        rule.replacement,
                        match.group(),
                        flags=re.IGNORECASE,
                    )
                    extracted_value = extracted_value.strip()

                    if extracted_value and len(extracted_value) > 0:
                        # Store the component
                        if rule.field not in result["components"]:
                            result["components"][rule.field] = []

                        result["components"][rule.field].append(
                            {
                                "value": extracted_value,
                                "confidence": rule.confidence,
                                "source": rule.description,
                                "match": match.group(),
                            }
                        )

                        result["applied_rules"].append(
                            {
                                "rule": rule.description,
                                "field": rule.field,
                                "matched": match.group(),
                                "extracted": extracted_value,
                                "confidence": rule.confidence,
                            }
                        )

                        total_confidence += rule.confidence
                        rule_count += 1

        # Calculate overall confidence
        if rule_count > 0:
            result["confidence"] = min(total_confidence / rule_count, 1.0)

        # Deduplicate and select best values for each field
        result["components"] = self._select_best_components(result["components"])

        return result

    def apply_heuristics(
        self, text: str, existing_components: Dict[str, str] = None
    ) -> Dict[str, any]:
        """
        Apply heuristic rules to assign probable field values.
        Very low confidence assignments as last resort.

        Args:
            text: Input address text
            existing_components: Already identified components

        Returns:
            Dictionary with heuristic assignments
        """

        if existing_components is None:
            existing_components = {}

        result = {
            "components": {},
            "heuristics_applied": [],
            "confidence": 0.1,  # Very low confidence for heuristics
            "warnings": ["Heuristic assignment - low confidence"],
        }

        # Clean text for analysis
        words = self._clean_text(text).lower().split()

        # City heuristics
        if "city" not in existing_components:
            for word in words:
                if word in self.city_keywords:
                    result["components"]["city"] = word.title()
                    result["heuristics_applied"].append(
                        f"City heuristic: '{word}' matches city keyword"
                    )
                    break

        # District heuristics
        if "district" not in existing_components:
            for word in words:
                if word in self.district_keywords:
                    result["components"]["district"] = word.title()
                    result["heuristics_applied"].append(
                        f"District heuristic: '{word}' matches district keyword"
                    )
                    break

        # Neighborhood heuristics
        if "neighborhood" not in existing_components:
            # Look for words before neighborhood keywords
            for i, word in enumerate(words):
                if word in self.neighborhood_keywords and i > 0:
                    # Take 1-2 words before the keyword
                    neighborhood_parts = []
                    start_idx = max(0, i - 2)
                    for j in range(start_idx, i):
                        if (
                            words[j] not in self.street_keywords
                            and words[j] not in self.city_keywords
                        ):
                            neighborhood_parts.append(words[j].title())

                    if neighborhood_parts:
                        result["components"]["neighborhood"] = " ".join(
                            neighborhood_parts
                        )
                        result["heuristics_applied"].append(
                            f"Neighborhood heuristic: words before '{word}'"
                        )
                        break

        # Number heuristics - standalone numbers
        if "number" not in existing_components:
            for word in words:
                if re.match(r"^\d+[a-zA-Z]?$", word):
                    result["components"]["number"] = word
                    result["heuristics_applied"].append(
                        f"Number heuristic: standalone number '{word}'"
                    )
                    break

        return result

    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""

        # Convert to lowercase for processing
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common punctuation
        text = re.sub(r"[,\.;:]+", " ", text)

        return text.strip()

    def _select_best_components(
        self, components: Dict[str, List[Dict]]
    ) -> Dict[str, str]:
        """Select the best value for each component based on confidence."""

        result = {}

        for field, candidates in components.items():
            if candidates:
                # Sort by confidence (descending)
                best_candidate = max(candidates, key=lambda x: x["confidence"])
                result[field] = best_candidate["value"]

        return result

    def get_pattern_stats(self) -> Dict[str, int]:
        """Get statistics about available patterns."""

        return {
            "number_rules": len(self.number_rules),
            "block_rules": len(self.block_rules),
            "floor_rules": len(self.floor_rules),
            "apartment_rules": len(self.apartment_rules),
            "street_rules": len(
                self.street_rules + self.boulevard_rules + self.avenue_rules
            ),
            "city_keywords": len(self.city_keywords),
            "district_keywords": len(self.district_keywords),
            "total_rules": (
                len(self.number_rules)
                + len(self.block_rules)
                + len(self.floor_rules)
                + len(self.apartment_rules)
                + len(self.street_rules)
                + len(self.boulevard_rules)
                + len(self.avenue_rules)
            ),
        }
