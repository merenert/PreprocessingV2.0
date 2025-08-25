"""
Rule Engine

Legacy rule-based component extraction engine.
Sıralı kurallarla adres component'larını extract eder.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Address component types"""

    IL = "il"
    ILCE = "ilce"
    MAHALLE = "mahalle"
    SOKAK = "sokak"
    BINA_NO = "bina_no"
    DAIRE_NO = "daire_no"
    KAT = "kat"
    SITE = "site"
    BLOK = "blok"
    POSTA_KODU = "posta_kodu"


@dataclass
class ExtractionRule:
    """Single extraction rule"""

    name: str
    component: ComponentType
    pattern: str
    priority: int = 1
    required: bool = False
    post_process: Optional[callable] = None


class RuleEngine:
    """
    Legacy rule-based extraction engine

    Basit kurallarla adres component'larını extract eder.
    """

    def __init__(self):
        """Initialize rule engine"""
        self.rules = self._create_extraction_rules()
        self.priority_order = self._get_priority_order()

    def _create_extraction_rules(self) -> List[ExtractionRule]:
        """Extraction kuralları oluştur"""

        rules = []

        # İl extraction rules
        rules.append(
            ExtractionRule(
                name="il_with_suffix",
                component=ComponentType.IL,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:ili|il)\b",
                priority=10,
                post_process=self._clean_province,
            )
        )

        rules.append(
            ExtractionRule(
                name="il_from_known_list",
                component=ComponentType.IL,
                pattern=r"\b(İstanbul|Ankara|İzmir|Bursa|Antalya|Adana|Konya|Gaziantep|Kayseri|Mersin)\b",
                priority=8,
            )
        )

        # İlçe extraction rules
        rules.append(
            ExtractionRule(
                name="ilce_with_suffix",
                component=ComponentType.ILCE,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:ilçesi|ilçe)\b",
                priority=9,
                post_process=self._clean_district,
            )
        )

        # Mahalle extraction rules
        rules.append(
            ExtractionRule(
                name="mahalle_with_suffix",
                component=ComponentType.MAHALLE,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:mahallesi|mahalle|mah)\b",
                priority=8,
                post_process=self._clean_neighborhood,
            )
        )

        # Sokak/Cadde extraction rules
        rules.append(
            ExtractionRule(
                name="cadde_with_suffix",
                component=ComponentType.SOKAK,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:caddesi|cadde|cad)\b",
                priority=7,
                post_process=self._clean_street,
            )
        )

        rules.append(
            ExtractionRule(
                name="sokak_with_suffix",
                component=ComponentType.SOKAK,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:sokağı|sokak|sk)\b",
                priority=7,
                post_process=self._clean_street,
            )
        )

        rules.append(
            ExtractionRule(
                name="bulvar_with_suffix",
                component=ComponentType.SOKAK,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:bulvarı|bulvar|blv)\b",
                priority=7,
                post_process=self._clean_street,
            )
        )

        # Numara extraction rules
        rules.append(
            ExtractionRule(
                name="bina_no_explicit",
                component=ComponentType.BINA_NO,
                pattern=r"\b(?:no|numara|num)[:.]?\s*(\d+(?:/\d+)?)\b",
                priority=6,
                post_process=self._clean_number,
            )
        )

        rules.append(
            ExtractionRule(
                name="bina_no_simple",
                component=ComponentType.BINA_NO,
                pattern=r"\b(\d+(?:/\d+)?)\b",
                priority=2,
                post_process=self._clean_number,
            )
        )

        # Daire extraction rules
        rules.append(
            ExtractionRule(
                name="daire_explicit",
                component=ComponentType.DAIRE_NO,
                pattern=r"\b(?:daire|d|da|d\.)\s*[:.]?\s*(\d+)\b",
                priority=5,
                post_process=self._clean_apartment,
            )
        )

        # Kat extraction rules
        rules.append(
            ExtractionRule(
                name="kat_explicit",
                component=ComponentType.KAT,
                pattern=r"\b(?:kat|k)\s*[:.]?\s*(\d+)\b",
                priority=5,
                post_process=self._clean_floor,
            )
        )

        # Site extraction rules
        rules.append(
            ExtractionRule(
                name="site_with_suffix",
                component=ComponentType.SITE,
                pattern=r"\b(\w+(?:\s+\w+)*)\s+(?:sitesi|site)\b",
                priority=4,
                post_process=self._clean_site,
            )
        )

        # Blok extraction rules
        rules.append(
            ExtractionRule(
                name="blok_identifier",
                component=ComponentType.BLOK,
                pattern=r"\b([A-Z]\d*|\d+[A-Z]?)\s+(?:blok|blok)\b",
                priority=4,
                post_process=self._clean_block,
            )
        )

        # Posta kodu extraction
        rules.append(
            ExtractionRule(
                name="postal_code",
                component=ComponentType.POSTA_KODU,
                pattern=r"\b(\d{5})\b",
                priority=3,
                post_process=self._clean_postal_code,
            )
        )

        return rules

    def _get_priority_order(self) -> List[ComponentType]:
        """Component extraction sırası"""
        return [
            ComponentType.IL,
            ComponentType.ILCE,
            ComponentType.MAHALLE,
            ComponentType.SOKAK,
            ComponentType.BINA_NO,
            ComponentType.DAIRE_NO,
            ComponentType.KAT,
            ComponentType.SITE,
            ComponentType.BLOK,
            ComponentType.POSTA_KODU,
        ]

    def extract_all_components(self, address: str) -> Dict[str, str]:
        """
        Tüm component'ları extract et

        Args:
            address: Ham adres string

        Returns:
            Dict of extracted components
        """
        if not address:
            return {}

        components = {}

        # Her component türü için extraction
        for component_type in self.priority_order:
            if component_type.value not in components:
                extracted = self.extract_component(address, component_type)
                if extracted:
                    components[component_type.value] = extracted

        return components

    def extract_component(self, address: str, component_type: ComponentType) -> Optional[str]:
        """
        Belirli bir component'ı extract et

        Args:
            address: Ham adres string
            component_type: Extract edilecek component türü

        Returns:
            Extracted component value or None
        """
        # Bu component türü için kuralları getir
        component_rules = [r for r in self.rules if r.component == component_type]

        # Priority sırasına göre sırala
        component_rules.sort(key=lambda x: x.priority, reverse=True)

        # Her kuralı dene
        for rule in component_rules:
            match = re.search(rule.pattern, address, re.IGNORECASE)
            if match:
                extracted_value = match.group(1).strip()

                # Post-processing uygula
                if rule.post_process:
                    extracted_value = rule.post_process(extracted_value)

                if extracted_value:  # Boş değilse kabul et
                    logger.debug(f"Extracted {component_type.value}: '{extracted_value}' using rule '{rule.name}'")
                    return extracted_value

        return None

    def extract_sequential(self, address: str) -> Dict[str, str]:
        """
        Sıralı extraction - bir component extract edildiğinde adresten çıkar

        Args:
            address: Ham adres string

        Returns:
            Dict of extracted components
        """
        components = {}
        remaining_address = address

        for component_type in self.priority_order:
            extracted = self.extract_component(remaining_address, component_type)
            if extracted:
                components[component_type.value] = extracted

                # Extract edilen kısmı adresten çıkar
                # Bu gelecek extraction'ları etkilemesin diye
                remaining_address = self._remove_extracted_part(remaining_address, extracted)

        return components

    def _remove_extracted_part(self, address: str, extracted_part: str) -> str:
        """Extract edilen kısmı adresten çıkar"""
        # Basit remove - daha sofistike yapılabilir
        cleaned = address.replace(extracted_part, "", 1)
        cleaned = re.sub(r"\s+", " ", cleaned)  # Çoklu boşlukları düzelt
        return cleaned.strip()

    # Post-processing functions
    def _clean_province(self, value: str) -> str:
        """İl değerini temizle"""
        return value.title().strip()

    def _clean_district(self, value: str) -> str:
        """İlçe değerini temizle"""
        return value.title().strip()

    def _clean_neighborhood(self, value: str) -> str:
        """Mahalle değerini temizle"""
        return value.title().strip()

    def _clean_street(self, value: str) -> str:
        """Sokak değerini temizle"""
        return value.title().strip()

    def _clean_number(self, value: str) -> str:
        """Numara değerini temizle"""
        return value.strip()

    def _clean_apartment(self, value: str) -> str:
        """Daire değerini temizle"""
        return value.strip()

    def _clean_floor(self, value: str) -> str:
        """Kat değerini temizle"""
        return value.strip()

    def _clean_site(self, value: str) -> str:
        """Site değerini temizle"""
        return value.title().strip()

    def _clean_block(self, value: str) -> str:
        """Blok değerini temizle"""
        return value.upper().strip()

    def _clean_postal_code(self, value: str) -> str:
        """Posta kodu değerini temizle"""
        return value.strip()

    def get_rules_for_component(self, component_type: ComponentType) -> List[ExtractionRule]:
        """Belirli component türü için kuralları getir"""
        return [r for r in self.rules if r.component == component_type]

    def add_custom_rule(self, rule: ExtractionRule):
        """Özel kural ekle"""
        self.rules.append(rule)

    def validate_extraction(self, components: Dict[str, str]) -> Dict[str, bool]:
        """Extraction sonuçlarını validate et"""
        validation_results = {}

        for component_name, value in components.items():
            # Basit validation - boş değil mi?
            validation_results[component_name] = bool(value and value.strip())

        return validation_results
