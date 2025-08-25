"""
Legacy Validators

Legacy sistem için temel validation layer.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation sonucu"""

    is_valid: bool
    component: str
    value: str
    error_message: Optional[str] = None
    suggestion: Optional[str] = None


class LegacyValidator:
    """
    Legacy validation layer

    Çok basit validation kuralları - sadece temel kontroller.
    """

    def __init__(self):
        """Initialize validator"""
        self.known_provinces = self._get_known_provinces()
        self.known_districts = self._get_known_districts()
        self.validation_rules = self._create_validation_rules()

    def _get_known_provinces(self) -> Set[str]:
        """Bilinen il listesi (basit subset)"""
        return {
            "İstanbul",
            "Ankara",
            "İzmir",
            "Bursa",
            "Antalya",
            "Adana",
            "Konya",
            "Gaziantep",
            "Kayseri",
            "Mersin",
            "Diyarbakır",
            "Samsun",
            "Denizli",
            "Şanlıurfa",
            "Adapazarı",
            "Malatya",
            "Trabzon",
            "Van",
            "Erzurum",
            "Batman",
            "Elazığ",
            "Manisa",
            "Sivas",
            "Gebze",
            "Balıkesir",
            "Tarsus",
            "Kütahya",
            "Bandırma",
        }

    def _get_known_districts(self) -> Set[str]:
        """Bilinen ilçe listesi (basit subset)"""
        return {
            # İstanbul
            "Kadıköy",
            "Beşiktaş",
            "Şişli",
            "Beyoğlu",
            "Fatih",
            "Üsküdar",
            "Bakırköy",
            "Pendik",
            "Maltepe",
            "Kartal",
            "Ataşehir",
            "Zeytinburnu",
            # Ankara
            "Çankaya",
            "Keçiören",
            "Yenimahalle",
            "Mamak",
            "Sincan",
            "Etimesgut",
            "Altındağ",
            "Gölbaşı",
            "Pursaklar",
            # İzmir
            "Konak",
            "Bornova",
            "Buca",
            "Karşıyaka",
            "Bayraklı",
            "Gaziemir",
            "Balçova",
            "Çiğli",
            "Narlıdere",
        }

    def _create_validation_rules(self) -> Dict[str, callable]:
        """Validation kuralları oluştur"""
        return {
            "il": self._validate_province,
            "ilce": self._validate_district,
            "mahalle": self._validate_neighborhood,
            "sokak": self._validate_street,
            "bina_no": self._validate_building_number,
            "daire_no": self._validate_apartment_number,
            "kat": self._validate_floor,
            "site": self._validate_site,
            "blok": self._validate_block,
            "posta_kodu": self._validate_postal_code,
        }

    def validate_components(self, components: Dict[str, str]) -> Dict[str, str]:
        """
        Component'ları validate et ve temizle

        Args:
            components: Ham component'lar

        Returns:
            Validated ve temizlenmiş component'lar
        """
        validated_components = {}

        for component_name, value in components.items():
            if not value or not value.strip():
                continue

            # Validation uygula
            validation_result = self.validate_component(component_name, value)

            if validation_result.is_valid:
                # Geçerli component'ı ekle
                validated_components[component_name] = validation_result.value
            else:
                # Geçersiz component için uyarı log'la ama tamamen silme
                logger.warning(f"Invalid {component_name}: '{value}' - {validation_result.error_message}")

                # Suggestion varsa kullan
                if validation_result.suggestion:
                    validated_components[component_name] = validation_result.suggestion
                # Yoksa original değeri koru (legacy sistemde hiçbir şeyi kaybetme)
                else:
                    validated_components[component_name] = value

        return validated_components

    def validate_component(self, component_name: str, value: str) -> ValidationResult:
        """
        Tek component validate et

        Args:
            component_name: Component adı
            value: Component değeri

        Returns:
            ValidationResult
        """
        if not value or not value.strip():
            return ValidationResult(is_valid=False, component=component_name, value=value, error_message="Empty value")

        # İlgili validation fonksiyonunu çağır
        validator_func = self.validation_rules.get(component_name)
        if validator_func:
            return validator_func(value)
        else:
            # Bilinmeyen component - geçerli say
            return ValidationResult(is_valid=True, component=component_name, value=value.strip())

    def _validate_province(self, value: str) -> ValidationResult:
        """İl validation"""
        cleaned_value = value.strip().title()

        # Bilinen il listesinde var mı?
        if cleaned_value in self.known_provinces:
            return ValidationResult(is_valid=True, component="il", value=cleaned_value)

        # Fuzzy match dene
        suggestion = self._fuzzy_match_province(cleaned_value)

        return ValidationResult(
            is_valid=False,
            component="il",
            value=cleaned_value,
            error_message=f"Unknown province: {cleaned_value}",
            suggestion=suggestion,
        )

    def _validate_district(self, value: str) -> ValidationResult:
        """İlçe validation"""
        cleaned_value = value.strip().title()

        # Bilinen ilçe listesinde var mı?
        if cleaned_value in self.known_districts:
            return ValidationResult(is_valid=True, component="ilce", value=cleaned_value)

        # Legacy sistemde bilinmeyen ilçeler de geçerli sayılsın
        return ValidationResult(is_valid=True, component="ilce", value=cleaned_value)

    def _validate_neighborhood(self, value: str) -> ValidationResult:
        """Mahalle validation"""
        cleaned_value = value.strip().title()

        # Basit format kontrolleri
        if len(cleaned_value) < 2:
            return ValidationResult(
                is_valid=False, component="mahalle", value=cleaned_value, error_message="Too short for neighborhood name"
            )

        return ValidationResult(is_valid=True, component="mahalle", value=cleaned_value)

    def _validate_street(self, value: str) -> ValidationResult:
        """Sokak validation"""
        cleaned_value = value.strip().title()

        # Basit format kontrolleri
        if len(cleaned_value) < 2:
            return ValidationResult(
                is_valid=False, component="sokak", value=cleaned_value, error_message="Too short for street name"
            )

        return ValidationResult(is_valid=True, component="sokak", value=cleaned_value)

    def _validate_building_number(self, value: str) -> ValidationResult:
        """Bina numarası validation"""
        cleaned_value = value.strip()

        # Numara formatı kontrolü
        if re.match(r"^\d+(?:/\d+)?$", cleaned_value):
            return ValidationResult(is_valid=True, component="bina_no", value=cleaned_value)

        return ValidationResult(
            is_valid=False, component="bina_no", value=cleaned_value, error_message="Invalid building number format"
        )

    def _validate_apartment_number(self, value: str) -> ValidationResult:
        """Daire numarası validation"""
        cleaned_value = value.strip()

        # Daire numarası formatı
        if re.match(r"^\d+$", cleaned_value):
            return ValidationResult(is_valid=True, component="daire_no", value=cleaned_value)

        return ValidationResult(
            is_valid=False, component="daire_no", value=cleaned_value, error_message="Invalid apartment number format"
        )

    def _validate_floor(self, value: str) -> ValidationResult:
        """Kat validation"""
        cleaned_value = value.strip()

        # Kat numarası formatı (negatif katlar da olabilir)
        if re.match(r"^-?\d+$", cleaned_value):
            return ValidationResult(is_valid=True, component="kat", value=cleaned_value)

        return ValidationResult(
            is_valid=False, component="kat", value=cleaned_value, error_message="Invalid floor number format"
        )

    def _validate_site(self, value: str) -> ValidationResult:
        """Site validation"""
        cleaned_value = value.strip().title()

        # Basit format kontrolleri
        if len(cleaned_value) < 2:
            return ValidationResult(
                is_valid=False, component="site", value=cleaned_value, error_message="Too short for site name"
            )

        return ValidationResult(is_valid=True, component="site", value=cleaned_value)

    def _validate_block(self, value: str) -> ValidationResult:
        """Blok validation"""
        cleaned_value = value.strip().upper()

        # Blok formatı (harf veya harf+rakam kombinasyonu)
        if re.match(r"^[A-Z]\d*$|^\d+[A-Z]?$", cleaned_value):
            return ValidationResult(is_valid=True, component="blok", value=cleaned_value)

        return ValidationResult(is_valid=False, component="blok", value=cleaned_value, error_message="Invalid block format")

    def _validate_postal_code(self, value: str) -> ValidationResult:
        """Posta kodu validation"""
        cleaned_value = value.strip()

        # Türkiye posta kodu formatı (5 haneli)
        if re.match(r"^\d{5}$", cleaned_value):
            return ValidationResult(is_valid=True, component="posta_kodu", value=cleaned_value)

        return ValidationResult(
            is_valid=False,
            component="posta_kodu",
            value=cleaned_value,
            error_message="Invalid postal code format (should be 5 digits)",
        )

    def _fuzzy_match_province(self, value: str) -> Optional[str]:
        """İl için fuzzy match"""
        value_lower = value.lower()

        # Basit similarity check
        for known_province in self.known_provinces:
            known_lower = known_province.lower()

            # Exact match
            if value_lower == known_lower:
                return known_province

            # Starts with
            if known_lower.startswith(value_lower) or value_lower.startswith(known_lower):
                return known_province

            # Simple edit distance (very basic)
            if self._simple_similarity(value_lower, known_lower) > 0.8:
                return known_province

        return None

    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Basit similarity hesaplama"""
        if not s1 or not s2:
            return 0.0

        # Jaccard similarity (basit)
        set1 = set(s1)
        set2 = set(s2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def validate_all_components(self, components: Dict[str, str]) -> Dict[str, ValidationResult]:
        """Tüm component'ları validate et ve detaylı sonuç döndür"""
        results = {}

        for component_name, value in components.items():
            results[component_name] = self.validate_component(component_name, value)

        return results
