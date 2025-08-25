"""
Legacy Normalizer

Eski kural tabanlı normalizasyon sistemi. Pattern ve ML başarısız olduğunda son şans olarak kullanılır.
Bu sistem çok basit regex kuralları ile adres componentlerini extract etmeye çalışır.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .patterns import LegacyPatterns
from .rule_engine import RuleEngine
from .validators import LegacyValidator

logger = logging.getLogger(__name__)


@dataclass
class LegacyResult:
    """Legacy normalization sonucu"""

    components: Dict[str, str]
    confidence: float
    processing_method: str = "legacy"
    extraction_details: Dict[str, Any] = None
    success: bool = True
    errors: List[str] = None


class LegacyNormalizer:
    """
    Legacy rule-based address normalizer

    En basit kural tabanlı sistem - hiçbir şey başarısız olmasın diye
    son güvenlik ağı olarak kullanılır.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize legacy normalizer"""
        self.config = config or {}
        self.patterns = LegacyPatterns()
        self.rule_engine = RuleEngine()
        self.validator = LegacyValidator()

        # Legacy specific settings
        self.min_confidence = self.config.get("min_confidence", 0.1)  # Çok düşük - son şans
        self.enable_heuristics = self.config.get("enable_heuristics", True)
        self.preserve_original = self.config.get("preserve_original", True)

        logger.info("Legacy normalizer initialized")

    def normalize(self, address: str) -> LegacyResult:
        """
        Legacy normalization - son şans yaklaşımı

        Args:
            address: Ham adres string

        Returns:
            LegacyResult with extracted components
        """
        start_time = datetime.now()

        try:
            # 1. Temel temizleme
            cleaned_address = self._clean_address(address)

            # 2. Component extraction
            components = self._extract_components(cleaned_address)

            # 3. Heuristic enhancement
            if self.enable_heuristics:
                components = self._apply_heuristics(components, cleaned_address)

            # 4. Validation
            components = self.validator.validate_components(components)

            # 5. Confidence calculation
            confidence = self._calculate_confidence(components, cleaned_address)

            # 6. Extraction details
            details = {
                "original_address": address,
                "cleaned_address": cleaned_address,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "rules_applied": len([k for k, v in components.items() if v]),
                "fallback_reason": "pattern_ml_failed",
            }

            return LegacyResult(components=components, confidence=confidence, extraction_details=details, success=True)

        except Exception as e:
            logger.error(f"Legacy normalization failed: {e}")

            # Son çare - en az bilgi
            return LegacyResult(
                components={"raw_address": address},
                confidence=0.01,  # En düşük confidence
                extraction_details={"error": str(e), "fallback_reason": "complete_failure"},
                success=False,
                errors=[str(e)],
            )

    def _clean_address(self, address: str) -> str:
        """Temel adres temizleme"""
        if not address:
            return ""

        # Unicode normalization
        cleaned = address.strip()

        # Çoklu boşlukları düzelt
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Özel karakterleri normalize et
        cleaned = re.sub(r"[/\\]", " ", cleaned)
        cleaned = re.sub(r"[.,;:]", " ", cleaned)

        # Türkçe karakter düzeltmeleri (basit)
        replacements = {
            "ı": "i",
            "İ": "I",
            "ğ": "g",
            "Ğ": "G",
            "ü": "u",
            "Ü": "U",
            "ş": "s",
            "Ş": "S",
            "ö": "o",
            "Ö": "O",
            "ç": "c",
            "Ç": "C",
        }

        # Sadece gerekirse normalize et
        if self.config.get("normalize_turkish_chars", False):
            for tr_char, en_char in replacements.items():
                cleaned = cleaned.replace(tr_char, en_char)

        return cleaned.strip()

    def _extract_components(self, address: str) -> Dict[str, str]:
        """Basit regex pattern'ler ile component extraction"""
        components = {}

        # Rule engine kullanarak extraction
        extraction_result = self.rule_engine.extract_all_components(address)

        # Legacy pattern'leri de dene
        legacy_components = self.patterns.extract_components(address)

        # Merge results - rule engine öncelikli
        components.update(legacy_components)
        components.update(extraction_result)

        # Temizleme
        for key, value in components.items():
            if value:
                components[key] = value.strip()

        return components

    def _apply_heuristics(self, components: Dict[str, str], address: str) -> Dict[str, str]:
        """Heuristic yaklaşımlarla eksik componentleri doldur"""

        # İl tahmini - büyük şehirler için basit heuristic
        if not components.get("il"):
            il_hints = self._guess_province(address)
            if il_hints:
                components["il"] = il_hints[0]  # En olası il

        # Adres tipi tahmini
        if not components.get("address_type"):
            address_type = self._guess_address_type(address)
            components["address_type"] = address_type

        # Mahalle/semt tahmini - "mahalle" kelimesi varsa
        if not components.get("mahalle") and "mahalle" in address.lower():
            mahalle_match = re.search(r"(\w+)\s+mahalle", address.lower())
            if mahalle_match:
                components["mahalle"] = mahalle_match.group(1).title()

        # Sokak tahmini - "sokak/cadde" kelimesi varsa
        if not components.get("sokak"):
            sokak_patterns = [r"(\w+(?:\s+\w+)*)\s+(?:sokak|sk|cad|cadde)", r"(\w+(?:\s+\w+)*)\s+(?:bulvar|blv)"]
            for pattern in sokak_patterns:
                match = re.search(pattern, address.lower())
                if match:
                    components["sokak"] = match.group(1).title()
                    break

        return components

    def _guess_province(self, address: str) -> List[str]:
        """Basit il tahmini"""
        address_lower = address.lower()

        # Büyük şehirler için basit kelime arama
        province_hints = {
            "istanbul": ["İstanbul"],
            "ankara": ["Ankara"],
            "izmir": ["İzmir"],
            "bursa": ["Bursa"],
            "antalya": ["Antalya"],
            "adana": ["Adana"],
            "konya": ["Konya"],
            "gaziantep": ["Gaziantep"],
            "kayseri": ["Kayseri"],
            "mersin": ["Mersin"],
        }

        for hint, provinces in province_hints.items():
            if hint in address_lower:
                return provinces

        return []

    def _guess_address_type(self, address: str) -> str:
        """Adres tipi tahmini"""
        address_lower = address.lower()

        commercial_keywords = ["mağaza", "dükkan", "ofis", "plaza", "avm", "merkez", "şirket"]
        residential_keywords = ["ev", "daire", "apart", "site", "konut", "blok"]

        commercial_score = sum(1 for kw in commercial_keywords if kw in address_lower)
        residential_score = sum(1 for kw in residential_keywords if kw in address_lower)

        if commercial_score > residential_score:
            return "commercial"
        elif residential_score > 0:
            return "residential"
        else:
            return "unknown"

    def _calculate_confidence(self, components: Dict[str, str], address: str) -> float:
        """Legacy confidence calculation - çok basit"""

        # Temel confidence factors
        base_confidence = 0.1  # Legacy minimum

        # Component sayısına göre artış
        filled_components = len([v for v in components.values() if v and v.strip()])
        component_bonus = min(0.1 * filled_components, 0.3)  # Max 0.3 bonus

        # Adres uzunluğuna göre normalizasyon
        address_length_factor = min(len(address) / 100.0, 1.0) * 0.1

        # Türkçe karakter varlığı
        turkish_chars = any(c in address for c in "ğıİşöüçĞIŞÖÜÇ")
        turkish_bonus = 0.05 if turkish_chars else 0.0

        # Toplam confidence
        total_confidence = base_confidence + component_bonus + address_length_factor + turkish_bonus

        # Legacy sistemde max confidence 0.5 olsun
        return min(total_confidence, 0.5)

    def batch_normalize(self, addresses: List[str]) -> List[LegacyResult]:
        """Batch normalization"""
        results = []

        for address in addresses:
            try:
                result = self.normalize(address)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch normalization failed for address '{address}': {e}")
                results.append(
                    LegacyResult(components={"raw_address": address}, confidence=0.01, success=False, errors=[str(e)])
                )

        return results
