"""
Legacy Patterns

Basit regex pattern'ler ve kural tabanlı component extraction için pattern'ler.
"""

import re
from typing import Dict, List, Optional, Pattern
from dataclasses import dataclass


@dataclass
class PatternRule:
    """Single pattern rule"""

    name: str
    pattern: Pattern[str]
    component: str
    priority: int = 1
    description: str = ""


class LegacyPatterns:
    """
    Legacy regex pattern'leri ve basit extraction kuralları

    Çok basit regex'ler ile temel component'ları extract etmek için.
    """

    def __init__(self):
        """Initialize legacy patterns"""
        self.patterns = self._create_patterns()
        self.abbreviations = self._create_abbreviations()

    def _create_patterns(self) -> List[PatternRule]:
        """Basit regex pattern'ler oluştur"""

        patterns = []

        # İl pattern'leri
        patterns.append(
            PatternRule(
                name="province_suffix",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:ili|il)\b", re.IGNORECASE),
                component="il",
                priority=3,
                description="Province with 'ili' suffix",
            )
        )

        # İlçe pattern'leri
        patterns.append(
            PatternRule(
                name="district_suffix",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:ilçesi|ilçe)\b", re.IGNORECASE),
                component="ilce",
                priority=3,
                description="District with 'ilçesi' suffix",
            )
        )

        # Mahalle pattern'leri
        patterns.append(
            PatternRule(
                name="neighborhood_full",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:mahallesi|mahalle|mah)\b", re.IGNORECASE),
                component="mahalle",
                priority=3,
                description="Neighborhood with suffix",
            )
        )

        # Sokak/Cadde pattern'leri
        patterns.append(
            PatternRule(
                name="street_full",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:caddesi|cadde|cad)\b", re.IGNORECASE),
                component="sokak",
                priority=3,
                description="Street with suffix",
            )
        )

        patterns.append(
            PatternRule(
                name="sokak_full",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:sokağı|sokak|sk)\b", re.IGNORECASE),
                component="sokak",
                priority=3,
                description="Sokak with suffix",
            )
        )

        # Bulvar pattern
        patterns.append(
            PatternRule(
                name="boulevard",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:bulvarı|bulvar|blv)\b", re.IGNORECASE),
                component="sokak",
                priority=3,
                description="Boulevard",
            )
        )

        # Numara pattern'leri
        patterns.append(
            PatternRule(
                name="number_explicit",
                pattern=re.compile(r"\b(?:no|numara|num)[:.]?\s*(\d+(?:/\d+)?)\b", re.IGNORECASE),
                component="bina_no",
                priority=4,
                description="Explicit number",
            )
        )

        patterns.append(
            PatternRule(
                name="number_simple",
                pattern=re.compile(r"\b(\d+(?:/\d+)?)\b"),
                component="bina_no",
                priority=1,
                description="Simple number",
            )
        )

        # Daire pattern'leri
        patterns.append(
            PatternRule(
                name="apartment_explicit",
                pattern=re.compile(r"\b(?:daire|d|da|d\.)\s*[:.]?\s*(\d+)\b", re.IGNORECASE),
                component="daire_no",
                priority=3,
                description="Apartment number",
            )
        )

        # Kat pattern'leri
        patterns.append(
            PatternRule(
                name="floor_explicit",
                pattern=re.compile(r"\b(?:kat|k)\s*[:.]?\s*(\d+)\b", re.IGNORECASE),
                component="kat",
                priority=3,
                description="Floor number",
            )
        )

        # Site pattern'leri
        patterns.append(
            PatternRule(
                name="site_complex",
                pattern=re.compile(r"\b(\w+(?:\s+\w+)*)\s+(?:sitesi|site)\b", re.IGNORECASE),
                component="site",
                priority=2,
                description="Site/complex",
            )
        )

        # Blok pattern'leri
        patterns.append(
            PatternRule(
                name="block",
                pattern=re.compile(r"\b([A-Z]\d*|\d+[A-Z]?)\s+(?:blok|blok)\b", re.IGNORECASE),
                component="blok",
                priority=2,
                description="Block identifier",
            )
        )

        return patterns

    def _create_abbreviations(self) -> Dict[str, str]:
        """Kısaltma çözümleme tablosu"""
        return {
            # Genel kısaltmalar
            "mh": "mahalle",
            "mah": "mahalle",
            "cd": "cadde",
            "cad": "cadde",
            "sk": "sokak",
            "blv": "bulvar",
            "no": "numara",
            "d": "daire",
            "da": "daire",
            "k": "kat",
            # İl kısaltmaları (örnekler)
            "ist": "İstanbul",
            "ank": "Ankara",
            "izm": "İzmir",
            # Yön ve pozisyon
            "krs": "karşısı",
            "yan": "yanı",
            "ark": "arkası",
            "ust": "üstü",
            "alt": "altı",
        }

    def extract_components(self, address: str) -> Dict[str, str]:
        """
        Pattern'ler ile component extraction

        Args:
            address: Ham adres string

        Returns:
            Dict of extracted components
        """
        components = {}

        if not address:
            return components

        # Kısaltmaları çöz
        expanded_address = self._expand_abbreviations(address)

        # Her pattern'i dene
        for pattern_rule in sorted(self.patterns, key=lambda x: x.priority, reverse=True):
            match = pattern_rule.pattern.search(expanded_address)
            if match and not components.get(pattern_rule.component):
                components[pattern_rule.component] = match.group(1).strip()

        return components

    def _expand_abbreviations(self, address: str) -> str:
        """Kısaltmaları aç"""
        expanded = address

        for abbrev, full in self.abbreviations.items():
            # Kelime sınırlarında kısaltma arama
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            expanded = re.sub(pattern, full, expanded, flags=re.IGNORECASE)

        return expanded

    def get_pattern_info(self, component: str) -> List[PatternRule]:
        """Belirli component için pattern'leri getir"""
        return [p for p in self.patterns if p.component == component]

    def test_pattern(self, pattern_name: str, test_address: str) -> Optional[str]:
        """Pattern test etmek için"""
        for pattern_rule in self.patterns:
            if pattern_rule.name == pattern_name:
                match = pattern_rule.pattern.search(test_address)
                return match.group(1) if match else None
        return None
