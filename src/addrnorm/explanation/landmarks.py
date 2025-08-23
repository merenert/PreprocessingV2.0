"""
Landmark detection for Turkish address explanations.
Identifies businesses, institutions, and points of interest mentioned in text.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from .models import Landmark


class LandmarkDetector:
    """Detects and extracts landmark information from Turkish text."""
    
    def __init__(self):
        """Initialize landmark patterns and categories."""
        self.landmark_categories = {
            # Accommodation
            'hotel': ['hotel', 'otel', 'motel', 'pansiyon', 'konaklama', 'misafirhane'],
            'otel': ['hotel', 'otel', 'motel', 'pansiyon', 'konaklama', 'misafirhane'],
            
            # Commercial
            'market': ['market', 'marketler', 'süpermarket', 'hipermarket', 'bakkal', 'büfe'],
            'mağaza': ['mağaza', 'mağazası', 'dükkan', 'dükkanı', 'market', 'shop'],
            'alışveriş': ['avm', 'alışveriş', 'merkezi', 'plaza', 'çarşı', 'pazar'],
            
            # Healthcare
            'hastane': ['hastane', 'hastanesi', 'klinik', 'kliniki', 'sağlık', 'tıp', 'doktor'],
            'eczane': ['eczane', 'eczanesi', 'pharmacy'],
            
            # Education
            'okul': ['okul', 'okulu', 'school', 'lise', 'ilkokul', 'ortaokul', 'anaokul'],
            'üniversite': ['üniversite', 'üniversitesi', 'university', 'kolej', 'college'],
            
            # Religious
            'cami': ['cami', 'camii', 'mosque', 'mescit', 'mescidi'],
            
            # Recreation
            'park': ['park', 'parkı', 'bahçe', 'bahçesi', 'mesire', 'rekreasyon'],
            'spor': ['spor', 'gym', 'fitness', 'stadyum', 'salon', 'kulüp'],
            
            # Financial
            'banka': ['banka', 'bankası', 'bank', 'atm', 'şube', 'şubesi'],
            
            # Food & Beverage
            'restoran': ['restoran', 'restoranı', 'restaurant', 'lokanta', 'lokantası'],
            'kafe': ['kafe', 'cafe', 'kahvehane', 'çay', 'kahve'],
            
            # Services
            'berber': ['berber', 'berberi', 'barber', 'kuaför', 'kuaförü'],
            'benzinlik': ['benzinlik', 'benzin', 'istasyon', 'akaryakıt', 'petrol'],
            'otopark': ['otopark', 'park', 'garaj', 'garage'],
            
            # Transportation
            'terminal': ['terminal', 'terminali', 'otogar', 'otobüs', 'bus'],
            'istasyon': ['istasyon', 'station', 'metro', 'tren', 'railway'],
            'durak': ['durak', 'durağı', 'stop', 'minibüs', 'dolmuş'],
            
            # Business/Industrial
            'fabrika': ['fabrika', 'fabrikası', 'factory', 'üretim', 'imalat'],
            'atölye': ['atölye', 'atölyesi', 'workshop', 'tamirci', 'tamiri'],
            'şirket': ['şirket', 'şirketi', 'company', 'firma', 'firması'],
            'limited': ['limited', 'ltd', 'ltd.', 'şti', 'şti.'],
            'anonim': ['anonim', 'a.ş', 'a.ş.', 'as', 'anonim şirket'],
            
            # Organizations
            'dernek': ['dernek', 'derneği', 'association', 'birlik', 'birliği'],
            'vakıf': ['vakıf', 'vakfı', 'foundation', 'kurum', 'kurumu'],
        }
        
        # Company/business suffixes that indicate commercial entities
        self.business_suffixes = {
            'limited': ['ltd', 'ltd.', 'limited', 'şti', 'şti.'],
            'anonim': ['a.ş', 'a.ş.', 'as', 'anonim'],
            'kollektif': ['kol', 'kol.', 'kollektif'],
            'kooperatif': ['koop', 'koop.', 'kooperatif', 'cooperative'],
            'ticaret': ['ticaret', 'tic', 'tic.', 'trade', 'trading']
        }
        
        # Compile patterns
        self.category_patterns = self._compile_category_patterns()
        self.business_patterns = self._compile_business_patterns()
        
        # Confidence weights
        self.confidence_weights = {
            'exact_match': 1.0,
            'suffix_match': 0.8,
            'partial_match': 0.6,
            'fuzzy_match': 0.4
        }
    
    def _compile_category_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for landmark categories."""
        patterns = {}
        
        for category, keywords in self.landmark_categories.items():
            category_patterns = []
            for keyword in keywords:
                # Pattern to match keyword with optional surrounding text
                pattern = rf'\b{re.escape(keyword)}\b'
                category_patterns.append(re.compile(pattern, re.IGNORECASE | re.UNICODE))
            patterns[category] = category_patterns
        
        return patterns
    
    def _compile_business_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for business suffixes."""
        patterns = {}
        
        for suffix_type, suffixes in self.business_suffixes.items():
            suffix_pattern = '|'.join(re.escape(suffix) for suffix in suffixes)
            pattern = rf'\b(?:{suffix_pattern})\b'
            patterns[suffix_type] = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        
        return patterns
    
    def detect_landmarks(self, text: str) -> List[Landmark]:
        """
        Detect landmarks in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected landmarks with confidence scores
        """
        landmarks = []
        text_clean = text.lower().strip()
        
        # Detect category-based landmarks
        landmarks.extend(self._detect_category_landmarks(text, text_clean))
        
        # Detect business entities
        landmarks.extend(self._detect_business_landmarks(text, text_clean))
        
        # Remove duplicates and sort by confidence
        landmarks = self._deduplicate_landmarks(landmarks)
        landmarks.sort(key=lambda x: x.confidence, reverse=True)
        
        return landmarks[:5]  # Return top 5 landmarks
    
    def _detect_category_landmarks(self, original_text: str, text_lower: str) -> List[Landmark]:
        """Detect landmarks based on category keywords."""
        landmarks = []
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                matches = list(pattern.finditer(text_lower))
                
                for match in matches:
                    # Extract potential landmark name around the match
                    landmark_info = self._extract_landmark_name(
                        original_text, text_lower, match, category
                    )
                    
                    if landmark_info:
                        landmarks.append(landmark_info)
        
        return landmarks
    
    def _detect_business_landmarks(self, original_text: str, text_lower: str) -> List[Landmark]:
        """Detect business entities based on suffixes."""
        landmarks = []
        
        for suffix_type, pattern in self.business_patterns.items():
            matches = list(pattern.finditer(text_lower))
            
            for match in matches:
                # Extract business name before the suffix
                landmark_info = self._extract_business_name(
                    original_text, text_lower, match, suffix_type
                )
                
                if landmark_info:
                    landmarks.append(landmark_info)
        
        return landmarks
    
    def _extract_landmark_name(self, original_text: str, text_lower: str, 
                             match: re.Match, category: str) -> Optional[Landmark]:
        """Extract landmark name around a category match."""
        start_pos = match.start()
        end_pos = match.end()
        
        # Expand to capture full landmark name
        # Look for words before the category keyword
        words_before = []
        text_before = text_lower[:start_pos].strip()
        if text_before:
            words_before = text_before.split()[-3:]  # Take up to 3 words before
        
        # Look for words after the category keyword
        words_after = []
        text_after = text_lower[end_pos:].strip()
        if text_after:
            words_after = text_after.split()[:2]  # Take up to 2 words after
        
        # Construct landmark name
        all_words = words_before + [match.group()] + words_after
        
        # Filter out common noise words
        noise_words = {'ve', 'ile', 'da', 'de', 'ki', 'olan', 'bulunan', 'yer', 'yeri'}
        filtered_words = [w for w in all_words if w not in noise_words and len(w) > 1]
        
        if not filtered_words:
            return None
        
        # Build landmark name (use original case)
        landmark_name = self._reconstruct_original_case(
            original_text, ' '.join(filtered_words)
        )
        
        # Calculate confidence
        confidence = self._calculate_landmark_confidence(
            landmark_name, category, 'category_match'
        )
        
        return Landmark(
            name=landmark_name.strip(),
            type=category,
            confidence=confidence,
            raw_text=original_text
        )
    
    def _extract_business_name(self, original_text: str, text_lower: str,
                              match: re.Match, suffix_type: str) -> Optional[Landmark]:
        """Extract business name before a business suffix."""
        start_pos = match.start()
        
        # Look for words before the suffix
        text_before = text_lower[:start_pos].strip()
        if not text_before:
            return None
        
        words_before = text_before.split()[-4:]  # Take up to 4 words before suffix
        
        if not words_before:
            return None
        
        # Construct business name
        business_name = self._reconstruct_original_case(
            original_text, ' '.join(words_before) + ' ' + match.group()
        )
        
        # Calculate confidence
        confidence = self._calculate_landmark_confidence(
            business_name, suffix_type, 'business_match'
        )
        
        return Landmark(
            name=business_name.strip(),
            type=suffix_type,
            confidence=confidence,
            raw_text=original_text
        )
    
    def _reconstruct_original_case(self, original_text: str, lowercase_text: str) -> str:
        """Reconstruct original case from lowercase text."""
        # Simple approach: find the text in original and return it
        original_lower = original_text.lower()
        start_idx = original_lower.find(lowercase_text.lower())
        
        if start_idx != -1:
            end_idx = start_idx + len(lowercase_text)
            return original_text[start_idx:end_idx]
        
        return lowercase_text
    
    def _calculate_landmark_confidence(self, name: str, category: str, match_type: str) -> float:
        """Calculate confidence score for a detected landmark."""
        base_confidence = self.confidence_weights.get(match_type, 0.5)
        
        # Adjust based on name length and quality
        name_words = name.split()
        
        # Penalize very short names
        if len(name) < 3:
            base_confidence -= 0.3
        
        # Boost multi-word names (more likely to be real landmarks)
        if len(name_words) > 1:
            base_confidence += 0.1
        
        # Penalize if name contains only numbers
        if name.replace(' ', '').isdigit():
            base_confidence -= 0.5
        
        # Boost if contains typical business words
        business_indicators = ['merkez', 'center', 'grup', 'group', 'international']
        if any(indicator in name.lower() for indicator in business_indicators):
            base_confidence += 0.05
        
        return max(0.0, min(1.0, base_confidence))
    
    def _deduplicate_landmarks(self, landmarks: List[Landmark]) -> List[Landmark]:
        """Remove duplicate landmarks based on name similarity."""
        if not landmarks:
            return landmarks
        
        unique_landmarks = []
        seen_names = set()
        
        for landmark in landmarks:
            name_key = landmark.name.lower().strip()
            
            # Check if we've seen a very similar name
            is_duplicate = False
            for seen_name in seen_names:
                if self._names_are_similar(name_key, seen_name):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_landmarks.append(landmark)
                seen_names.add(name_key)
        
        return unique_landmarks
    
    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """Check if two landmark names are similar enough to be duplicates."""
        # Simple similarity check
        if name1 == name2:
            return True
        
        # Check if one is contained in the other
        if name1 in name2 or name2 in name1:
            return True
        
        # Check word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            min_words = min(len(words1), len(words2))
            overlap_ratio = overlap / min_words
            
            return overlap_ratio > 0.7
        
        return False
    
    def get_supported_categories(self) -> List[str]:
        """Get list of all supported landmark categories."""
        return list(self.landmark_categories.keys())
    
    def get_category_keywords(self, category: str) -> List[str]:
        """Get keywords for a specific landmark category."""
        return self.landmark_categories.get(category, [])
