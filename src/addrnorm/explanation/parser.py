"""
Main explanation parser for Turkish address landmark descriptions.
Combines landmark detection and spatial relation extraction.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from .models import ExplanationResult, ExplanationConfig, Landmark, SpatialRelation
from .landmarks import LandmarkDetector
from .relations import SpatialRelationExtractor


class ExplanationParser:
    """
    Main parser for Turkish address explanations.
    
    Processes text like "Amorium Hotel karşısı" and extracts:
    - Landmark: "Amorium Hotel" (type: hotel)
    - Spatial relation: "karşısı"
    """
    
    def __init__(self, config: Optional[ExplanationConfig] = None):
        """
        Initialize the explanation parser.
        
        Args:
            config: Configuration for parsing behavior
        """
        self.config = config or ExplanationConfig()
        self.landmark_detector = LandmarkDetector()
        self.relation_extractor = SpatialRelationExtractor()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if self.config.debug_mode:
            self.logger.setLevel(logging.DEBUG)
        
        # Common noise patterns to filter out
        self.noise_patterns = [
            r'\b\d{1,4}\s*(?:numara|no|sokak|cadde)\b',  # Street numbers
            r'\b(?:mahalle|mah|sokak|sok|cadde|cad)\b',   # Common address words
            r'\b(?:istanbul|ankara|izmir|bursa|antalya)\b',  # City names
        ]
        self.noise_regex = re.compile('|'.join(self.noise_patterns), re.IGNORECASE)
    
    def parse(self, explanation_text: str) -> ExplanationResult:
        """
        Parse explanation text and extract landmark and spatial information.
        
        Args:
            explanation_text: Text to parse (e.g., "Migros yanı")
            
        Returns:
            ExplanationResult with detected landmark and spatial relation
        """
        self.logger.debug(f"Parsing explanation: '{explanation_text}'")
        
        # Initialize result
        result = ExplanationResult(
            raw_explanation=explanation_text,
            type="landmark"
        )
        
        if not explanation_text or not explanation_text.strip():
            result.processing_notes.append("Empty or whitespace-only input")
            return result
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(explanation_text)
        self.logger.debug(f"Cleaned text: '{cleaned_text}'")
        
        # Extract landmarks
        landmarks = self.landmark_detector.detect_landmarks(cleaned_text)
        if landmarks:
            # Take the best landmark
            best_landmark = landmarks[0]
            if best_landmark.confidence >= self.config.landmark_detection_threshold:
                result.landmark = best_landmark
                self.logger.debug(f"Detected landmark: {best_landmark.name} ({best_landmark.type})")
            else:
                result.processing_notes.append(
                    f"Landmark confidence {best_landmark.confidence:.2f} below threshold "
                    f"{self.config.landmark_detection_threshold}"
                )
        
        # Extract spatial relations
        relations = self.relation_extractor.extract_relations(cleaned_text)
        if relations:
            # Take the best relation
            best_relation = relations[0]
            if best_relation.confidence >= self.config.relation_detection_threshold:
                result.relation = best_relation
                self.logger.debug(f"Detected relation: {best_relation.relation}")
            else:
                result.processing_notes.append(
                    f"Relation confidence {best_relation.confidence:.2f} below threshold "
                    f"{self.config.relation_detection_threshold}"
                )
        
        # Calculate overall confidence
        result.confidence = self._calculate_overall_confidence(result)
        
        # Add processing notes
        if not result.landmark and not result.relation:
            result.processing_notes.append("No landmarks or spatial relations detected")
        
        self.logger.debug(f"Final confidence: {result.confidence:.2f}")
        
        return result
    
    def parse_batch(self, explanation_texts: List[str]) -> List[ExplanationResult]:
        """
        Parse multiple explanation texts in batch.
        
        Args:
            explanation_texts: List of texts to parse
            
        Returns:
            List of ExplanationResult objects
        """
        results = []
        for text in explanation_texts:
            try:
                result = self.parse(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error parsing '{text}': {e}")
                error_result = ExplanationResult(
                    raw_explanation=text,
                    type="error",
                    processing_notes=[f"Parse error: {str(e)}"]
                )
                results.append(error_result)
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        cleaned = text.strip().lower()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common punctuation but keep Turkish characters
        cleaned = re.sub(r'[^\w\sğüşıöçĞÜŞIÖÇ]', ' ', cleaned)
        
        # Remove noise patterns if enabled
        if not self.config.debug_mode:  # Keep noise in debug mode for analysis
            cleaned = self.noise_regex.sub(' ', cleaned)
        
        # Remove extra whitespace again
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _calculate_overall_confidence(self, result: ExplanationResult) -> float:
        """
        Calculate overall confidence score for the parsing result.
        
        Args:
            result: ExplanationResult to calculate confidence for
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        confidences = []
        weights = []
        
        # Landmark confidence
        if result.landmark:
            confidences.append(result.landmark.confidence)
            weights.append(0.6)  # Landmark is more important
        
        # Spatial relation confidence
        if result.relation:
            confidences.append(result.relation.confidence)
            weights.append(0.4)  # Relation is less important
        
        if not confidences:
            return 0.0
        
        # Weighted average
        weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights))
        total_weight = sum(weights)
        
        overall_confidence = weighted_sum / total_weight
        
        # Apply bonus for having both landmark and relation
        if result.landmark and result.relation:
            overall_confidence += 0.1
        
        # Apply penalty for very short explanations
        if len(result.raw_explanation.strip()) < 5:
            overall_confidence -= 0.2
        
        return max(0.0, min(1.0, overall_confidence))
    
    def is_valid_result(self, result: ExplanationResult) -> bool:
        """
        Check if parsing result meets minimum quality thresholds.
        
        Args:
            result: ExplanationResult to validate
            
        Returns:
            True if result is valid, False otherwise
        """
        return result.confidence >= self.config.min_confidence_threshold
    
    def get_debug_info(self, explanation_text: str) -> Dict[str, Any]:
        """
        Get detailed debug information for parsing process.
        
        Args:
            explanation_text: Text to analyze
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            'original_text': explanation_text,
            'preprocessed_text': self._preprocess_text(explanation_text),
            'detected_landmarks': [],
            'detected_relations': [],
            'config': self.config.dict()
        }
        
        # Get all detected landmarks (not just the best one)
        landmarks = self.landmark_detector.detect_landmarks(explanation_text)
        for landmark in landmarks:
            debug_info['detected_landmarks'].append({
                'name': landmark.name,
                'type': landmark.type,
                'confidence': landmark.confidence
            })
        
        # Get all detected relations (not just the best one)
        relations = self.relation_extractor.extract_relations(explanation_text)
        for relation in relations:
            debug_info['detected_relations'].append({
                'relation': relation.relation,
                'confidence': relation.confidence
            })
        
        return debug_info


def create_parser(debug_mode: bool = False, 
                 min_confidence: float = 0.3) -> ExplanationParser:
    """
    Create a configured explanation parser.
    
    Args:
        debug_mode: Enable debug logging
        min_confidence: Minimum confidence threshold
        
    Returns:
        Configured ExplanationParser instance
    """
    config = ExplanationConfig(
        debug_mode=debug_mode,
        min_confidence_threshold=min_confidence
    )
    return ExplanationParser(config)


# Convenience functions for quick usage
def parse_explanation(text: str, debug: bool = False) -> Dict[str, Any]:
    """
    Quick parse function that returns JSON-compatible result.
    
    Args:
        text: Explanation text to parse
        debug: Enable debug mode
        
    Returns:
        Dictionary with parsing results
    """
    parser = create_parser(debug_mode=debug)
    result = parser.parse(text)
    return result.to_json_output()


def extract_landmark_info(text: str) -> Optional[Dict[str, str]]:
    """
    Simple function to extract just landmark information.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with landmark name and type, or None
    """
    parser = create_parser()
    result = parser.parse(text)
    
    if result.landmark and parser.is_valid_result(result):
        return {
            'name': result.landmark.name,
            'type': result.landmark.type
        }
    
    return None
