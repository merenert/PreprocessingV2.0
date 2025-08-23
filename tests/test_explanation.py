"""
Unit tests for the explanation parsing module.
Tests landmark detection, spatial relation extraction, and integration.
"""

import pytest
from addrnorm.explanation import (
    ExplanationParser,
    parse_explanation,
    extract_landmark_info,
    LandmarkDetector,
    SpatialRelationExtractor,
    ExplanationConfig
)


class TestLandmarkDetector:
    """Test landmark detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = LandmarkDetector()
    
    def test_hotel_detection(self):
        """Test hotel landmark detection."""
        text = "Amorium Hotel karşısı"
        landmarks = self.detector.detect_landmarks(text)
        
        assert len(landmarks) > 0
        hotel = landmarks[0]
        assert "hotel" in hotel.name.lower()
        assert hotel.type in ['hotel', 'otel']
        assert hotel.confidence > 0.5
    
    def test_market_detection(self):
        """Test market landmark detection."""
        text = "Migros yanında"
        landmarks = self.detector.detect_landmarks(text)
        
        assert len(landmarks) > 0
        market = landmarks[0]
        assert "migros" in market.name.lower()
        assert market.type == 'market'
    
    def test_bank_detection(self):
        """Test bank landmark detection."""
        text = "Şekerbank şubesi önünde"
        landmarks = self.detector.detect_landmarks(text)
        
        assert len(landmarks) > 0
        bank = landmarks[0]
        assert "şekerbank" in bank.name.lower()
        assert bank.type == 'banka'
    
    def test_business_suffix_detection(self):
        """Test business entity detection by suffix."""
        test_cases = [
            ("Koç Holding A.Ş.", "anonim"),
            ("ABC Limited Şti.", "limited"),
            ("XYZ Ticaret Ltd.", "limited")
        ]
        
        for text, expected_type in test_cases:
            landmarks = self.detector.detect_landmarks(text)
            assert len(landmarks) > 0
            business = landmarks[0]
            assert business.type == expected_type
    
    def test_multiple_landmarks(self):
        """Test detection of multiple landmarks in text."""
        text = "Migros ve Şekerbank arasında"
        landmarks = self.detector.detect_landmarks(text)
        
        # Should detect both Migros and Şekerbank
        assert len(landmarks) >= 2
        
        landmark_names = [l.name.lower() for l in landmarks]
        assert any("migros" in name for name in landmark_names)
        assert any("şekerbank" in name for name in landmark_names)
    
    def test_no_landmarks(self):
        """Test text with no detectable landmarks."""
        text = "5. sokak numara 12"
        landmarks = self.detector.detect_landmarks(text)
        
        # Should not detect any significant landmarks
        assert len(landmarks) == 0 or all(l.confidence < 0.5 for l in landmarks)


class TestSpatialRelationExtractor:
    """Test spatial relation extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = SpatialRelationExtractor()
    
    def test_basic_relations(self):
        """Test basic spatial relation detection."""
        test_cases = [
            ("hotel karşısı", "karşısı"),
            ("market yanında", "yanı"),
            ("banka arkasında", "arkası"),
            ("okul önünde", "önü"),
            ("plaza üstünde", "üstü"),
            ("garage altında", "altı")
        ]
        
        for text, expected_relation in test_cases:
            relations = self.extractor.extract_relations(text)
            assert len(relations) > 0
            
            best_relation = relations[0]
            assert best_relation.relation == expected_relation
            assert best_relation.confidence > 0.5
    
    def test_relation_variations(self):
        """Test different variations of spatial relations."""
        test_cases = [
            "karşısında duran bina",
            "yan tarafında bulunan",
            "arkasından gelen",
            "önde olan yer"
        ]
        
        for text in test_cases:
            relations = self.extractor.extract_relations(text)
            assert len(relations) > 0
            assert relations[0].confidence > 0.3
    
    def test_multiple_relations(self):
        """Test text with multiple spatial relations."""
        text = "hotel karşısında ve market yanında"
        relations = self.extractor.extract_relations(text)
        
        assert len(relations) >= 2
        relation_types = [r.relation for r in relations]
        assert "karşısı" in relation_types
        assert "yanı" in relation_types
    
    def test_no_relations(self):
        """Test text with no spatial relations."""
        text = "istanbul kadıköy moda mahalle"
        relations = self.extractor.extract_relations(text)
        
        assert len(relations) == 0 or all(r.confidence < 0.3 for r in relations)


class TestExplanationParser:
    """Test the main explanation parser integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.parser = ExplanationParser()
    
    def test_complete_explanation_parsing(self):
        """Test parsing complete explanations with landmark and relation."""
        test_cases = [
            {
                "input": "Migros yanı",
                "expected_landmark": "migros",
                "expected_relation": "yanı"
            },
            {
                "input": "Amorium Hotel karşısı",
                "expected_landmark": "hotel",
                "expected_relation": "karşısı"
            },
            {
                "input": "Şekerbank şubesi önünde",
                "expected_landmark": "şekerbank",
                "expected_relation": "önü"
            }
        ]
        
        for case in test_cases:
            result = self.parser.parse(case["input"])
            
            # Check landmark detection
            assert result.landmark is not None
            assert case["expected_landmark"] in result.landmark.name.lower()
            
            # Check spatial relation
            assert result.relation is not None
            assert result.relation.relation == case["expected_relation"]
            
            # Check overall confidence
            assert result.confidence > 0.3
    
    def test_landmark_only_parsing(self):
        """Test parsing explanations with only landmarks."""
        text = "Migros market"
        result = self.parser.parse(text)
        
        assert result.landmark is not None
        assert "migros" in result.landmark.name.lower()
        # May or may not have spatial relation
    
    def test_relation_only_parsing(self):
        """Test parsing explanations with only spatial relations."""
        text = "karşısındaki bina"
        result = self.parser.parse(text)
        
        assert result.relation is not None
        assert result.relation.relation == "karşısı"
        # May or may not have landmark
    
    def test_empty_input(self):
        """Test parsing empty or whitespace input."""
        test_cases = ["", "   ", "\n\t"]
        
        for empty_text in test_cases:
            result = self.parser.parse(empty_text)
            assert result.confidence == 0.0
            assert "Empty" in " ".join(result.processing_notes)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # High confidence case
        result1 = self.parser.parse("Migros yanı")
        
        # Lower confidence case  
        result2 = self.parser.parse("bir yer civarında")
        
        assert result1.confidence > result2.confidence
    
    def test_batch_parsing(self):
        """Test batch parsing functionality."""
        texts = [
            "Migros yanı",
            "Hotel karşısı", 
            "invalid nonsense text xyz"
        ]
        
        results = self.parser.parse_batch(texts)
        assert len(results) == 3
        
        # First two should parse reasonably well
        assert results[0].confidence > 0.3
        assert results[1].confidence > 0.3


class TestConvenienceFunctions:
    """Test convenience functions for quick usage."""
    
    def test_parse_explanation_function(self):
        """Test the parse_explanation convenience function."""
        result = parse_explanation("Migros yanı")
        
        assert isinstance(result, dict)
        assert result["type"] == "landmark"
        assert "landmark_name" in result
        assert "spatial_relation" in result
        assert result["confidence"] > 0.3
    
    def test_extract_landmark_info_function(self):
        """Test the extract_landmark_info convenience function."""
        # Valid landmark
        info = extract_landmark_info("Migros market")
        assert info is not None
        assert "migros" in info["name"].lower()
        assert info["type"] == "market"
        
        # Invalid/unclear text
        info = extract_landmark_info("random text 123")
        assert info is None or len(info) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.parser = ExplanationParser()
    
    def test_very_long_text(self):
        """Test parsing very long explanations."""
        long_text = "Bu çok uzun bir açıklama metni " * 20 + " Migros yanı"
        result = self.parser.parse(long_text)
        
        # Should still detect the landmark
        assert result.landmark is not None
        assert "migros" in result.landmark.name.lower()
    
    def test_turkish_characters(self):
        """Test handling of Turkish characters."""
        test_cases = [
            "Öğretmenler Sitesi yanı",
            "Ülker Çikolata fabrikası karşısı",
            "İş Bankası şubesi önü"
        ]
        
        for text in test_cases:
            result = self.parser.parse(text)
            # Should parse without errors
            assert result.confidence >= 0.0
    
    def test_mixed_case_input(self):
        """Test handling of mixed case input."""
        test_cases = [
            "MİGROS YANI",
            "migros yanı", 
            "MiGrOs YaNı"
        ]
        
        results = [self.parser.parse(text) for text in test_cases]
        
        # All should detect the same landmark
        for result in results:
            assert result.landmark is not None
            assert "migros" in result.landmark.name.lower()
    
    def test_numbers_and_punctuation(self):
        """Test handling of numbers and punctuation."""
        text = "Migros-123 yanı!!! (açıklama)"
        result = self.parser.parse(text)
        
        # Should still detect landmark and relation
        assert result.landmark is not None
        assert result.relation is not None


class TestConfiguration:
    """Test configuration and customization."""
    
    def test_custom_config(self):
        """Test parser with custom configuration."""
        config = ExplanationConfig(
            min_confidence_threshold=0.8,
            debug_mode=True
        )
        parser = ExplanationParser(config)
        
        # With high threshold, some results might be rejected
        result = parser.parse("belirsiz bir yer yanı")
        valid = parser.is_valid_result(result)
        
        # Check that threshold is being applied
        assert valid == (result.confidence >= 0.8)
    
    def test_debug_info(self):
        """Test debug information generation."""
        parser = ExplanationParser()
        debug_info = parser.get_debug_info("Migros yanı")
        
        assert "original_text" in debug_info
        assert "preprocessed_text" in debug_info
        assert "detected_landmarks" in debug_info
        assert "detected_relations" in debug_info
        assert len(debug_info["detected_landmarks"]) > 0
        assert len(debug_info["detected_relations"]) > 0


# Performance tests (can be run separately)
class TestPerformance:
    """Performance tests for the explanation parser."""
    
    @pytest.mark.performance
    def test_parsing_speed(self):
        """Test parsing speed for typical usage."""
        import time
        
        parser = ExplanationParser()
        test_texts = [
            "Migros yanı",
            "Amorium Hotel karşısı",
            "Şekerbank şubesi önünde"
        ] * 100  # 300 total
        
        start_time = time.time()
        results = parser.parse_batch(test_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(test_texts)
        
        print(f"Processed {len(test_texts)} texts in {total_time:.2f}s")
        print(f"Average time per text: {avg_time*1000:.2f}ms")
        
        # Should be reasonably fast (under 10ms per text)
        assert avg_time < 0.01


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
