"""
Pydantic models for explanation/landmark processing.
Defines data structures for parsed landmark information.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class SpatialRelation(BaseModel):
    """Spatial relationship between address and landmark."""
    
    relation: str = Field(..., description="Spatial relation keyword")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    
    @validator('relation')
    def validate_relation(cls, v):
        """Validate spatial relation keywords."""
        valid_relations = {
            'karşısı', 'yanı', 'arkası', 'üstü', 'altı', 'önü', 'bitişiği',
            'yakını', 'çevresinde', 'içinde', 'dışında', 'arasında'
        }
        if v.lower() not in valid_relations:
            raise ValueError(f"Invalid spatial relation: {v}")
        return v.lower()


class Landmark(BaseModel):
    """Detected landmark information."""
    
    name: str = Field(..., description="Landmark name")
    type: str = Field(..., description="Landmark type/category")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    raw_text: Optional[str] = Field(None, description="Original text segment")
    
    @validator('type')
    def validate_type(cls, v):
        """Validate landmark types."""
        valid_types = {
            'hotel', 'otel', 'market', 'mağaza', 'hastane', 'okul', 'cami', 
            'park', 'banka', 'restoran', 'kafe', 'eczane', 'berber', 'kuaför',
            'benzinlik', 'garaj', 'otopark', 'terminal', 'istasyon', 'durak',
            'fabrika', 'atölye', 'dükkan', 'büro', 'şirket', 'firma', 'limited',
            'anonim', 'kollektif', 'kooperatif', 'dernek', 'vakıf', 'organizasyon'
        }
        if v.lower() not in valid_types:
            # Allow unknown types but mark with low confidence
            return v.lower()
        return v.lower()


class ExplanationResult(BaseModel):
    """Complete explanation parsing result."""
    
    type: str = Field(default="landmark", description="Result type")
    landmark: Optional[Landmark] = Field(None, description="Detected landmark")
    relation: Optional[SpatialRelation] = Field(None, description="Spatial relation")
    raw_explanation: str = Field(..., description="Original explanation text")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence")
    processing_notes: List[str] = Field(default_factory=list, description="Processing notes")
    
    def to_json_output(self) -> Dict[str, Any]:
        """Convert to simple JSON output format."""
        result = {
            "type": self.type,
            "raw_explanation": self.raw_explanation,
            "confidence": self.confidence
        }
        
        if self.landmark:
            result.update({
                "landmark_name": self.landmark.name,
                "landmark_type": self.landmark.type,
                "landmark_confidence": self.landmark.confidence
            })
        
        if self.relation:
            result.update({
                "spatial_relation": self.relation.relation,
                "relation_confidence": self.relation.confidence
            })
        
        if self.processing_notes:
            result["notes"] = self.processing_notes
            
        return result


class ExplanationConfig(BaseModel):
    """Configuration for explanation parsing."""
    
    min_confidence_threshold: float = Field(default=0.3, description="Minimum confidence for results")
    landmark_detection_threshold: float = Field(default=0.5, description="Landmark detection threshold")
    relation_detection_threshold: float = Field(default=0.6, description="Relation detection threshold")
    enable_fuzzy_matching: bool = Field(default=True, description="Enable fuzzy string matching")
    max_tokens_per_landmark: int = Field(default=5, description="Max tokens in landmark name")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
