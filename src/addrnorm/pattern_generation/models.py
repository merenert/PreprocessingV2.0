"""
ML Pattern Generation Models

Pydantic models for ML-based pattern generation system.
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np


class PatternType(str, Enum):
    """Pattern türleri"""

    REGEX = "regex"
    TEMPLATE = "template"
    SEQUENCE = "sequence"
    COMPOSITE = "composite"


class ClusteringMethod(str, Enum):
    """Clustering metotları"""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


# Alias for backward compatibility
ClusteringAlgorithm = ClusteringMethod


class ValidationStatus(str, Enum):
    """Validation durumları"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MODIFICATION = "needs_modification"


class ConflictSeverity(str, Enum):
    """Conflict şiddet seviyeleri"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictType(str, Enum):
    """Conflict türleri"""

    REGEX_OVERLAP = "regex_overlap"
    FIELD_MAPPING = "field_mapping"
    PATTERN_REDUNDANCY = "pattern_redundancy"
    PERFORMANCE = "performance"
    SEMANTIC_OVERLAP = "semantic_overlap"


class ResolutionStrategy(str, Enum):
    """Conflict çözüm stratejileri"""

    MERGE_PATTERNS = "merge_patterns"
    ADD_SPECIFICITY = "add_specificity"
    PRIORITY_BASED = "priority_based"
    RENAME_FIELDS = "rename_fields"
    REPLACE_PATTERN = "replace_pattern"
    KEEP_EXISTING = "keep_existing"
    OPTIMIZE_PATTERN = "optimize_pattern"
    ADD_CONSTRAINTS = "add_constraints"
    UPDATE_METADATA = "update_metadata"
    MANUAL_REVIEW = "manual_review"


class ReviewStatus(str, Enum):
    """Review durumları"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CRITICAL = "critical"


class AddressComponent(BaseModel):
    """Adres bileşeni"""

    component: str = Field(..., description="Bileşen adı (il, ilce, mahalle, etc.)")
    value: str = Field(..., description="Bileşen değeri")
    confidence: float = Field(ge=0.0, le=1.0, description="Güven skoru")
    position: int = Field(ge=0, description="Metindeki pozisyon")


class ClusterResult(BaseModel):
    """Clustering sonucu"""

    cluster_id: int = Field(..., description="Cluster ID")
    addresses: List[str] = Field(..., description="Cluster'daki adresler")
    centroid: Optional[str] = Field(None, description="Cluster merkezi")
    size: int = Field(..., ge=1, description="Cluster boyutu")
    similarity_score: float = Field(ge=0.0, le=1.0, description="İç benzerlik skoru")
    representative_pattern: Optional[str] = Field(None, description="Temsili pattern")

    @validator("size", always=True)
    def validate_size(cls, v, values):
        """Size addresses ile tutarlı olmalı"""
        addresses = values.get("addresses", [])
        if len(addresses) != v:
            raise ValueError(f"Size ({v}) must match addresses length ({len(addresses)})")
        return v


class PatternTemplate(BaseModel):
    """Pattern şablonu"""

    template: str = Field(..., description="Pattern şablonu")
    components: List[str] = Field(..., description="Bileşen listesi")
    example: str = Field(..., description="Örnek kullanım")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Karmaşıklık skoru")
    generalizability: float = Field(ge=0.0, le=1.0, description="Genelleştirilebilirlik")


class PatternSuggestion(BaseModel):
    """Pattern önerisi"""

    pattern_id: str = Field(..., description="Unique pattern ID")
    pattern_type: PatternType = Field(..., description="Pattern türü")
    regex_pattern: str = Field(..., description="Regex pattern")
    template: PatternTemplate = Field(..., description="Pattern şablonu")
    source_cluster: ClusterResult = Field(..., description="Kaynak cluster")
    confidence: float = Field(ge=0.0, le=1.0, description="Öneri güven skoru")
    coverage: float = Field(ge=0.0, le=1.0, description="Kapsam oranı")
    examples: List[str] = Field(..., description="Örnek adresler")
    extracted_components: List[AddressComponent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def quality_score(self) -> float:
        """Genel kalite skoru"""
        return self.confidence * 0.4 + self.coverage * 0.3 + self.template.generalizability * 0.3


class ValidationResult(BaseModel):
    """Pattern validation sonucu"""

    pattern_id: str = Field(..., description="Pattern ID")
    is_valid: bool = Field(..., description="Pattern geçerli mi")
    quality_score: float = Field(ge=0.0, le=1.0, description="Kalite skoru")
    complexity_analysis: Dict[str, float] = Field(default_factory=dict)
    generalizability_score: float = Field(ge=0.0, le=1.0)
    collision_risk: float = Field(ge=0.0, le=1.0, description="Çakışma riski")
    coverage_analysis: Dict[str, Any] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list, description="Tespit edilen sorunlar")
    recommendations: List[str] = Field(default_factory=list, description="İyileştirme önerileri")
    validated_at: datetime = Field(default_factory=datetime.now)


class ConflictReport(BaseModel):
    """Pattern conflict raporu"""

    pattern_id: str = Field(..., description="Pattern ID")
    conflicting_patterns: List[str] = Field(default_factory=list)
    severity: ConflictSeverity = Field(default=ConflictSeverity.LOW)
    overlap_score: float = Field(ge=0.0, le=1.0, description="Örtüşme skoru")
    ambiguity_examples: List[str] = Field(default_factory=list)
    resolution_suggestions: List[str] = Field(default_factory=list)
    priority_recommendation: str = Field(..., description="Öncelik önerisi")
    analysis_details: Dict[str, Any] = Field(default_factory=dict)


class ReviewDecision(BaseModel):
    """Human review kararı"""

    pattern_id: str = Field(..., description="Pattern ID")
    decision: ValidationStatus = Field(..., description="Review kararı")
    reviewer: str = Field(..., description="Reviewer adı")
    comments: str = Field(default="", description="Review yorumları")
    modifications: Dict[str, str] = Field(default_factory=dict, description="Önerilen değişiklikler")
    priority: int = Field(ge=1, le=5, default=3, description="Öncelik (1=düşük, 5=yüksek)")
    reviewed_at: datetime = Field(default_factory=datetime.now)


class MLPatternConfig(BaseModel):
    """ML Pattern sistem konfigürasyonu"""

    clustering_method: ClusteringMethod = Field(default=ClusteringMethod.KMEANS)
    n_clusters: int = Field(default=5, ge=2)  # Eklendi
    min_cluster_size: int = Field(default=3, ge=2)  # 5'ten 3'e düşürüldü
    max_clusters: int = Field(default=50, ge=2)
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)  # Eklendi
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)  # Eklendi
    min_pattern_confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # 0.7'den 0.5'e düşürüldü
    min_coverage: float = Field(default=0.1, ge=0.0, le=1.0)
    max_pattern_complexity: float = Field(default=0.8, ge=0.0, le=1.0)

    # Vectorization settings
    use_tfidf: bool = Field(default=True)
    use_char_ngrams: bool = Field(default=True)
    ngram_range: Tuple[int, int] = Field(default=(2, 4))
    max_features: int = Field(default=10000, ge=100)

    # Quality thresholds
    min_generalizability: float = Field(default=0.6, ge=0.0, le=1.0)
    max_collision_risk: float = Field(default=0.3, ge=0.0, le=1.0)

    # Review settings
    auto_approve_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    auto_reject_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class PatternPerformanceMetrics(BaseModel):
    """Pattern performans metrikleri"""

    pattern_id: str = Field(..., description="Pattern ID")
    extraction_accuracy: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0)
    false_negative_rate: float = Field(ge=0.0, le=1.0)
    processing_speed: float = Field(ge=0.0, description="ms per address")
    memory_usage: float = Field(ge=0.0, description="MB")

    # Component-specific accuracy
    component_accuracy: Dict[str, float] = Field(default_factory=dict)

    # Coverage metrics
    addresses_matched: int = Field(ge=0)
    total_addresses_tested: int = Field(ge=0)
    coverage_percentage: float = Field(ge=0.0, le=100.0)

    @validator("coverage_percentage", always=True)
    def calculate_coverage(cls, v, values):
        """Coverage yüzdesini hesapla"""
        matched = values.get("addresses_matched", 0)
        total = values.get("total_addresses_tested", 0)
        if total > 0:
            return (matched / total) * 100
        return 0.0


class MLModelInfo(BaseModel):
    """ML model bilgileri"""

    model_type: str = Field(..., description="Model türü")
    model_version: str = Field(..., description="Model versiyonu")
    training_date: datetime = Field(..., description="Eğitim tarihi")
    training_size: int = Field(ge=0, description="Eğitim veri boyutu")
    validation_accuracy: float = Field(ge=0.0, le=1.0)
    feature_count: int = Field(ge=0, description="Feature sayısı")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    # Performance metrics
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)

    @property
    def model_quality(self) -> str:
        """Model kalite değerlendirmesi"""
        if self.f1_score >= 0.9:
            return "excellent"
        elif self.f1_score >= 0.8:
            return "good"
        elif self.f1_score >= 0.7:
            return "acceptable"
        else:
            return "poor"


class SystemState(BaseModel):
    """ML Pattern sistem durumu"""

    total_patterns_generated: int = Field(default=0, ge=0)
    patterns_approved: int = Field(default=0, ge=0)
    patterns_rejected: int = Field(default=0, ge=0)
    patterns_in_review: int = Field(default=0, ge=0)

    active_clusters: int = Field(default=0, ge=0)
    last_clustering_date: Optional[datetime] = None
    last_model_update: Optional[datetime] = None

    # Performance stats
    avg_pattern_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_review_time_hours: float = Field(default=0.0, ge=0.0)
    system_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    config: MLPatternConfig = Field(default_factory=MLPatternConfig)

    @property
    def approval_rate(self) -> float:
        """Onay oranı"""
        total_reviewed = self.patterns_approved + self.patterns_rejected
        if total_reviewed > 0:
            return self.patterns_approved / total_reviewed
        return 0.0


class ConflictResolution(BaseModel):
    """Conflict çözüm önerisi"""

    strategy: ResolutionStrategy
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    implementation_notes: str = ""
    impact_assessment: str = ""


class PatternConflict(BaseModel):
    """Pattern çakışması"""

    conflict_id: str
    existing_pattern_id: str
    new_pattern_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    affected_samples: List[str] = Field(default_factory=list)
    resolution_suggestions: List[ConflictResolution] = Field(default_factory=list)
    conflict_details: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.now)


class PatternReview(BaseModel):
    """Pattern inceleme kaydı"""

    review_id: str
    pattern_id: str
    reviewer_id: str
    status: ReviewStatus
    decision: str  # approve, reject, modify
    notes: str = ""
    modifications: Dict[str, Any] = Field(default_factory=dict)
    reviewed_at: datetime = Field(default_factory=datetime.now)


class AddressCluster(BaseModel):
    """Adres kümesi"""

    cluster_id: str
    addresses: List[str]
    common_features: Dict[str, Any] = Field(default_factory=dict)
    cluster_size: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    representative_address: str = ""


class PatternGenerationConfig(BaseModel):
    """Pattern oluşturma konfigürasyonu"""

    min_pattern_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    max_clusters: int = Field(default=10, ge=1)
    min_cluster_size: int = Field(default=3, ge=1)
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    enable_conflict_detection: bool = True
    require_human_review: bool = True
    quality_weights: Dict[str, float] = Field(
        default_factory=lambda: {"accuracy": 0.3, "coverage": 0.2, "complexity": 0.2, "performance": 0.1, "completeness": 0.2}
    )


@dataclass
class PatternPerformance:
    """Pattern performance metrics"""

    pattern_id: str
    success_rate: float
    average_processing_time: float
    memory_usage: int
    coverage: float
    false_positive_rate: float
    last_updated: datetime
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    strategy: ResolutionStrategy
    confidence: float
    explanation: str
    auto_applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternQuality:
    """Pattern kalite metrikleri"""

    pattern_id: str
    accuracy: float
    coverage: float
    complexity: float
    performance_score: float
    overall_score: float
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
