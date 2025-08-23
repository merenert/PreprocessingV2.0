"""
Adaptive Threshold Management Models

Pydantic models for adaptive threshold management system.
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class PatternType(str, Enum):
    """Pattern türleri"""
    STREET = "street"
    DISTRICT = "district"
    CITY = "city"
    BUILDING = "building"
    NUMBER = "number"
    POSTAL = "postal"
    EXPLANATION = "explanation"


class PerformanceStats(BaseModel):
    """
    Pattern performans istatistikleri
    
    Attributes:
        success_count: Başarılı işlem sayısı
        total_count: Toplam işlem sayısı
        success_rate: Başarı oranı (0.0-1.0)
        avg_confidence: Ortalama güven skoru
        usage_count: Toplam kullanım sayısı
        last_updated: Son güncelleme zamanı
        confidence_sum: Güven skorları toplamı (ortalama hesabı için)
    """
    success_count: int = Field(default=0, ge=0)
    total_count: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence_sum: float = Field(default=0.0, ge=0.0)
    
    @validator('success_rate', always=True)
    def calculate_success_rate(cls, v, values):
        """Başarı oranını hesapla"""
        total = values.get('total_count', 0)
        success = values.get('success_count', 0)
        if total > 0:
            return success / total
        return 0.0
    
    @validator('avg_confidence', always=True) 
    def calculate_avg_confidence(cls, v, values):
        """Ortalama güven skorunu hesapla"""
        total = values.get('total_count', 0)
        conf_sum = values.get('confidence_sum', 0.0)
        if total > 0:
            return conf_sum / total
        return 0.0
    
    def update_with_result(self, success: bool, confidence: float) -> 'PerformanceStats':
        """
        Yeni sonuçla istatistikleri güncelle
        
        Args:
            success: İşlem başarılı mı
            confidence: Güven skoru
            
        Returns:
            Güncellenmiş PerformanceStats
        """
        new_total = self.total_count + 1
        new_success = self.success_count + (1 if success else 0)
        new_conf_sum = self.confidence_sum + confidence
        new_usage = self.usage_count + 1
        
        return PerformanceStats(
            success_count=new_success,
            total_count=new_total,
            confidence_sum=new_conf_sum,
            usage_count=new_usage,
            last_updated=datetime.now()
        )


class ThresholdConfig(BaseModel):
    """
    Threshold konfigürasyonu
    
    Attributes:
        min_threshold: Minimum threshold değeri
        max_threshold: Maximum threshold değeri  
        default_threshold: Varsayılan threshold değeri
        adjustment_step: Ayarlama adım büyüklüğü
        min_samples: Minimum örnek sayısı (threshold ayarlamak için)
        high_performance_threshold: Yüksek performans eşiği
        low_performance_threshold: Düşük performans eşiği
    """
    min_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    default_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    adjustment_step: float = Field(default=0.05, ge=0.01, le=0.2)
    min_samples: int = Field(default=10, ge=1)
    high_performance_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    low_performance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @validator('max_threshold')
    def max_greater_than_min(cls, v, values):
        """Max threshold min'den büyük olmalı"""
        min_val = values.get('min_threshold', 0.6)
        if v <= min_val:
            raise ValueError(f"max_threshold ({v}) must be greater than min_threshold ({min_val})")
        return v
    
    @validator('default_threshold')
    def default_in_range(cls, v, values):
        """Default threshold aralıkta olmalı"""
        min_val = values.get('min_threshold', 0.6)
        max_val = values.get('max_threshold', 0.95)
        if not (min_val <= v <= max_val):
            raise ValueError(f"default_threshold ({v}) must be between {min_val} and {max_val}")
        return v


class PatternResult(BaseModel):
    """
    Pattern sonucu
    
    Attributes:
        pattern_name: Pattern adı
        pattern_type: Pattern türü
        success: İşlem başarılı mı
        confidence: Güven skoru
        threshold_used: Kullanılan threshold değeri
        timestamp: İşlem zamanı
        metadata: Ek bilgiler
    """
    pattern_name: str
    pattern_type: PatternType
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class AdaptiveState(BaseModel):
    """
    Adaptive threshold manager durumu
    
    Attributes:
        pattern_stats: Pattern istatistikleri
        current_thresholds: Mevcut threshold değerleri
        config: Threshold konfigürasyonu
        last_save: Son kaydetme zamanı
        total_updates: Toplam güncelleme sayısı
    """
    pattern_stats: Dict[str, PerformanceStats] = Field(default_factory=dict)
    current_thresholds: Dict[str, float] = Field(default_factory=dict)
    config: ThresholdConfig = Field(default_factory=ThresholdConfig)
    last_save: datetime = Field(default_factory=datetime.now)
    total_updates: int = Field(default=0, ge=0)
    
    def get_pattern_threshold(self, pattern_name: str) -> float:
        """Pattern için threshold al"""
        return self.current_thresholds.get(pattern_name, self.config.default_threshold)
    
    def set_pattern_threshold(self, pattern_name: str, threshold: float) -> None:
        """Pattern threshold'unu ayarla"""
        # Threshold'u aralığa sınırla
        threshold = max(self.config.min_threshold, 
                       min(self.config.max_threshold, threshold))
        self.current_thresholds[pattern_name] = threshold
    
    def get_pattern_stats(self, pattern_name: str) -> PerformanceStats:
        """Pattern istatistiklerini al"""
        return self.pattern_stats.get(pattern_name, PerformanceStats())
    
    def update_pattern_stats(self, pattern_name: str, stats: PerformanceStats) -> None:
        """Pattern istatistiklerini güncelle"""
        self.pattern_stats[pattern_name] = stats
        self.total_updates += 1
        self.last_save = datetime.now()


class LearningMetrics(BaseModel):
    """
    Öğrenme metrikleri
    
    Attributes:
        patterns_tracked: İzlenen pattern sayısı
        total_adjustments: Toplam ayarlama sayısı
        improvement_rate: İyileşme oranı
        stability_score: Kararlılık skoru
        last_calculated: Son hesaplama zamanı
    """
    patterns_tracked: int = Field(default=0, ge=0)
    total_adjustments: int = Field(default=0, ge=0)
    improvement_rate: float = Field(default=0.0, ge=0.0)
    stability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_calculated: datetime = Field(default_factory=datetime.now)
