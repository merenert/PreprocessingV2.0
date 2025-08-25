"""
Enhanced data models for advanced adaptive learning system.

Updated models with comprehensive features for automatic threshold optimization,
pattern performance tracking, and intelligent learning algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class OptimizationStrategy(Enum):
    """Threshold optimization strategies"""

    CONSERVATIVE = "conservative"  # Slow, safe adjustments
    AGGRESSIVE = "aggressive"  # Fast adjustments
    BALANCED = "balanced"  # Default balanced approach
    VOLUME_WEIGHTED = "volume_weighted"  # Based on processing volume


class PerformanceTrend(Enum):
    """Performance trend indicators"""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for a pattern"""

    pattern_id: str
    success_rate: float
    total_processed: int
    successful_matches: int
    failed_matches: int
    average_confidence: float
    confidence_variance: float
    processing_time_avg: float
    last_updated: datetime = field(default_factory=datetime.now)

    def update_metrics(self, success: bool, confidence: float, processing_time: float):
        """Update metrics with new data point"""
        self.total_processed += 1

        if success:
            self.successful_matches += 1
        else:
            self.failed_matches += 1

        # Update success rate
        self.success_rate = self.successful_matches / self.total_processed

        # Update average confidence (weighted moving average)
        alpha = 0.1  # Learning rate
        self.average_confidence = (1 - alpha) * self.average_confidence + alpha * confidence

        # Update processing time
        self.processing_time_avg = (1 - alpha) * self.processing_time_avg + alpha * processing_time

        self.last_updated = datetime.now()


@dataclass
class PatternPerformance:
    """Historical performance data for a pattern"""

    pattern_id: str
    pattern_type: str  # "address", "street", "district", etc.
    current_threshold: float
    optimal_threshold: float
    metrics: PerformanceMetrics
    historical_data: List[Dict] = field(default_factory=list)
    trend: PerformanceTrend = PerformanceTrend.STABLE
    last_optimization: Optional[datetime] = None
    optimization_count: int = 0

    def add_historical_point(self, success_rate: float, threshold: float, volume: int, timestamp: datetime = None):
        """Add historical performance data point"""
        if timestamp is None:
            timestamp = datetime.now()

        self.historical_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "success_rate": success_rate,
                "threshold": threshold,
                "volume": volume,
                "confidence_avg": self.metrics.average_confidence,
            }
        )

        # Keep only last 1000 points
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]

    def calculate_trend(self, lookback_days: int = 7) -> PerformanceTrend:
        """Calculate performance trend over lookback period"""
        if len(self.historical_data) < 2:
            return PerformanceTrend.STABLE

        cutoff_time = datetime.now().timestamp() - (lookback_days * 24 * 3600)
        recent_data = [
            point for point in self.historical_data if datetime.fromisoformat(point["timestamp"]).timestamp() > cutoff_time
        ]

        if len(recent_data) < 2:
            return PerformanceTrend.STABLE

        # Calculate trend using linear regression
        success_rates = [point["success_rate"] for point in recent_data]
        n = len(success_rates)

        if n < 3:
            return PerformanceTrend.STABLE

        # Simple trend calculation
        first_half = sum(success_rates[: n // 2]) / (n // 2)
        second_half = sum(success_rates[n // 2 :]) / (n - n // 2)

        diff = second_half - first_half
        variance = sum((x - sum(success_rates) / n) ** 2 for x in success_rates) / n

        if variance > 0.01:  # High variance
            self.trend = PerformanceTrend.VOLATILE
        elif diff > 0.05:  # Significant improvement
            self.trend = PerformanceTrend.IMPROVING
        elif diff < -0.05:  # Significant decline
            self.trend = PerformanceTrend.DECLINING
        else:
            self.trend = PerformanceTrend.STABLE

        return self.trend


@dataclass
class ThresholdUpdate:
    """Record of a threshold update"""

    pattern_id: str
    old_threshold: float
    new_threshold: float
    reason: str
    performance_before: float
    expected_improvement: float
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_used: OptimizationStrategy = OptimizationStrategy.BALANCED

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "pattern_id": self.pattern_id,
            "old_threshold": self.old_threshold,
            "new_threshold": self.new_threshold,
            "reason": self.reason,
            "performance_before": self.performance_before,
            "expected_improvement": self.expected_improvement,
            "timestamp": self.timestamp.isoformat(),
            "strategy_used": self.strategy_used.value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ThresholdUpdate":
        """Create from dictionary"""
        return cls(
            pattern_id=data["pattern_id"],
            old_threshold=data["old_threshold"],
            new_threshold=data["new_threshold"],
            reason=data["reason"],
            performance_before=data["performance_before"],
            expected_improvement=data["expected_improvement"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            strategy_used=OptimizationStrategy(data["strategy_used"]),
        )


@dataclass
class LearningConfig:
    """Configuration for adaptive learning"""

    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    min_samples_for_optimization: int = 100
    optimization_frequency_hours: int = 24
    max_threshold_change: float = 0.2
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    success_rate_target: float = 0.85
    confidence_weight: float = 0.3
    volume_weight: float = 0.4
    trend_weight: float = 0.3
    volatility_penalty: float = 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "optimization_strategy": self.optimization_strategy.value,
            "min_samples_for_optimization": self.min_samples_for_optimization,
            "optimization_frequency_hours": self.optimization_frequency_hours,
            "max_threshold_change": self.max_threshold_change,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "success_rate_target": self.success_rate_target,
            "confidence_weight": self.confidence_weight,
            "volume_weight": self.volume_weight,
            "trend_weight": self.trend_weight,
            "volatility_penalty": self.volatility_penalty,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearningConfig":
        """Create from dictionary"""
        return cls(
            optimization_strategy=OptimizationStrategy(data["optimization_strategy"]),
            min_samples_for_optimization=data["min_samples_for_optimization"],
            optimization_frequency_hours=data["optimization_frequency_hours"],
            max_threshold_change=data["max_threshold_change"],
            min_threshold=data["min_threshold"],
            max_threshold=data["max_threshold"],
            success_rate_target=data["success_rate_target"],
            confidence_weight=data["confidence_weight"],
            volume_weight=data["volume_weight"],
            trend_weight=data["trend_weight"],
            volatility_penalty=data["volatility_penalty"],
        )


@dataclass
class OptimizationResult:
    """Result of threshold optimization"""

    pattern_id: str
    old_threshold: float
    new_threshold: float
    expected_improvement: float
    confidence_score: float
    optimization_applied: bool
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "old_threshold": self.old_threshold,
            "new_threshold": self.new_threshold,
            "expected_improvement": self.expected_improvement,
            "confidence_score": self.confidence_score,
            "optimization_applied": self.optimization_applied,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LearningState:
    """Current state of the learning system"""

    total_patterns_tracked: int = 0
    total_optimizations: int = 0
    average_improvement: float = 0.0
    last_optimization_run: Optional[datetime] = None
    active_patterns: int = 0
    system_performance: float = 0.0

    def update_stats(self, optimization_results: List[OptimizationResult]):
        """Update learning system statistics"""
        if optimization_results:
            self.total_optimizations += len(optimization_results)
            improvements = [r.expected_improvement for r in optimization_results if r.optimization_applied]
            if improvements:
                self.average_improvement = sum(improvements) / len(improvements)
            self.last_optimization_run = datetime.now()


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

    @validator("success_rate", always=True)
    def calculate_success_rate(cls, v, values):
        """Başarı oranını hesapla"""
        total = values.get("total_count", 0)
        success = values.get("success_count", 0)
        if total > 0:
            return success / total
        return 0.0

    @validator("avg_confidence", always=True)
    def calculate_avg_confidence(cls, v, values):
        """Ortalama güven skorunu hesapla"""
        total = values.get("total_count", 0)
        conf_sum = values.get("confidence_sum", 0.0)
        if total > 0:
            return conf_sum / total
        return 0.0

    def update_with_result(self, success: bool, confidence: float) -> "PerformanceStats":
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
            last_updated=datetime.now(),
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

    @validator("max_threshold")
    def max_greater_than_min(cls, v, values):
        """Max threshold min'den büyük olmalı"""
        min_val = values.get("min_threshold", 0.6)
        if v <= min_val:
            raise ValueError(f"max_threshold ({v}) must be greater than min_threshold ({min_val})")
        return v

    @validator("default_threshold")
    def default_in_range(cls, v, values):
        """Default threshold aralıkta olmalı"""
        min_val = values.get("min_threshold", 0.6)
        max_val = values.get("max_threshold", 0.95)
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
        threshold = max(self.config.min_threshold, min(self.config.max_threshold, threshold))
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
