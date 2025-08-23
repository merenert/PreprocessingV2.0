"""
Adaptive Learning Engine

Pattern performansına göre threshold'ları otomatik ayarlayan öğrenme motoru.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from .models import PerformanceStats, ThresholdConfig, LearningMetrics
from .pattern_tracker import PatternTracker


class LearningEngine:
    """
    Adaptive öğrenme motoru
    
    Bu sınıf pattern performanslarını analiz ederek threshold değerlerini
    otomatik olarak ayarlar.
    """
    
    def __init__(self, 
                 config: ThresholdConfig,
                 tracker: PatternTracker):
        """
        LearningEngine başlatıcı
        
        Args:
            config: Threshold konfigürasyonu
            tracker: Pattern tracker instance
        """
        self.config = config
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        
        # Öğrenme metrikleri
        self.metrics = LearningMetrics()
        
        # Ayarlama geçmişi
        self._adjustment_history: Dict[str, List[Tuple[datetime, float, str]]] = {}
    
    def calculate_optimal_threshold(self, pattern_name: str) -> Tuple[float, str]:
        """
        Pattern için optimal threshold hesapla
        
        Args:
            pattern_name: Pattern adı
            
        Returns:
            Tuple[float, str]: (yeni_threshold, açıklama)
        """
        stats = self.tracker.get_pattern_stats(pattern_name)
        
        # Yeterli veri yoksa varsayılan değer
        if stats.total_count < self.config.min_samples:
            return self.config.default_threshold, "Yetersiz veri - varsayılan threshold"
        
        current_threshold = self._get_current_threshold(pattern_name)
        success_rate = stats.success_rate
        avg_confidence = stats.avg_confidence
        
        # Kararlılık skorunu al
        stability = self.tracker.calculate_stability_score(pattern_name)
        
        # Son performansı kontrol et
        recent_success, recent_confidence = self.tracker.get_recent_performance(
            pattern_name, window_size=20
        )
        
        # Threshold ayarlama mantığı
        if success_rate > self.config.high_performance_threshold:
            # Yüksek performans - threshold'u düşür
            adjustment = -self.config.adjustment_step
            
            # Kararlılık yüksekse daha agresif ayarlama
            if stability > 0.8:
                adjustment *= 1.5
                
            reason = f"Yüksek performans (%.2f%%) - threshold düşürüldü" % (success_rate * 100)
            
        elif success_rate < self.config.low_performance_threshold:
            # Düşük performans - threshold'u yükselt
            adjustment = self.config.adjustment_step
            
            # Son performans daha da kötüyse daha agresif ayarlama
            if recent_success < success_rate * 0.8:
                adjustment *= 1.5
                
            reason = f"Düşük performans (%.2f%%) - threshold yükseltildi" % (success_rate * 100)
            
        else:
            # Orta performans - ince ayar
            if avg_confidence > 0.9 and stability > 0.7:
                # Güven yüksek ve kararlı - threshold'u biraz düşür
                adjustment = -self.config.adjustment_step * 0.5
                reason = f"Yüksek güven (%.2f) - threshold ince ayar" % avg_confidence
            elif avg_confidence < 0.6:
                # Güven düşük - threshold'u biraz yükselt
                adjustment = self.config.adjustment_step * 0.5
                reason = f"Düşük güven (%.2f) - threshold ince ayar" % avg_confidence
            else:
                # Değişiklik yok
                adjustment = 0.0
                reason = "Performans kabul edilebilir - değişiklik yok"
        
        # Kullanım sıklığını hesaba kat
        usage_factor = self._calculate_usage_factor(stats.usage_count)
        adjustment *= usage_factor
        
        # Yeni threshold hesapla
        new_threshold = current_threshold + adjustment
        
        # Sınırları kontrol et
        new_threshold = max(self.config.min_threshold,
                           min(self.config.max_threshold, new_threshold))
        
        # Minimum değişim kontrolü (çok küçük değişiklikleri engelle)
        if abs(new_threshold - current_threshold) < 0.01:
            new_threshold = current_threshold
            reason += " (minimum değişim eşiği)"
        
        # Ayarlama geçmişine ekle
        self._record_adjustment(pattern_name, new_threshold, reason)
        
        return new_threshold, reason
    
    def batch_optimize_thresholds(self, 
                                 pattern_names: Optional[List[str]] = None) -> Dict[str, Tuple[float, str]]:
        """
        Toplu threshold optimizasyonu
        
        Args:
            pattern_names: Optimize edilecek pattern'lar (None ise tümü)
            
        Returns:
            Dict[str, Tuple[float, str]]: Pattern → (threshold, açıklama) mapping
        """
        if pattern_names is None:
            pattern_names = self.tracker.get_all_patterns()
        
        results = {}
        adjustments_made = 0
        
        for pattern_name in pattern_names:
            old_threshold = self._get_current_threshold(pattern_name)
            new_threshold, reason = self.calculate_optimal_threshold(pattern_name)
            
            results[pattern_name] = (new_threshold, reason)
            
            if abs(new_threshold - old_threshold) > 0.01:
                adjustments_made += 1
                
                self.logger.info(f"Pattern {pattern_name}: "
                               f"{old_threshold:.3f} → {new_threshold:.3f} ({reason})")
        
        # Metrikleri güncelle
        self.metrics.patterns_tracked = len(pattern_names)
        self.metrics.total_adjustments += adjustments_made
        self.metrics.last_calculated = datetime.now()
        
        self.logger.info(f"Toplu optimizasyon tamamlandı: "
                        f"{adjustments_made}/{len(pattern_names)} pattern ayarlandı")
        
        return results
    
    def analyze_pattern_performance(self, pattern_name: str) -> Dict[str, any]:
        """
        Pattern performansını detaylı analiz et
        
        Args:
            pattern_name: Pattern adı
            
        Returns:
            Dict: Analiz sonuçları
        """
        stats = self.tracker.get_pattern_stats(pattern_name)
        
        if stats.total_count == 0:
            return {"error": "Veri bulunamadı"}
        
        # Temel istatistikler
        analysis = {
            "pattern_name": pattern_name,
            "total_count": stats.total_count,
            "success_rate": stats.success_rate,
            "avg_confidence": stats.avg_confidence,
            "usage_count": stats.usage_count,
            "last_updated": stats.last_updated
        }
        
        # Son performans
        recent_success, recent_confidence = self.tracker.get_recent_performance(
            pattern_name, window_size=50
        )
        analysis["recent_success_rate"] = recent_success
        analysis["recent_avg_confidence"] = recent_confidence
        
        # Kararlılık
        stability = self.tracker.calculate_stability_score(pattern_name)
        analysis["stability_score"] = stability
        
        # Trend analizi
        trend_data = self.tracker.get_performance_trend(pattern_name, days=7)
        if len(trend_data) >= 2:
            # Son 7 günlük trend
            early_confidence = sum(conf for _, conf in trend_data[:len(trend_data)//2])
            late_confidence = sum(conf for _, conf in trend_data[len(trend_data)//2:])
            
            early_avg = early_confidence / (len(trend_data)//2) if len(trend_data)//2 > 0 else 0
            late_avg = late_confidence / (len(trend_data) - len(trend_data)//2)
            
            trend = "improving" if late_avg > early_avg + 0.05 else \
                   "declining" if late_avg < early_avg - 0.05 else "stable"
                   
            analysis["trend"] = trend
            analysis["trend_change"] = late_avg - early_avg
        else:
            analysis["trend"] = "insufficient_data"
            analysis["trend_change"] = 0.0
        
        # Öneriler
        recommendations = self._generate_recommendations(analysis)
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Öğrenme metriklerini al"""
        # İyileşme oranını hesapla
        if self.metrics.total_adjustments > 0:
            # Son ayarlamalardan sonraki ortalama performans artışını hesapla
            improvement = self._calculate_improvement_rate()
            self.metrics.improvement_rate = improvement
        
        return self.metrics
    
    def _get_current_threshold(self, pattern_name: str) -> float:
        """Mevcut threshold'u al (varsayılan olarak config'den)"""
        # Bu gerçek implementasyonda AdaptiveThresholdManager'dan alınacak
        return self.config.default_threshold
    
    def _calculate_usage_factor(self, usage_count: int) -> float:
        """
        Kullanım sıklığına göre ayarlama faktörü hesapla
        
        Sık kullanılan pattern'lar için daha dikkatli ayarlama.
        """
        if usage_count < 100:
            return 1.0  # Normal ayarlama
        elif usage_count < 500:
            return 0.8  # Biraz daha dikkatli
        elif usage_count < 1000:
            return 0.6  # Dikkatli
        else:
            return 0.4  # Çok dikkatli
    
    def _record_adjustment(self, pattern_name: str, threshold: float, reason: str) -> None:
        """Threshold ayarlamasını kaydet"""
        if pattern_name not in self._adjustment_history:
            self._adjustment_history[pattern_name] = []
        
        self._adjustment_history[pattern_name].append(
            (datetime.now(), threshold, reason)
        )
        
        # Geçmişi sınırla (son 100 kayıt)
        if len(self._adjustment_history[pattern_name]) > 100:
            self._adjustment_history[pattern_name] = self._adjustment_history[pattern_name][-100:]
    
    def _calculate_improvement_rate(self) -> float:
        """
        Genel iyileşme oranını hesapla
        
        Returns:
            float: İyileşme oranı (0.0-1.0)
        """
        # Bu basit bir implementasyon - gerçekte daha karmaşık olabilir
        all_patterns = self.tracker.get_all_patterns()
        
        if not all_patterns:
            return 0.0
        
        total_improvement = 0.0
        pattern_count = 0
        
        for pattern_name in all_patterns:
            stats = self.tracker.get_pattern_stats(pattern_name)
            
            if stats.total_count >= self.config.min_samples:
                # Son performansla genel performansı karşılaştır
                recent_success, _ = self.tracker.get_recent_performance(pattern_name, 20)
                improvement = max(0.0, recent_success - stats.success_rate)
                total_improvement += improvement
                pattern_count += 1
        
        if pattern_count > 0:
            return total_improvement / pattern_count
        else:
            return 0.0
    
    def _generate_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Analiz sonucuna göre öneriler oluştur"""
        recommendations = []
        
        success_rate = analysis.get("success_rate", 0.0)
        stability = analysis.get("stability_score", 0.0)
        trend = analysis.get("trend", "unknown")
        
        if success_rate < 0.7:
            recommendations.append("Düşük başarı oranı - pattern kurallarını gözden geçirin")
        
        if stability < 0.5:
            recommendations.append("Düşük kararlılık - pattern'in tutarlılığını artırın")
        
        if trend == "declining":
            recommendations.append("Azalan performans - pattern güncellemesi gerekebilir")
        
        if analysis.get("total_count", 0) < 50:
            recommendations.append("Daha fazla test verisi toplayın")
        
        if not recommendations:
            recommendations.append("Pattern performansı kabul edilebilir seviyede")
        
        return recommendations
