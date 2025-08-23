"""
Adaptive Threshold Manager

Pattern performansına göre threshold değerlerini otomatik ayarlayan ana sınıf.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .models import (
    PerformanceStats, ThresholdConfig, PatternResult, 
    AdaptiveState, PatternType, LearningMetrics
)
from .pattern_tracker import PatternTracker
from .learning import LearningEngine


class AdaptiveThresholdManager:
    """
    Adaptive threshold yönetici sınıfı
    
    Bu sınıf pattern'lerin performansını izleyerek threshold değerlerini
    otomatik olarak ayarlar.
    
    Temel mantık:
    - Başarı oranı >0.95 → threshold düşür (0.6'ya kadar)
    - Başarı oranı <0.7 → threshold yükselt (0.95'e kadar)
    - Pattern kullanım sıklığını da hesaba kat
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 persistence_file: Optional[str] = None,
                 auto_save: bool = True):
        """
        AdaptiveThresholdManager başlatıcı
        
        Args:
            config_file: Konfigürasyon dosya yolu
            persistence_file: Durum kayıt dosya yolu
            auto_save: Otomatik kaydetme aktif mi
        """
        self.config_file = config_file
        self.persistence_file = persistence_file or "adaptive_thresholds.json"
        self.auto_save = auto_save
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Konfigürasyonu yükle
        self.config = self._load_config()
        
        # Durumu başlat
        self.state = AdaptiveState(config=self.config)
        
        # Pattern tracker
        tracker_file = self.persistence_file.replace('.json', '_tracker.json')
        self.tracker = PatternTracker(persistence_file=tracker_file)
        
        # Learning engine
        self.learning_engine = LearningEngine(self.config, self.tracker)
        
        # Durum verilerini yükle
        self._load_state()
        
        self.logger.info("AdaptiveThresholdManager başlatıldı")
    
    def get_threshold(self, pattern_name: str, pattern_type: Optional[PatternType] = None) -> float:
        """
        Pattern için threshold değeri al
        
        Args:
            pattern_name: Pattern adı
            pattern_type: Pattern türü (opsiyonel)
            
        Returns:
            float: Threshold değeri
        """
        threshold = self.state.get_pattern_threshold(pattern_name)
        
        self.logger.debug(f"Pattern {pattern_name} için threshold: {threshold}")
        
        return threshold
    
    def update_performance(self, 
                          pattern_name: str,
                          success: bool,
                          confidence: float,
                          pattern_type: Optional[PatternType] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Pattern performansını güncelle
        
        Args:
            pattern_name: Pattern adı
            success: İşlem başarılı mı
            confidence: Güven skoru
            pattern_type: Pattern türü (opsiyonel)
            metadata: Ek bilgiler
        """
        # Pattern sonucunu oluştur
        result = PatternResult(
            pattern_name=pattern_name,
            pattern_type=pattern_type or PatternType.STREET,
            success=success,
            confidence=confidence,
            threshold_used=self.get_threshold(pattern_name),
            metadata=metadata
        )
        
        # Tracker'a kaydet
        self.tracker.track_result(result)
        
        # İstatistikleri güncelle
        current_stats = self.state.get_pattern_stats(pattern_name)
        updated_stats = current_stats.update_with_result(success, confidence)
        self.state.update_pattern_stats(pattern_name, updated_stats)
        
        # Yeterli veri varsa threshold'u güncelle
        if updated_stats.total_count >= self.config.min_samples:
            self._maybe_adjust_threshold(pattern_name)
        
        # Otomatik kaydet
        if self.auto_save and self.state.total_updates % 50 == 0:
            self.save_state()
        
        self.logger.debug(f"Pattern {pattern_name} performansı güncellendi: "
                         f"success={success}, confidence={confidence:.3f}")
    
    def get_pattern_stats(self, pattern_name: str) -> PerformanceStats:
        """
        Pattern istatistiklerini al
        
        Args:
            pattern_name: Pattern adı
            
        Returns:
            PerformanceStats: Pattern istatistikleri
        """
        return self.state.get_pattern_stats(pattern_name)
    
    def optimize_thresholds(self, pattern_names: Optional[List[str]] = None) -> Dict[str, Tuple[float, str]]:
        """
        Threshold'ları optimize et
        
        Args:
            pattern_names: Optimize edilecek pattern'lar (None ise tümü)
            
        Returns:
            Dict[str, Tuple[float, str]]: Pattern → (yeni_threshold, açıklama)
        """
        if pattern_names is None:
            pattern_names = list(self.state.pattern_stats.keys())
        
        # Learning engine ile optimize et
        optimizations = self.learning_engine.batch_optimize_thresholds(pattern_names)
        
        # Yeni threshold'ları uygula
        for pattern_name, (new_threshold, reason) in optimizations.items():
            old_threshold = self.state.get_pattern_threshold(pattern_name)
            
            if abs(new_threshold - old_threshold) > 0.01:
                self.state.set_pattern_threshold(pattern_name, new_threshold)
                
                self.logger.info(f"Pattern {pattern_name} threshold güncellendi: "
                               f"{old_threshold:.3f} → {new_threshold:.3f} ({reason})")
        
        # Durumu kaydet
        if self.auto_save:
            self.save_state()
        
        return optimizations
    
    def analyze_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """
        Pattern performansını detaylı analiz et
        
        Args:
            pattern_name: Pattern adı
            
        Returns:
            Dict: Analiz sonuçları
        """
        return self.learning_engine.analyze_pattern_performance(pattern_name)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Genel performans raporu al
        
        Returns:
            Dict: Performans raporu
        """
        all_patterns = list(self.state.pattern_stats.keys())
        
        report = {
            "total_patterns": len(all_patterns),
            "total_updates": self.state.total_updates,
            "last_save": self.state.last_save,
            "config": self.config.dict(),
            "learning_metrics": self.learning_engine.get_learning_metrics().dict()
        }
        
        # En iyi performans gösterenler
        top_performers = self.tracker.get_top_performers(limit=5)
        report["top_performers"] = [
            {"pattern": name, "success_rate": rate}
            for name, rate in top_performers
        ]
        
        # Düşük performans gösterenler
        low_performers = self.tracker.get_low_performers(
            threshold=self.config.low_performance_threshold, 
            limit=5
        )
        report["low_performers"] = [
            {"pattern": name, "success_rate": rate}
            for name, rate in low_performers
        ]
        
        # Threshold dağılımı
        threshold_distribution = {}
        for pattern_name in all_patterns:
            threshold = self.state.get_pattern_threshold(pattern_name)
            bucket = f"{threshold:.1f}"
            threshold_distribution[bucket] = threshold_distribution.get(bucket, 0) + 1
        
        report["threshold_distribution"] = threshold_distribution
        
        return report
    
    def save_state(self) -> None:
        """Durumu dosyaya kaydet"""
        try:
            # Ana durum dosyası
            with open(self.persistence_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.dict(), f, ensure_ascii=False, indent=2, default=str)
            
            # Tracker verilerini kaydet
            self.tracker.save_data()
            
            self.logger.info(f"Durum kaydedildi: {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Durum kaydetme hatası: {e}")
    
    def load_state(self) -> None:
        """Durumu dosyadan yükle"""
        self._load_state()
    
    def reset_pattern(self, pattern_name: str) -> None:
        """
        Pattern verilerini sıfırla
        
        Args:
            pattern_name: Pattern adı
        """
        # State'ten kaldır
        self.state.pattern_stats.pop(pattern_name, None)
        self.state.current_thresholds.pop(pattern_name, None)
        
        # Tracker'dan temizle
        self.tracker.clear_pattern_data(pattern_name)
        
        self.logger.info(f"Pattern {pattern_name} verileri sıfırlandı")
    
    def set_threshold(self, pattern_name: str, threshold: float) -> None:
        """
        Pattern threshold'unu manuel olarak ayarla
        
        Args:
            pattern_name: Pattern adı
            threshold: Yeni threshold değeri
        """
        # Sınırları kontrol et
        threshold = max(self.config.min_threshold,
                       min(self.config.max_threshold, threshold))
        
        old_threshold = self.state.get_pattern_threshold(pattern_name)
        self.state.set_pattern_threshold(pattern_name, threshold)
        
        self.logger.info(f"Pattern {pattern_name} threshold manuel ayarlandı: "
                        f"{old_threshold:.3f} → {threshold:.3f}")
    
    def _load_config(self) -> ThresholdConfig:
        """Konfigürasyonu yükle"""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return ThresholdConfig(**config_data)
            except Exception as e:
                self.logger.warning(f"Config yükleme hatası: {e}, varsayılan config kullanılıyor")
        
        return ThresholdConfig()
    
    def _load_state(self) -> None:
        """Durumu yükle"""
        if not Path(self.persistence_file).exists():
            return
        
        try:
            with open(self.persistence_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # datetime'ları parse et
            if 'last_save' in state_data:
                state_data['last_save'] = datetime.fromisoformat(state_data['last_save'])
            
            # Pattern stats'leri parse et
            if 'pattern_stats' in state_data:
                parsed_stats = {}
                for name, stats_data in state_data['pattern_stats'].items():
                    if 'last_updated' in stats_data:
                        stats_data['last_updated'] = datetime.fromisoformat(stats_data['last_updated'])
                    parsed_stats[name] = PerformanceStats(**stats_data)
                state_data['pattern_stats'] = parsed_stats
            
            self.state = AdaptiveState(**state_data)
            
            # Config'i güncelle
            if hasattr(self.state, 'config'):
                self.config = self.state.config
            
            self.logger.info(f"Durum yüklendi: {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Durum yükleme hatası: {e}")
    
    def _maybe_adjust_threshold(self, pattern_name: str) -> None:
        """
        Gerekirse threshold'u ayarla
        
        Args:
            pattern_name: Pattern adı
        """
        stats = self.state.get_pattern_stats(pattern_name)
        
        # Yeterli veri yoksa çık
        if stats.total_count < self.config.min_samples:
            return
        
        # Learning engine ile optimal threshold hesapla
        old_threshold = self.state.get_pattern_threshold(pattern_name)
        new_threshold, reason = self.learning_engine.calculate_optimal_threshold(pattern_name)
        
        # Anlamlı değişiklik varsa uygula
        if abs(new_threshold - old_threshold) > 0.01:
            self.state.set_pattern_threshold(pattern_name, new_threshold)
            
            self.logger.info(f"Pattern {pattern_name} threshold otomatik ayarlandı: "
                           f"{old_threshold:.3f} → {new_threshold:.3f} ({reason})")
    
    def __enter__(self):
        """Context manager giriş"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkış"""
        if self.auto_save:
            self.save_state()
