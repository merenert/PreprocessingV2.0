"""
Pattern Performance Tracker

Pattern'lerin performansını takip eden sınıf.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from .models import PerformanceStats, PatternResult, PatternType


class PatternTracker:
    """
    Pattern performans takip sınıfı
    
    Bu sınıf pattern'lerin başarı oranlarını, güven skorlarını ve
    kullanım sıklıklarını takip eder.
    """
    
    def __init__(self, 
                 history_size: int = 1000,
                 persistence_file: Optional[str] = None):
        """
        PatternTracker başlatıcı
        
        Args:
            history_size: Tutulacak maksimum geçmiş sonuç sayısı
            persistence_file: Verilerin kaydedileceği dosya yolu
        """
        self.history_size = history_size
        self.persistence_file = persistence_file
        
        # Pattern istatistikleri
        self._pattern_stats: Dict[str, PerformanceStats] = {}
        
        # Son sonuçlar (pattern bazında)
        self._recent_results: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Performans geçmişi (zaman bazlı analiz için)
        self._performance_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Veri yükle
        self._load_data()
    
    def track_result(self, result: PatternResult) -> None:
        """
        Pattern sonucunu takip et
        
        Args:
            result: Pattern sonucu
        """
        pattern_name = result.pattern_name
        
        # Mevcut istatistikleri al
        current_stats = self._pattern_stats.get(pattern_name, PerformanceStats())
        
        # İstatistikleri güncelle
        updated_stats = current_stats.update_with_result(
            success=result.success,
            confidence=result.confidence
        )
        
        # Kaydet
        self._pattern_stats[pattern_name] = updated_stats
        
        # Son sonuçlara ekle
        self._recent_results[pattern_name].append(result)
        
        # Performans geçmişine ekle
        self._performance_history[pattern_name].append(
            (result.timestamp, result.confidence)
        )
        
        # Geçmişi temizle (çok eski kayıtları sil)
        self._cleanup_old_history(pattern_name)
        
        self.logger.debug(f"Pattern {pattern_name} sonucu kaydedildi: "
                         f"success={result.success}, confidence={result.confidence:.3f}")
    
    def get_pattern_stats(self, pattern_name: str) -> PerformanceStats:
        """
        Pattern istatistiklerini al
        
        Args:
            pattern_name: Pattern adı
            
        Returns:
            PerformanceStats: Pattern istatistikleri
        """
        return self._pattern_stats.get(pattern_name, PerformanceStats())
    
    def get_recent_performance(self, 
                             pattern_name: str, 
                             window_size: int = 50) -> Tuple[float, float]:
        """
        Son N sonuca göre performans al
        
        Args:
            pattern_name: Pattern adı
            window_size: Pencere boyutu
            
        Returns:
            Tuple[float, float]: (başarı_oranı, ortalama_güven)
        """
        recent = list(self._recent_results[pattern_name])[-window_size:]
        
        if not recent:
            return 0.0, 0.0
        
        success_count = sum(1 for r in recent if r.success)
        success_rate = success_count / len(recent)
        
        avg_confidence = sum(r.confidence for r in recent) / len(recent)
        
        return success_rate, avg_confidence
    
    def get_performance_trend(self, 
                            pattern_name: str, 
                            days: int = 7) -> List[Tuple[datetime, float]]:
        """
        Performans trendini al
        
        Args:
            pattern_name: Pattern adı
            days: Kaç günlük trend
            
        Returns:
            List[Tuple[datetime, float]]: Zaman ve güven skoru çiftleri
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        history = self._performance_history.get(pattern_name, [])
        recent_history = [
            (timestamp, confidence) 
            for timestamp, confidence in history 
            if timestamp >= cutoff_date
        ]
        
        return recent_history
    
    def get_all_patterns(self) -> List[str]:
        """
        Tüm takip edilen pattern'leri al
        
        Returns:
            List[str]: Pattern adları
        """
        return list(self._pattern_stats.keys())
    
    def get_top_performers(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        En iyi performans gösteren pattern'leri al
        
        Args:
            limit: Maksimum sonuç sayısı
            
        Returns:
            List[Tuple[str, float]]: (pattern_adı, başarı_oranı) çiftleri
        """
        performers = [
            (name, stats.success_rate)
            for name, stats in self._pattern_stats.items()
            if stats.total_count >= 10  # Minimum örnek sayısı
        ]
        
        # Başarı oranına göre sırala
        performers.sort(key=lambda x: x[1], reverse=True)
        
        return performers[:limit]
    
    def get_low_performers(self, 
                          threshold: float = 0.7, 
                          limit: int = 10) -> List[Tuple[str, float]]:
        """
        Düşük performans gösteren pattern'leri al
        
        Args:
            threshold: Düşük performans eşiği
            limit: Maksimum sonuç sayısı
            
        Returns:
            List[Tuple[str, float]]: (pattern_adı, başarı_oranı) çiftleri
        """
        low_performers = [
            (name, stats.success_rate)
            for name, stats in self._pattern_stats.items()
            if stats.success_rate < threshold and stats.total_count >= 10
        ]
        
        # Başarı oranına göre sırala (en düşük önce)
        low_performers.sort(key=lambda x: x[1])
        
        return low_performers[:limit]
    
    def calculate_stability_score(self, pattern_name: str) -> float:
        """
        Pattern kararlılık skorunu hesapla
        
        Kararlılık, son sonuçların ne kadar tutarlı olduğunu gösterir.
        
        Args:
            pattern_name: Pattern adı
            
        Returns:
            float: Kararlılık skoru (0.0-1.0)
        """
        recent = list(self._recent_results[pattern_name])[-50:]  # Son 50 sonuç
        
        if len(recent) < 10:
            return 0.0
        
        # Güven skorlarının standart sapmasını hesapla
        confidences = [r.confidence for r in recent]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        
        # Kararlılık = 1 - normalize_edilmiş_std_sapma
        # Std sapma 0.2'den büyükse kararlılık 0'a yaklaşır
        stability = max(0.0, 1.0 - (std_dev / 0.2))
        
        return stability
    
    def save_data(self) -> None:
        """Verileri dosyaya kaydet"""
        if not self.persistence_file:
            return
        
        try:
            data = {
                'pattern_stats': {
                    name: stats.dict() 
                    for name, stats in self._pattern_stats.items()
                },
                'recent_results': {
                    name: [result.dict() for result in results]
                    for name, results in self._recent_results.items()
                },
                'last_save': datetime.now().isoformat()
            }
            
            with open(self.persistence_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                
            self.logger.info(f"Pattern tracker verileri kaydedildi: {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Veri kaydetme hatası: {e}")
    
    def _load_data(self) -> None:
        """Verileri dosyadan yükle"""
        if not self.persistence_file or not Path(self.persistence_file).exists():
            return
        
        try:
            with open(self.persistence_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Pattern istatistiklerini yükle
            for name, stats_data in data.get('pattern_stats', {}).items():
                # datetime'ları parse et
                if 'last_updated' in stats_data:
                    stats_data['last_updated'] = datetime.fromisoformat(stats_data['last_updated'])
                
                self._pattern_stats[name] = PerformanceStats(**stats_data)
            
            # Son sonuçları yükle
            for name, results_data in data.get('recent_results', {}).items():
                results_deque = deque(maxlen=self.history_size)
                
                for result_data in results_data:
                    # datetime'ları parse et
                    if 'timestamp' in result_data:
                        result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp'])
                    
                    result = PatternResult(**result_data)
                    results_deque.append(result)
                
                self._recent_results[name] = results_deque
            
            self.logger.info(f"Pattern tracker verileri yüklendi: {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Veri yükleme hatası: {e}")
    
    def _cleanup_old_history(self, pattern_name: str, days: int = 30) -> None:
        """
        Eski geçmiş kayıtlarını temizle
        
        Args:
            pattern_name: Pattern adı
            days: Kaç günden eski kayıtları sil
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        history = self._performance_history[pattern_name]
        self._performance_history[pattern_name] = [
            (timestamp, confidence)
            for timestamp, confidence in history
            if timestamp >= cutoff_date
        ]
    
    def clear_pattern_data(self, pattern_name: str) -> None:
        """
        Belirtilen pattern'in verilerini temizle
        
        Args:
            pattern_name: Pattern adı
        """
        self._pattern_stats.pop(pattern_name, None)
        self._recent_results.pop(pattern_name, None)
        self._performance_history.pop(pattern_name, None)
        
        self.logger.info(f"Pattern {pattern_name} verileri temizlendi")
    
    def reset_all_data(self) -> None:
        """Tüm verileri sıfırla"""
        self._pattern_stats.clear()
        self._recent_results.clear()
        self._performance_history.clear()
        
        self.logger.info("Tüm pattern tracker verileri sıfırlandı")
