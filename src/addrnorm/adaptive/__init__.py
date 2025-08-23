"""
Adaptive Threshold Management Module

Bu modül Türkçe adres normalizasyon pipeline'ında kullanılan pattern'lerin
performansını izleyerek threshold değerlerini otomatik olarak ayarlar.

Temel özellikler:
- Pattern performans takibi
- Dinamik threshold hesaplama  
- JSON persistence
- Pipeline entegrasyonu

Kullanım örneği:
    from addrnorm.adaptive import AdaptiveThresholdManager
    
    manager = AdaptiveThresholdManager()
    threshold = manager.get_threshold("street_pattern")
    
    # Pattern sonucunu bildir
    manager.update_performance("street_pattern", success=True, confidence=0.85)
"""

from .threshold_manager import AdaptiveThresholdManager
from .pattern_tracker import PatternTracker
from .learning import LearningEngine
from .models import PerformanceStats, ThresholdConfig, PatternResult, PatternType

__all__ = [
    'AdaptiveThresholdManager',
    'PatternTracker', 
    'LearningEngine',
    'PerformanceStats',
    'ThresholdConfig',
    'PatternResult',
    'PatternType'
]

def create_adaptive_manager(config_file: str = None) -> AdaptiveThresholdManager:
    """
    Adaptive threshold manager oluştur
    
    Args:
        config_file: Configuration dosya yolu
        
    Returns:
        AdaptiveThresholdManager instance
    """
    return AdaptiveThresholdManager(config_file=config_file)

# Örnek kullanım
EXAMPLE_USAGE = """
# Temel kullanım
manager = create_adaptive_manager()

# Threshold al
threshold = manager.get_threshold("street_pattern")

# Pattern sonucunu bildir
manager.update_performance("street_pattern", success=True, confidence=0.85)

# Performans istatistikleri
stats = manager.get_pattern_stats("street_pattern")
print(f"Başarı oranı: {stats.success_rate:.2%}")
print(f"Ortalama güven: {stats.avg_confidence:.2f}")
print(f"Kullanım sayısı: {stats.usage_count}")
"""
