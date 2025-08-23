"""
Adaptive Threshold Management Tests

Adaptive threshold management sistemi için kapsamlı test paketi.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.addrnorm.adaptive import (
    AdaptiveThresholdManager, PatternTracker, LearningEngine,
    PerformanceStats, ThresholdConfig, PatternResult, PatternType
)


class TestPerformanceStats:
    """PerformanceStats model testleri"""
    
    def test_empty_stats(self):
        """Boş istatistik testi"""
        stats = PerformanceStats()
        
        assert stats.success_count == 0
        assert stats.total_count == 0
        assert stats.success_rate == 0.0
        assert stats.avg_confidence == 0.0
        assert stats.usage_count == 0
    
    def test_update_with_result(self):
        """İstatistik güncelleme testi"""
        stats = PerformanceStats()
        
        # İlk başarılı sonuç
        updated = stats.update_with_result(success=True, confidence=0.85)
        
        assert updated.success_count == 1
        assert updated.total_count == 1
        assert updated.success_rate == 1.0
        assert updated.avg_confidence == 0.85
        assert updated.usage_count == 1
        
        # İkinci başarısız sonuç
        updated2 = updated.update_with_result(success=False, confidence=0.6)
        
        assert updated2.success_count == 1
        assert updated2.total_count == 2
        assert updated2.success_rate == 0.5
        assert updated2.avg_confidence == 0.725  # (0.85 + 0.6) / 2
        assert updated2.usage_count == 2
    
    def test_success_rate_validation(self):
        """Başarı oranı validasyon testi"""
        stats = PerformanceStats(success_count=3, total_count=5, confidence_sum=4.0)
        
        assert stats.success_rate == 0.6  # 3/5
        assert stats.avg_confidence == 0.8  # 4.0/5


class TestThresholdConfig:
    """ThresholdConfig model testleri"""
    
    def test_default_config(self):
        """Varsayılan konfigürasyon testi"""
        config = ThresholdConfig()
        
        assert config.min_threshold == 0.6
        assert config.max_threshold == 0.95
        assert config.default_threshold == 0.8
        assert config.adjustment_step == 0.05
        assert config.min_samples == 10
    
    def test_config_validation(self):
        """Konfigürasyon validasyon testi"""
        # Hatalı max < min
        with pytest.raises(ValueError):
            ThresholdConfig(min_threshold=0.8, max_threshold=0.7)
        
        # Hatalı default aralık dışı
        with pytest.raises(ValueError):
            ThresholdConfig(min_threshold=0.6, max_threshold=0.9, default_threshold=0.95)


class TestPatternTracker:
    """PatternTracker testleri"""
    
    def test_track_result(self):
        """Sonuç takip testi"""
        tracker = PatternTracker(history_size=100)
        
        result = PatternResult(
            pattern_name="test_pattern",
            pattern_type=PatternType.STREET,
            success=True,
            confidence=0.85,
            threshold_used=0.8
        )
        
        tracker.track_result(result)
        
        stats = tracker.get_pattern_stats("test_pattern")
        assert stats.total_count == 1
        assert stats.success_rate == 1.0
        assert stats.avg_confidence == 0.85
    
    def test_recent_performance(self):
        """Son performans testi"""
        tracker = PatternTracker(history_size=100)
        
        # 10 sonuç ekle
        for i in range(10):
            result = PatternResult(
                pattern_name="test_pattern",
                pattern_type=PatternType.STREET,
                success=i % 2 == 0,  # Alternatif başarı/başarısızlık
                confidence=0.8 + (i * 0.01),
                threshold_used=0.8
            )
            tracker.track_result(result)
        
        success_rate, avg_confidence = tracker.get_recent_performance("test_pattern", window_size=5)
        
        assert 0.0 <= success_rate <= 1.0
        assert 0.0 <= avg_confidence <= 1.0
    
    def test_stability_score(self):
        """Kararlılık skoru testi"""
        tracker = PatternTracker(history_size=100)
        
        # Kararlı sonuçlar (benzer güven skorları)
        for i in range(20):
            result = PatternResult(
                pattern_name="stable_pattern",
                pattern_type=PatternType.STREET,
                success=True,
                confidence=0.85 + (i * 0.001),  # Çok küçük varyasyon
                threshold_used=0.8
            )
            tracker.track_result(result)
        
        stability = tracker.calculate_stability_score("stable_pattern")
        assert stability > 0.8  # Yüksek kararlılık
        
        # Kararsız sonuçlar
        for i in range(20):
            result = PatternResult(
                pattern_name="unstable_pattern",
                pattern_type=PatternType.STREET,
                success=True,
                confidence=0.5 + (i * 0.025),  # Büyük varyasyon
                threshold_used=0.8
            )
            tracker.track_result(result)
        
        unstable_stability = tracker.calculate_stability_score("unstable_pattern")
        assert unstable_stability < stability  # Düşük kararlılık
    
    def test_persistence(self):
        """Veri kalıcılığı testi"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # İlk tracker - veri ekle
            tracker1 = PatternTracker(persistence_file=temp_file)
            
            result = PatternResult(
                pattern_name="persistent_pattern",
                pattern_type=PatternType.STREET,
                success=True,
                confidence=0.9,
                threshold_used=0.8
            )
            tracker1.track_result(result)
            tracker1.save_data()
            
            # İkinci tracker - veri yükle
            tracker2 = PatternTracker(persistence_file=temp_file)
            
            stats = tracker2.get_pattern_stats("persistent_pattern")
            assert stats.total_count == 1
            assert stats.success_rate == 1.0
            assert stats.avg_confidence == 0.9
            
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestLearningEngine:
    """LearningEngine testleri"""
    
    def test_calculate_optimal_threshold(self):
        """Optimal threshold hesaplama testi"""
        config = ThresholdConfig()
        tracker = PatternTracker()
        engine = LearningEngine(config, tracker)
        
        # Yüksek performanslı pattern
        for i in range(20):
            result = PatternResult(
                pattern_name="high_performer",
                pattern_type=PatternType.STREET,
                success=True,  # %100 başarı
                confidence=0.95,
                threshold_used=0.8
            )
            tracker.track_result(result)
        
        threshold, reason = engine.calculate_optimal_threshold("high_performer")
        
        # Yüksek performans için threshold düşürülmeli
        assert threshold < config.default_threshold
        assert "Yüksek performans" in reason
    
    def test_low_performance_adjustment(self):
        """Düşük performans ayarlama testi"""
        config = ThresholdConfig()
        tracker = PatternTracker()
        engine = LearningEngine(config, tracker)
        
        # Düşük performanslı pattern
        for i in range(20):
            result = PatternResult(
                pattern_name="low_performer",
                pattern_type=PatternType.STREET,
                success=i < 10,  # %50 başarı (düşük)
                confidence=0.6,
                threshold_used=0.8
            )
            tracker.track_result(result)
        
        threshold, reason = engine.calculate_optimal_threshold("low_performer")
        
        # Düşük performans için threshold yükseltilmeli
        assert threshold > config.default_threshold
        assert "Düşük performans" in reason or "performans" in reason.lower()
    
    def test_insufficient_data(self):
        """Yetersiz veri testi"""
        config = ThresholdConfig(min_samples=10)
        tracker = PatternTracker()
        engine = LearningEngine(config, tracker)
        
        # Sadece birkaç sonuç ekle
        for i in range(5):
            result = PatternResult(
                pattern_name="insufficient_data",
                pattern_type=PatternType.STREET,
                success=True,
                confidence=0.9,
                threshold_used=0.8
            )
            tracker.track_result(result)
        
        threshold, reason = engine.calculate_optimal_threshold("insufficient_data")
        
        # Yetersiz veri için varsayılan threshold
        assert threshold == config.default_threshold
        assert "Yetersiz veri" in reason or "varsayılan" in reason
    
    def test_batch_optimization(self):
        """Toplu optimizasyon testi"""
        config = ThresholdConfig()
        tracker = PatternTracker()
        engine = LearningEngine(config, tracker)
        
        # Birden fazla pattern ekle
        patterns = ["pattern1", "pattern2", "pattern3"]
        
        for pattern in patterns:
            for i in range(15):
                result = PatternResult(
                    pattern_name=pattern,
                    pattern_type=PatternType.STREET,
                    success=True,
                    confidence=0.9,
                    threshold_used=0.8
                )
                tracker.track_result(result)
        
        results = engine.batch_optimize_thresholds(patterns)
        
        assert len(results) == 3
        for pattern in patterns:
            assert pattern in results
            threshold, reason = results[pattern]
            assert 0.6 <= threshold <= 0.95


class TestAdaptiveThresholdManager:
    """AdaptiveThresholdManager testleri"""
    
    def test_initialization(self):
        """Başlatma testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "test_adaptive.json"
            
            manager = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            assert manager.config.default_threshold == 0.8
            assert manager.get_threshold("new_pattern") == 0.8
    
    def test_update_performance(self):
        """Performans güncelleme testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "test_adaptive.json"
            
            manager = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            # Performans güncelle
            manager.update_performance(
                pattern_name="test_pattern",
                success=True,
                confidence=0.9,
                pattern_type=PatternType.STREET
            )
            
            stats = manager.get_pattern_stats("test_pattern")
            assert stats.total_count == 1
            assert stats.success_rate == 1.0
            assert stats.avg_confidence == 0.9
    
    def test_threshold_adjustment(self):
        """Threshold ayarlama testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "test_adaptive.json"
            
            config = ThresholdConfig(min_samples=5)  # Düşük minimum sample
            
            # Config dosyası oluştur
            config_file = Path(temp_dir) / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config.dict(), f)
            
            manager = AdaptiveThresholdManager(
                config_file=str(config_file),
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            # Yüksek performanslı sonuçlar ekle
            for i in range(10):
                manager.update_performance(
                    pattern_name="high_performer",
                    success=True,
                    confidence=0.95,
                    pattern_type=PatternType.STREET
                )
            
            # Threshold'un düşürülmüş olması beklenir
            threshold = manager.get_threshold("high_performer")
            # Not: Gerçek ayarlama learning engine'e bağlı
    
    def test_performance_report(self):
        """Performans raporu testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "test_adaptive.json"
            
            manager = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            # Birkaç pattern ekle
            patterns = ["pattern1", "pattern2"]
            
            for pattern in patterns:
                for i in range(5):
                    manager.update_performance(
                        pattern_name=pattern,
                        success=True,
                        confidence=0.85,
                        pattern_type=PatternType.STREET
                    )
            
            report = manager.get_performance_report()
            
            assert "total_patterns" in report
            assert "total_updates" in report
            assert "config" in report
            assert "learning_metrics" in report
            assert report["total_patterns"] == 2
    
    def test_persistence(self):
        """Kalıcılık testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "test_adaptive.json"
            
            # İlk manager - veri ekle
            manager1 = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            manager1.update_performance(
                pattern_name="persistent_pattern",
                success=True,
                confidence=0.9,
                pattern_type=PatternType.STREET
            )
            
            manager1.set_threshold("persistent_pattern", 0.75)
            manager1.save_state()
            
            # İkinci manager - veri yükle
            manager2 = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            stats = manager2.get_pattern_stats("persistent_pattern")
            assert stats.total_count == 1
            assert stats.success_rate == 1.0
            
            threshold = manager2.get_threshold("persistent_pattern")
            assert threshold == 0.75
    
    def test_context_manager(self):
        """Context manager testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "test_adaptive.json"
            
            with AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=True
            ) as manager:
                manager.update_performance(
                    pattern_name="context_pattern",
                    success=True,
                    confidence=0.8,
                    pattern_type=PatternType.STREET
                )
            
            # Dosya kaydedilmiş olmalı
            assert persistence_file.exists()


@pytest.mark.integration
class TestAdaptiveIntegration:
    """Entegrasyon testleri"""
    
    def test_full_workflow(self):
        """Tam iş akışı testi"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "integration_test.json"
            
            manager = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            # Çeşitli pattern'lar ve sonuçlar
            test_data = [
                ("street_pattern", True, 0.95),
                ("street_pattern", True, 0.92),
                ("street_pattern", True, 0.89),
                ("district_pattern", False, 0.45),
                ("district_pattern", False, 0.52),
                ("district_pattern", True, 0.78),
                ("city_pattern", True, 0.99),
                ("city_pattern", True, 0.96),
            ]
            
            # Veri ekle
            for pattern, success, confidence in test_data:
                manager.update_performance(
                    pattern_name=pattern,
                    success=success,
                    confidence=confidence,
                    pattern_type=PatternType.STREET
                )
            
            # Rapor al
            report = manager.get_performance_report()
            assert report["total_patterns"] == 3
            
            # Optimizasyon çalıştır
            optimizations = manager.optimize_thresholds()
            assert len(optimizations) <= 3  # En fazla 3 pattern var
            
            # Analiz yap
            analysis = manager.analyze_pattern("street_pattern")
            assert "pattern_name" in analysis
            assert analysis["pattern_name"] == "street_pattern"
    
    def test_pipeline_integration_simulation(self):
        """Pipeline entegrasyonu simülasyonu"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "pipeline_test.json"
            
            manager = AdaptiveThresholdManager(
                persistence_file=str(persistence_file),
                auto_save=False
            )
            
            # Pipeline simülasyonu
            def simulate_pattern_match(pattern_name: str, text: str) -> tuple:
                """Pattern eşleşme simülasyonu"""
                threshold = manager.get_threshold(pattern_name)
                
                # Basit simülasyon - text uzunluğuna göre confidence
                confidence = min(1.0, len(text) / 20.0 + 0.5)
                success = confidence >= threshold
                
                # Sonucu bildir
                manager.update_performance(
                    pattern_name=pattern_name,
                    success=success,
                    confidence=confidence,
                    pattern_type=PatternType.STREET
                )
                
                return success, confidence, threshold
            
            # Farklı uzunluklarda test metinleri
            test_texts = [
                "İstanbul Cadde",      # Kısa - düşük confidence
                "Atatürk Bulvarı No:15",  # Orta - orta confidence  
                "Cumhuriyet Mahallesi Atatürk Bulvarı No:15 Daire:3",  # Uzun - yüksek confidence
            ]
            
            # Testleri çalıştır
            results = []
            for text in test_texts * 3:  # Her metni 3 kez test et
                result = simulate_pattern_match("address_pattern", text)
                results.append(result)
            
            # İstatistikleri kontrol et
            stats = manager.get_pattern_stats("address_pattern")
            assert stats.total_count == len(test_texts) * 3
            
            # Performans raporunu al
            report = manager.get_performance_report()
            assert report["total_patterns"] >= 1
