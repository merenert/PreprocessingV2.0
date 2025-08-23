"""
Adaptive Threshold Management Demo

Bu script adaptive threshold management sisteminin kullanımını gösterir.
"""

import json
import time
import random
from pathlib import Path

from src.addrnorm.adaptive import (
    AdaptiveThresholdManager, PatternType, create_adaptive_manager
)


def simulate_address_processing():
    """Adres işleme simülasyonu"""
    
    print("🔧 Adaptive Threshold Management Demo")
    print("=" * 50)
    
    # Demo için geçici dosyalar
    demo_dir = Path("demo_adaptive_data")
    demo_dir.mkdir(exist_ok=True)
    
    config_file = demo_dir / "demo_config.json"
    persistence_file = demo_dir / "demo_state.json"
    
    # Demo config oluştur
    demo_config = {
        "min_threshold": 0.6,
        "max_threshold": 0.95,
        "default_threshold": 0.8,
        "adjustment_step": 0.05,
        "min_samples": 5,  # Demo için düşük
        "high_performance_threshold": 0.95,
        "low_performance_threshold": 0.7
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(demo_config, f, indent=2)
    
    # Manager oluştur
    print(f"📁 Config dosyası: {config_file}")
    print(f"💾 Persistence dosyası: {persistence_file}")
    print()
    
    with AdaptiveThresholdManager(
        config_file=str(config_file),
        persistence_file=str(persistence_file),
        auto_save=True
    ) as manager:
        
        # Test pattern'ları
        patterns = [
            ("street_cleaner", PatternType.STREET),
            ("district_detector", PatternType.DISTRICT), 
            ("building_finder", PatternType.BUILDING),
            ("number_extractor", PatternType.NUMBER),
            ("explanation_parser", PatternType.EXPLANATION)
        ]
        
        print("📊 Başlangıç threshold'ları:")
        for pattern_name, pattern_type in patterns:
            threshold = manager.get_threshold(pattern_name)
            print(f"  {pattern_name}: {threshold:.3f}")
        print()
        
        # Simülasyon parametreleri
        simulation_rounds = 5
        results_per_round = 20
        
        print(f"🎯 Simülasyon başlıyor: {simulation_rounds} round, "
              f"round başına {results_per_round} sonuç")
        print()
        
        for round_num in range(1, simulation_rounds + 1):
            print(f"Round {round_num}/{simulation_rounds}")
            print("-" * 30)
            
            round_results = {}
            
            for pattern_name, pattern_type in patterns:
                # Pattern performans karakteristiği belirle
                pattern_characteristics = get_pattern_characteristics(pattern_name)
                
                round_results[pattern_name] = []
                
                # Bu round için sonuçlar üret
                for _ in range(results_per_round):
                    success, confidence = simulate_pattern_result(pattern_characteristics)
                    
                    # Manager'a bildir
                    manager.update_performance(
                        pattern_name=pattern_name,
                        success=success,
                        confidence=confidence,
                        pattern_type=pattern_type,
                        metadata={"round": round_num}
                    )
                    
                    round_results[pattern_name].append((success, confidence))
                
                # Round sonuçlarını göster
                successes = sum(1 for s, _ in round_results[pattern_name] if s)
                avg_conf = sum(c for _, c in round_results[pattern_name]) / len(round_results[pattern_name])
                
                print(f"  {pattern_name}:")
                print(f"    Başarı: {successes}/{results_per_round} ({successes/results_per_round:.1%})")
                print(f"    Ort. güven: {avg_conf:.3f}")
            
            print()
            
            # Round sonunda threshold'ları göster
            print("🎚️ Güncel threshold'lar:")
            for pattern_name, _ in patterns:
                threshold = manager.get_threshold(pattern_name)
                print(f"  {pattern_name}: {threshold:.3f}")
            
            print()
            time.sleep(1)  # Demo efekti için bekle
        
        # Optimizasyon çalıştır
        print("🔧 Threshold optimizasyonu çalıştırılıyor...")
        optimizations = manager.optimize_thresholds()
        
        print("\n📈 Optimizasyon sonuçları:")
        for pattern_name, (new_threshold, reason) in optimizations.items():
            print(f"  {pattern_name}: {new_threshold:.3f}")
            print(f"    Açıklama: {reason}")
        
        print()
        
        # Detaylı analiz
        print("🔍 Detaylı pattern analizleri:")
        print("=" * 40)
        
        for pattern_name, _ in patterns:
            analysis = manager.analyze_pattern(pattern_name)
            
            if "error" not in analysis:
                print(f"\n📊 {pattern_name}:")
                print(f"  Toplam test: {analysis['total_count']}")
                print(f"  Başarı oranı: {analysis['success_rate']:.1%}")
                print(f"  Ort. güven: {analysis['avg_confidence']:.3f}")
                print(f"  Son başarı: {analysis['recent_success_rate']:.1%}")
                print(f"  Kararlılık: {analysis['stability_score']:.3f}")
                print(f"  Trend: {analysis.get('trend', 'bilinmiyor')}")
                
                if analysis.get('recommendations'):
                    print("  📋 Öneriler:")
                    for rec in analysis['recommendations']:
                        print(f"    • {rec}")
        
        # Genel performans raporu
        print("\n📋 Genel Performans Raporu:")
        print("=" * 30)
        
        report = manager.get_performance_report()
        
        print(f"Toplam pattern: {report['total_patterns']}")
        print(f"Toplam güncelleme: {report['total_updates']}")
        
        if report.get('top_performers'):
            print("\n🏆 En iyi performans gösterenler:")
            for perf in report['top_performers'][:3]:
                print(f"  {perf['pattern']}: {perf['success_rate']:.1%}")
        
        if report.get('low_performers'):
            print("\n⚠️ Düşük performans gösterenler:")
            for perf in report['low_performers'][:3]:
                print(f"  {perf['pattern']}: {perf['success_rate']:.1%}")
        
        if report.get('threshold_distribution'):
            print("\n📊 Threshold dağılımı:")
            for threshold, count in sorted(report['threshold_distribution'].items()):
                print(f"  {threshold}: {count} pattern")
        
        print(f"\n💾 Veriler kaydedildi: {persistence_file}")
        
        # Kaydedilen veriyi göster
        if persistence_file.exists():
            file_size = persistence_file.stat().st_size
            print(f"📄 Dosya boyutu: {file_size} byte")


def get_pattern_characteristics(pattern_name: str) -> dict:
    """Pattern karakteristiklerini belirle"""
    
    characteristics = {
        "street_cleaner": {
            "base_success_rate": 0.85,
            "confidence_range": (0.7, 0.95),
            "stability": 0.8
        },
        "district_detector": {
            "base_success_rate": 0.92,
            "confidence_range": (0.8, 0.98),
            "stability": 0.9
        },
        "building_finder": {
            "base_success_rate": 0.75,
            "confidence_range": (0.6, 0.9),
            "stability": 0.6
        },
        "number_extractor": {
            "base_success_rate": 0.95,
            "confidence_range": (0.85, 0.99),
            "stability": 0.95
        },
        "explanation_parser": {
            "base_success_rate": 0.68,
            "confidence_range": (0.5, 0.85),
            "stability": 0.5
        }
    }
    
    return characteristics.get(pattern_name, {
        "base_success_rate": 0.8,
        "confidence_range": (0.6, 0.9),
        "stability": 0.7
    })


def simulate_pattern_result(characteristics: dict) -> tuple:
    """Pattern sonucu simüle et"""
    
    base_success = characteristics.get("base_success_rate", 0.8)
    conf_min, conf_max = characteristics.get("confidence_range", (0.6, 0.9))
    stability = characteristics.get("stability", 0.7)
    
    # Confidence üret
    if stability > 0.8:
        # Yüksek stabilite - dar aralık
        conf_center = (conf_min + conf_max) / 2
        conf_range = (conf_max - conf_min) * 0.3
        confidence = random.gauss(conf_center, conf_range / 3)
    else:
        # Düşük stabilite - geniş aralık
        confidence = random.uniform(conf_min, conf_max)
    
    # Sınırla
    confidence = max(conf_min, min(conf_max, confidence))
    
    # Başarıyı belirle (confidence'a bağlı olarak)
    success_threshold = base_success
    
    # Confidence yüksekse başarı olasılığı artar
    if confidence > (conf_min + conf_max) / 2:
        success_threshold += 0.1
    else:
        success_threshold -= 0.1
    
    success = random.random() < success_threshold
    
    return success, confidence


if __name__ == "__main__":
    try:
        simulate_address_processing()
        
        print("\n✅ Demo tamamlandı!")
        print("\n💡 Gerçek kullanım örneği:")
        print("""
from addrnorm.adaptive import AdaptiveThresholdManager

# Manager oluştur
manager = AdaptiveThresholdManager(
    config_file="adaptive_config.json",
    persistence_file="adaptive_state.json"
)

# Threshold al
threshold = manager.get_threshold("street_pattern")

# Pattern sonucunu bildir  
manager.update_performance(
    pattern_name="street_pattern",
    success=True,
    confidence=0.85
)

# Performans analizi
analysis = manager.analyze_pattern("street_pattern")
print(f"Başarı oranı: {analysis['success_rate']:.1%}")
        """)
        
    except Exception as e:
        print(f"❌ Demo hatası: {e}")
        import traceback
        traceback.print_exc()
