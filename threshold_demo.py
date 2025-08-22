#!/usr/bin/env python3
"""Threshold değişimini gösteren demo."""

import sys

sys.path.append("src")

from addrnorm.patterns.matcher import PatternMatcher
from addrnorm.preprocess import preprocess


def main():
    """Dramatic threshold demo."""
    matcher = PatternMatcher()

    # Test adresi
    text = "Çankaya Mahallesi Atatürk Caddesi No 123"
    processed = preprocess(text)
    normalized = processed["text"]

    result = matcher.get_best_match(normalized)
    if result:
        pattern_id = result.pattern_id
        print(f"✅ Eşleşme: {pattern_id} (confidence: {result.confidence:.3f})")
        print(
            f"🎯 Başlangıç threshold: {matcher.threshold_manager.get_threshold(pattern_id):.4f}"  # noqa: E501
        )

        # Çok sayıda başarılı feedback
        print("\n📈 5 başarılı feedback...")
        for i in range(5):
            matcher.provide_feedback(pattern_id, True)
            threshold = matcher.threshold_manager.get_threshold(pattern_id)
            print(f"  {i + 1}. feedback sonrası: {threshold:.4f}")

        print("\n📉 5 başarısız feedback...")
        for i in range(5):
            matcher.provide_feedback(pattern_id, False)
            threshold = matcher.threshold_manager.get_threshold(pattern_id)
            print(f"  {i + 1}. feedback sonrası: {threshold:.4f}")

        # Persistence testi
        print("\n💾 Persistence test...")
        final_threshold = matcher.threshold_manager.get_threshold(pattern_id)

        # Yeni matcher oluştur
        new_matcher = PatternMatcher()
        loaded_threshold = new_matcher.threshold_manager.get_threshold(pattern_id)

        print(f"Kaydedilen threshold: {final_threshold:.4f}")
        print(f"Yüklenen threshold: {loaded_threshold:.4f}")

        if abs(final_threshold - loaded_threshold) < 0.0001:
            print("✅ Persistence başarılı!")
        else:
            print("❌ Persistence başarısız!")

    else:
        print("❌ Eşleşme bulunamadı")


if __name__ == "__main__":
    main()
