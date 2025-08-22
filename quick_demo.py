#!/usr/bin/env python3
"""Quick threshold demo."""

import sys

sys.path.append("src")

from addrnorm.patterns.matcher import PatternMatcher


def main():
    """Quick threshold demonstration."""
    matcher = PatternMatcher()
    print(f"Pattern sayısı: {len(matcher.patterns)}")

    # Önce preprocess edelim
    from addrnorm.preprocess import preprocess

    # Basit test adresi
    test_addresses = [
        "Çankaya Mahallesi Atatürk Caddesi No 123",
        "Bahçelievler Mah. Kızılırmak Sok. No:45",
        "ankara çankaya atatürk bulvarı 123",
    ]

    for text in test_addresses:
        processed = preprocess(text)
        normalized = processed["text"]
        print(f"\nOrjinal: {text}")
        print(f"Normalized: {normalized}")

        result = matcher.get_best_match(normalized)
        if result:
            print(
                f"✅ Eşleşme: {result.pattern_id} (confidence: {result.confidence:.3f})"
            )
            break
        else:
            matches = matcher.match_text(normalized, max_matches=3)
            print(f"Toplam {len(matches)} eşleşme")

    # Eşleşme bulundu mu kontrol et
    if "result" in locals() and result:
        print("\n📊 Threshold Demo:")
        print(
            f"Başlangıç threshold: {matcher.threshold_manager.get_threshold(result.pattern_id):.3f}"  # noqa: E501
        )

        # Başarılı feedback
        matcher.provide_feedback(result.pattern_id, True)
        print(
            f"📈 Feedback sonrası: {matcher.threshold_manager.get_threshold(result.pattern_id):.3f}"  # noqa: E501
        )

        # Başarısız feedback
        matcher.provide_feedback(result.pattern_id, False)
        print(
            f"📉 Başarısız feedback sonrası: {matcher.threshold_manager.get_threshold(result.pattern_id):.3f}"  # noqa: E501
        )

        print("\n✅ Dinamik eşik sistemi çalışıyor!")
    else:
        print("\n❌ Hiçbir test adresinde eşleşme bulunamadı")
        print("Pattern'lar çok spesifik olabilir.")


if __name__ == "__main__":
    main()
