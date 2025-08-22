#!/usr/bin/env python3
"""
Dönüştürülmüş NER verilerini görüntüleme aracı.
"""

import json
from pathlib import Path


def show_converted_data(file_path, max_examples=10):
    """Dönüştürülmüş NER verilerini güzel formatta göster."""

    print(f"📊 {file_path} dosyasından örnekler:")
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break

            item = json.loads(line.strip())
            text = item["text"]
            entities = item["entities"]

            print(f"\n{i+1}. ADRES:")
            print(f"   📝 Metin: {text}")
            print(f"   🏷️  Entity sayısı: {len(entities)}")

            # Entity'leri sıralı göster
            if entities:
                print("   📍 Tespit edilen bileşenler:")
                for start, end, label in entities:
                    entity_text = text[start:end]
                    label_tr = {
                        "IL": "İl",
                        "ILCE": "İlçe",
                        "MAH": "Mahalle",
                        "SOKAK": "Sokak",
                        "CADDE": "Cadde",
                        "BULVAR": "Bulvar",
                        "NO": "Numara",
                        "DAIRE": "Daire",
                        "KAT": "Kat",
                    }.get(label, label)

                    print(
                        f"     • '{entity_text}' → {label_tr} ({label}) "
                        f"[pozisyon: {start}-{end}]"
                    )
            else:
                print("   ❌ Hiç entity bulunamadı")


def show_statistics():
    """Veri istatistiklerini göster."""
    train_path = "data/train/train_ner.jsonl"
    test_path = "data/train/test_ner.jsonl"

    # Satır sayılarını say
    train_count = 0
    test_count = 0

    if Path(train_path).exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_count = sum(1 for _ in f)

    if Path(test_path).exists():
        with open(test_path, "r", encoding="utf-8") as f:
            test_count = sum(1 for _ in f)

    # Entity istatistikleri
    entity_counts = {}

    if Path(train_path).exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                for start, end, label in item["entities"]:
                    entity_counts[label] = entity_counts.get(label, 0) + 1

    print("\n📈 VERİ İSTATİSTİKLERİ:")
    print("=" * 40)
    print(f"🎯 Toplam eğitim verisi: {train_count:,} örnek")
    print(f"🎯 Toplam test verisi: {test_count:,} örnek")
    print(f"🎯 Genel toplam: {train_count + test_count:,} örnek")

    print("\n🏷️  ENTITY TİP İSTATİSTİKLERİ:")
    print("-" * 30)
    entity_names = {
        "IL": "İl (Province)",
        "ILCE": "İlçe (District)",
        "MAH": "Mahalle (Neighborhood)",
        "SOKAK": "Sokak (Street)",
        "CADDE": "Cadde (Avenue)",
        "BULVAR": "Bulvar (Boulevard)",
        "NO": "Numara (Number)",
        "DAIRE": "Daire (Apartment)",
        "KAT": "Kat (Floor)",
    }

    for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        name = entity_names.get(label, label)
        print(f"  {name}: {count:,} adet")


if __name__ == "__main__":
    print("🏠 TÜRKÇE ADRES NER VERİLERİ GÖRÜNTÜLEME ARACI")
    print("=" * 60)

    # Eğitim verilerinden örnekler
    train_path = "data/train/train_ner.jsonl"
    if Path(train_path).exists():
        show_converted_data(train_path, 5)

    # Test verilerinden örnekler
    test_path = "data/train/test_ner.jsonl"
    if Path(test_path).exists():
        print("\n" + "=" * 70)
        show_converted_data(test_path, 3)

    # İstatistikler
    show_statistics()
