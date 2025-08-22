#!/usr/bin/env python3
"""
Turkish Address NER Model Demo - Modelin çıktılarını canlı olarak gösterir
"""

import argparse
import os
import sys

import spacy


def load_model(model_path):
    """Model yükleme"""
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return None

    try:
        nlp = spacy.load(model_path)
        print(f"✅ Model yüklendi: {model_path}")
        return nlp
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return None


def predict_entities(nlp, text):
    """Metin için entity tahmin eder"""
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, "score", "N/A"),
            }
        )

    return entities


def display_results(text, entities):
    """Sonuçları görsel olarak gösterir"""
    print(f"\n📝 Giriş Metni: '{text}'")
    print("=" * 60)

    if not entities:
        print("❌ Hiç entity bulunamadı!")
        return

    print("🎯 Tespit Edilen Entity'ler:")
    print("-" * 40)

    for i, ent in enumerate(entities, 1):
        print(f"{i:2d}. '{ent['text']}' → {ent['label']}")
        print(f"     Pozisyon: {ent['start']}-{ent['end']}")
        if ent["confidence"] != "N/A":
            print(f"     Güven: {ent['confidence']:.3f}")
        print()


def interactive_demo(nlp):
    """Interaktif demo"""
    print("\n🚀 Turkish Address NER Model Demo")
    print("=" * 50)
    print("Türkçe adres girin (çıkmak için 'quit' yazın)")
    print("-" * 50)

    while True:
        try:
            text = input("\n📍 Adres: ").strip()

            if text.lower() in ["quit", "exit", "q", "çık"]:
                print("👋 Demo sonlandırıldı!")
                break

            if not text:
                print("⚠️ Lütfen bir adres girin!")
                continue

            entities = predict_entities(nlp, text)
            display_results(text, entities)

        except KeyboardInterrupt:
            print("\n\n👋 Demo sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")


def batch_demo(nlp):
    """Önceden tanımlı örneklerle demo"""

    test_addresses = [
        "atatürk mahallesi cumhuriyet caddesi no 15 kat 3 daire 5 ankara",
        "konak mahallesi alsancak bulvarı numara 23 izmir",
        "bağdat caddesi no 125 kadıköy istanbul",
        "güzelyurt mahalle güzelyurt mahalle anadolu caddesi no 42 izmir",
        "fatih mahalle bergama cadde numara 18 daire9 efeler aydın",
        "kazımdirik mahalle 161 sokak özgen apartman no 10 daire 1 bornova",
        "inönü mahalle 1005 sokak numara 5 daire2 kat3 buca",
    ]

    print("\n🎯 Önceden Tanımlı Test Adresleri")
    print("=" * 60)

    for i, address in enumerate(test_addresses, 1):
        print(f"\n【 Test {i} 】")
        entities = predict_entities(nlp, address)
        display_results(address, entities)

        # Kısa bekleme
        input("Devam etmek için Enter'a basın...")


def main():
    parser = argparse.ArgumentParser(description="Turkish Address NER Model Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/turkish_address_ner_improved",
        help="Model dizini (default: models/turkish_address_ner_improved)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Demo modu (default: interactive)",
    )

    args = parser.parse_args()

    # Model yükle
    nlp = load_model(args.model)
    if nlp is None:
        sys.exit(1)

    # Entity labels bilgisi
    labels = nlp.get_pipe("ner").labels
    print(f"📋 Desteklenen Entity'ler: {', '.join(labels)}")

    # Demo moduna göre çalıştır
    if args.mode == "interactive":
        interactive_demo(nlp)
    else:
        batch_demo(nlp)


if __name__ == "__main__":
    main()
