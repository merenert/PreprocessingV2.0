#!/usr/bin/env python3
"""
Eski ve yeni algoritma sonuçlarını karşılaştırma.
"""

import json


def compare_algorithms():
    print("🔍 KARŞILAŞTIRMA: Eski vs Yeni Algoritma")
    print("=" * 60)

    # Eski veri örneği
    with open("data/train/train_ner.jsonl", "r", encoding="utf-8") as f:
        old_lines = [json.loads(line) for line in f.readlines()[:3]]

    # Yeni veri örneği
    with open("data/train/train_ner_improved.jsonl", "r", encoding="utf-8") as f:
        new_lines = [json.loads(line) for line in f.readlines()[:3]]

    for i, (old_data, new_data) in enumerate(zip(old_lines, new_lines), 1):
        print(f"\n📊 Örnek {i}:")
        print("-" * 40)

        print("ESKİ ALGORİTMA:")
        print(f'  Text: {old_data["text"]}')
        for start, end, label in old_data["entities"]:
            text_part = old_data["text"][start:end]
            print(f'    "{text_part}" -> {label}')

        print("\nYENİ ALGORİTMA:")
        print(f'  Text: {new_data["text"]}')
        for start, end, label in new_data["entities"]:
            text_part = new_data["text"][start:end]
            print(f'    "{text_part}" -> {label}')

        print()


if __name__ == "__main__":
    compare_algorithms()
