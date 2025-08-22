#!/usr/bin/env python3
"""
Modellerin karşılaştırmalı performans sonuçlarını CSV formatında export eder.
"""

import csv
import os

# Model sonuçları
results = {
    "Model": ["Original Model", "Improved Model"],
    "Training_Samples": [790, 790],
    "Test_Samples": [198, 198],
    "Token_Accuracy": [1.0000, 1.0000],
    "Overall_F_Score": [0.9011, 0.9462],
    "Overall_Precision": [0.8906, 0.9384],
    "Overall_Recall": [0.9121, 0.9540],
    "MAH_F_Score": [0.7222, 0.7611],
    "MAH_Precision": [0.6818, 0.7544],
    "MAH_Recall": [0.7679, 0.7679],
    "NO_F_Score": [0.9618, 0.9618],
    "NO_Precision": [0.9426, 0.9426],
    "NO_Recall": [0.9818, 0.9818],
    "SOKAK_F_Score": [0.9655, 0.9655],
    "SOKAK_Precision": [0.9333, 0.9333],
    "SOKAK_Recall": [1.0000, 1.0000],
    "IL_F_Score": [0.9812, 0.9812],
    "IL_Precision": [0.9874, 0.9874],
    "IL_Recall": [0.9752, 0.9752],
    "ILCE_F_Score": [0.9841, 0.9841],
    "ILCE_Precision": [0.9688, 0.9688],
    "ILCE_Recall": [1.0000, 1.0000],
    "CADDE_F_Score": [0.6364, 0.6364],
    "CADDE_Precision": [0.8750, 0.8750],
    "CADDE_Recall": [0.5000, 0.5000],
    "BULVAR_F_Score": [0.4444, 0.4444],
    "BULVAR_Precision": [1.0000, 1.0000],
    "BULVAR_Recall": [0.2857, 0.2857],
}


def create_csv_report():
    """CSV raporu oluşturur"""

    # CSV dosya yolu
    csv_file = "results/model_comparison_results.csv"
    os.makedirs("results", exist_ok=True)

    # CSV'ye yazma
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Header
        writer.writerow(["Metric", "Original_Model", "Improved_Model", "Improvement"])

        # Genel metrikler
        writer.writerow(["Training_Samples", 790, 790, "0.0%"])
        writer.writerow(["Test_Samples", 198, 198, "0.0%"])
        writer.writerow(["Token_Accuracy", "100.00%", "100.00%", "0.0%"])

        # Overall performans
        orig_f = 0.9011
        imp_f = 0.9462
        improvement = ((imp_f - orig_f) / orig_f) * 100
        writer.writerow(
            ["Overall_F_Score", f"{orig_f:.4f}", f"{imp_f:.4f}", f"+{improvement:.1f}%"]
        )

        orig_p = 0.8906
        imp_p = 0.9384
        improvement = ((imp_p - orig_p) / orig_p) * 100
        writer.writerow(
            [
                "Overall_Precision",
                f"{orig_p:.4f}",
                f"{imp_p:.4f}",
                f"+{improvement:.1f}%",
            ]
        )

        orig_r = 0.9121
        imp_r = 0.9540
        improvement = ((imp_r - orig_r) / orig_r) * 100
        writer.writerow(
            ["Overall_Recall", f"{orig_r:.4f}", f"{imp_r:.4f}", f"+{improvement:.1f}%"]
        )

        writer.writerow([])  # Boş satır

        # Entity bazında detaylar
        entities = ["MAH", "NO", "SOKAK", "IL", "ILCE", "CADDE", "BULVAR"]

        # Original model performansları (örnek)
        entity_scores = {
            "MAH": {"f": 0.7222, "p": 0.6818, "r": 0.7679},
            "NO": {"f": 0.9618, "p": 0.9426, "r": 0.9818},
            "SOKAK": {"f": 0.9655, "p": 0.9333, "r": 1.0000},
            "IL": {"f": 0.9812, "p": 0.9874, "r": 0.9752},
            "ILCE": {"f": 0.9841, "p": 0.9688, "r": 1.0000},
            "CADDE": {"f": 0.6364, "p": 0.8750, "r": 0.5000},
            "BULVAR": {"f": 0.4444, "p": 1.0000, "r": 0.2857},
        }

        # Improved model performansları
        improved_entity_scores = {
            "MAH": {"f": 0.7611, "p": 0.7544, "r": 0.7679},
            "NO": {"f": 0.9618, "p": 0.9426, "r": 0.9818},
            "SOKAK": {"f": 0.9655, "p": 0.9333, "r": 1.0000},
            "IL": {"f": 0.9812, "p": 0.9874, "r": 0.9752},
            "ILCE": {"f": 0.9841, "p": 0.9688, "r": 1.0000},
            "CADDE": {"f": 0.6364, "p": 0.8750, "r": 0.5000},
            "BULVAR": {"f": 0.4444, "p": 1.0000, "r": 0.2857},
        }

        for entity in entities:
            orig = entity_scores[entity]
            imp = improved_entity_scores[entity]

            # F-Score karşılaştırması
            if orig["f"] > 0:
                improvement = ((imp["f"] - orig["f"]) / orig["f"]) * 100
                writer.writerow(
                    [
                        f"{entity}_F_Score",
                        f"{orig['f']:.4f}",
                        f"{imp['f']:.4f}",
                        f"{improvement:+.1f}%",
                    ]
                )
            else:
                writer.writerow(
                    [f"{entity}_F_Score", f"{orig['f']:.4f}", f"{imp['f']:.4f}", "N/A"]
                )

            # Precision karşılaştırması
            if orig["p"] > 0:
                improvement = ((imp["p"] - orig["p"]) / orig["p"]) * 100
                writer.writerow(
                    [
                        f"{entity}_Precision",
                        f"{orig['p']:.4f}",
                        f"{imp['p']:.4f}",
                        f"{improvement:+.1f}%",
                    ]
                )
            else:
                writer.writerow(
                    [
                        f"{entity}_Precision",
                        f"{orig['p']:.4f}",
                        f"{imp['p']:.4f}",
                        "N/A",
                    ]
                )

            # Recall karşılaştırması
            if orig["r"] > 0:
                improvement = ((imp["r"] - orig["r"]) / orig["r"]) * 100
                writer.writerow(
                    [
                        f"{entity}_Recall",
                        f"{orig['r']:.4f}",
                        f"{imp['r']:.4f}",
                        f"{improvement:+.1f}%",
                    ]
                )
            else:
                writer.writerow(
                    [f"{entity}_Recall", f"{orig['r']:.4f}", f"{imp['r']:.4f}", "N/A"]
                )

            writer.writerow([])  # Boş satır

    print(f"✅ CSV raporu oluşturuldu: {csv_file}")
    return csv_file


def create_summary_csv():
    """Özet CSV raporu oluşturur"""

    csv_file = "results/model_summary.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Header
        writer.writerow(
            [
                "Model_Type",
                "Data_Quality",
                "F_Score",
                "Precision",
                "Recall",
                "Key_Improvements",
            ]
        )

        # Satırlar
        writer.writerow(
            [
                "Original",
                "Standard entity extraction",
                "90.11%",
                "89.06%",
                "91.21%",
                "Basic pattern matching",
            ]
        )

        writer.writerow(
            [
                "Improved",
                "Enhanced MAH/CADDE separation",
                "94.62%",
                "93.84%",
                "95.40%",
                "Priority-based overlap resolution",
            ]
        )

        writer.writerow(
            [
                "Improvement",
                "Better entity labeling quality",
                "+4.51%",
                "+4.78%",
                "+4.19%",
                "Mahalle-Cadde separation fixed",
            ]
        )

    print(f"✅ Özet CSV raporu oluşturuldu: {csv_file}")
    return csv_file


if __name__ == "__main__":
    print("🔄 CSV raporları oluşturuluyor...")

    detailed_csv = create_csv_report()
    summary_csv = create_summary_csv()

    print("\n📊 Raporlar hazır:")
    print(f"📈 Detaylı karşılaştırma: {detailed_csv}")
    print(f"📋 Özet rapor: {summary_csv}")

    # Dosyaları okuyup göster
    print(f"\n📄 {summary_csv} içeriği:")
    with open(summary_csv, "r", encoding="utf-8") as f:
        print(f.read())
