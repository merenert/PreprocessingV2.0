"""
Comprehensive address samples for testing and benchmarking

Contains various categories of Turkish addresses including:
- Standard residential addresses
- Commercial addresses
- Landmark-based addresses
- Edge cases and problematic inputs
- Performance testing samples
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import random


@dataclass
class AddressSample:
    """Single address sample with expected results"""

    input_address: str
    expected_components: Dict[str, str]
    difficulty_level: str  # easy, medium, hard, extreme
    category: str
    expected_confidence_min: float = 0.0
    expected_success: bool = True
    notes: str = ""


# Standard residential addresses
RESIDENTIAL_ADDRESSES = [
    AddressSample(
        input_address="İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15 Daire: 3",
        expected_components={
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15",
            "daire_no": "3",
        },
        difficulty_level="easy",
        category="residential",
        expected_confidence_min=0.85,
        notes="Standard full address format",
    ),
    AddressSample(
        input_address="Ankara Çankaya Çukurambar Mahallesi Dumlupınar Bulvarı 234/A",
        expected_components={
            "il": "Ankara",
            "ilce": "Çankaya",
            "mahalle": "Çukurambar",
            "yol": "Dumlupınar Bulvarı",
            "bina_no": "234/A",
        },
        difficulty_level="easy",
        category="residential",
        expected_confidence_min=0.80,
        notes="Bulvar with apartment designation",
    ),
    AddressSample(
        input_address="İzmir Alsancak Cumhuriyet Bulvarı No: 45 Kat: 2",
        expected_components={
            "il": "İzmir",
            "ilce": "Konak",
            "mahalle": "Alsancak",
            "yol": "Cumhuriyet Bulvarı",
            "bina_no": "45",
            "kat": "2",
        },
        difficulty_level="medium",
        category="residential",
        expected_confidence_min=0.75,
        notes="Missing ilçe, needs inference",
    ),
    AddressSample(
        input_address="Bursa Nilüfer Özlüce Mahallesi Akpınar Caddesi 67/B No: 12",
        expected_components={
            "il": "Bursa",
            "ilce": "Nilüfer",
            "mahalle": "Özlüce",
            "yol": "Akpınar Caddesi",
            "bina_no": "67/B",
            "daire_no": "12",
        },
        difficulty_level="medium",
        category="residential",
        expected_confidence_min=0.70,
        notes="Complex numbering format",
    ),
    AddressSample(
        input_address="Antalya Muratpaşa Lara Mahallesi Barış Manço Bulvarı 156 Site: Palmiye Residans Blok: C Daire: 34",
        expected_components={
            "il": "Antalya",
            "ilce": "Muratpaşa",
            "mahalle": "Lara",
            "yol": "Barış Manço Bulvarı",
            "bina_no": "156",
            "site": "Palmiye Residans",
            "blok": "C",
            "daire_no": "34",
        },
        difficulty_level="medium",
        category="residential",
        expected_confidence_min=0.75,
        notes="Site complex with detailed structure",
    ),
]

# Commercial addresses
COMMERCIAL_ADDRESSES = [
    AddressSample(
        input_address="İstanbul Beyoğlu Galatasaray Mahallesi İstiklal Caddesi No: 24 Galatasaray Lisesi karşısı",
        expected_components={
            "il": "İstanbul",
            "ilce": "Beyoğlu",
            "mahalle": "Galatasaray",
            "yol": "İstiklal Caddesi",
            "bina_no": "24",
            "landmark": "Galatasaray Lisesi",
        },
        difficulty_level="medium",
        category="commercial",
        expected_confidence_min=0.70,
        notes="Commercial with landmark reference",
    ),
    AddressSample(
        input_address="Ankara Kızılay Meşrutiyet Caddesi Kızılay AVM 3. Kat Mağaza No: 45",
        expected_components={
            "il": "Ankara",
            "ilce": "Çankaya",
            "mahalle": "Kızılay",
            "yol": "Meşrutiyet Caddesi",
            "magaza": "Kızılay AVM",
            "kat": "3",
            "daire_no": "45",
        },
        difficulty_level="medium",
        category="commercial",
        expected_confidence_min=0.65,
        notes="Shopping mall internal address",
    ),
    AddressSample(
        input_address="İzmir Konak Alsancak Kıbrıs Şehitleri Caddesi No: 78 Büro Plaza Kat: 5 Ofis: 12",
        expected_components={
            "il": "İzmir",
            "ilce": "Konak",
            "mahalle": "Alsancak",
            "yol": "Kıbrıs Şehitleri Caddesi",
            "bina_no": "78",
            "bina_adi": "Büro Plaza",
            "kat": "5",
            "daire_no": "12",
        },
        difficulty_level="medium",
        category="commercial",
        expected_confidence_min=0.70,
        notes="Office building address",
    ),
]

# Landmark-based addresses
LANDMARK_ADDRESSES = [
    AddressSample(
        input_address="Amorium Hotel karşısı",
        expected_components={"landmark": "Amorium Hotel", "spatial_relation": "karşısı"},
        difficulty_level="hard",
        category="landmark",
        expected_confidence_min=0.40,
        notes="Landmark-only address requiring geocoding",
    ),
    AddressSample(
        input_address="McDonald's yanı yeşil bina",
        expected_components={"landmark": "McDonald's", "spatial_relation": "yanı", "description": "yeşil bina"},
        difficulty_level="hard",
        category="landmark",
        expected_confidence_min=0.30,
        notes="Chain restaurant with description",
    ),
    AddressSample(
        input_address="İstanbul Üniversitesi Rektörlük binası arkası",
        expected_components={"landmark": "İstanbul Üniversitesi Rektörlük binası", "spatial_relation": "arkası"},
        difficulty_level="hard",
        category="landmark",
        expected_confidence_min=0.50,
        notes="University landmark with specific building",
    ),
    AddressSample(
        input_address="Taksim Meydanı otopark girişi",
        expected_components={"landmark": "Taksim Meydanı", "description": "otopark girişi"},
        difficulty_level="medium",
        category="landmark",
        expected_confidence_min=0.60,
        notes="Famous square with functional description",
    ),
    AddressSample(
        input_address="Beşiktaş İskelesi 50 metre İnönü Stadyumu yönü",
        expected_components={"landmark": "Beşiktaş İskelesi", "distance": "50 metre", "direction": "İnönü Stadyumu yönü"},
        difficulty_level="hard",
        category="landmark",
        expected_confidence_min=0.45,
        notes="Distance and direction specification",
    ),
]

# Edge cases and problematic inputs
EDGE_CASES = [
    AddressSample(
        input_address="",
        expected_components={},
        difficulty_level="extreme",
        category="edge_case",
        expected_success=False,
        notes="Empty input",
    ),
    AddressSample(
        input_address="xyz123 invalid address format !@#",
        expected_components={},
        difficulty_level="extreme",
        category="edge_case",
        expected_success=False,
        notes="Complete gibberish",
    ),
    AddressSample(
        input_address="İstanbul İstanbul İstanbul Kadıköy Kadıköy",
        expected_components={"il": "İstanbul", "ilce": "Kadıköy"},
        difficulty_level="hard",
        category="edge_case",
        expected_confidence_min=0.30,
        notes="Repetitive terms",
    ),
    AddressSample(
        input_address="ISTANBUL KADIKOY MODA MAHALLESİ BAHARIYE CADDESİ NO 15",
        expected_components={
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15",
        },
        difficulty_level="medium",
        category="edge_case",
        expected_confidence_min=0.60,
        notes="All uppercase without Turkish characters",
    ),
    AddressSample(
        input_address="istanbul kadıköy moda mahallesi bahariye caddesi no 15",
        expected_components={
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15",
        },
        difficulty_level="medium",
        category="edge_case",
        expected_confidence_min=0.60,
        notes="All lowercase",
    ),
    AddressSample(
        input_address="İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15/A-B Kat: 3 Daire: 45 Kapı No: 678",
        expected_components={
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "yol": "Bahariye Caddesi",
            "bina_no": "15/A-B",
            "kat": "3",
            "daire_no": "45",
            "kapi_no": "678",
        },
        difficulty_level="hard",
        category="edge_case",
        expected_confidence_min=0.50,
        notes="Overly complex numbering",
    ),
    AddressSample(
        input_address="123 Main Street, New York, NY 10001, USA",
        expected_components={},
        difficulty_level="extreme",
        category="edge_case",
        expected_success=False,
        notes="Foreign address format",
    ),
    AddressSample(
        input_address="İstanbul Avrupa Yakası",
        expected_components={"il": "İstanbul", "region": "Avrupa Yakası"},
        difficulty_level="hard",
        category="edge_case",
        expected_confidence_min=0.25,
        notes="Very general geographic reference",
    ),
]

# Rural and small town addresses
RURAL_ADDRESSES = [
    AddressSample(
        input_address="Muğla Fethiye Kayaköy Mahallesi Eski Yunan Evleri Sokağı No: 5",
        expected_components={
            "il": "Muğla",
            "ilce": "Fethiye",
            "mahalle": "Kayaköy",
            "yol": "Eski Yunan Evleri Sokağı",
            "bina_no": "5",
        },
        difficulty_level="medium",
        category="rural",
        expected_confidence_min=0.70,
        notes="Rural tourist area",
    ),
    AddressSample(
        input_address="Bolu Abant Gölü yakını Çam Hotel",
        expected_components={"il": "Bolu", "landmark": "Abant Gölü", "spatial_relation": "yakını", "bina_adi": "Çam Hotel"},
        difficulty_level="hard",
        category="rural",
        expected_confidence_min=0.45,
        notes="Nature-based landmark",
    ),
    AddressSample(
        input_address="Çanakkale Gelibolu Şehitlik yolu üzeri 3. km",
        expected_components={"il": "Çanakkale", "ilce": "Gelibolu", "yol": "Şehitlik yolu", "mesafe": "3. km"},
        difficulty_level="hard",
        category="rural",
        expected_confidence_min=0.50,
        notes="Distance-based address on highway",
    ),
]

# Historical and special addresses
HISTORICAL_ADDRESSES = [
    AddressSample(
        input_address="İstanbul Fatih Sultanahmet Ayasofya Müzesi",
        expected_components={"il": "İstanbul", "ilce": "Fatih", "mahalle": "Sultanahmet", "landmark": "Ayasofya Müzesi"},
        difficulty_level="easy",
        category="historical",
        expected_confidence_min=0.85,
        notes="Famous historical landmark",
    ),
    AddressSample(
        input_address="Ankara Altındağ Ulus Anıtkabir girişi",
        expected_components={
            "il": "Ankara",
            "ilce": "Altındağ",
            "mahalle": "Ulus",
            "landmark": "Anıtkabir",
            "description": "girişi",
        },
        difficulty_level="medium",
        category="historical",
        expected_confidence_min=0.75,
        notes="National monument",
    ),
    AddressSample(
        input_address="İzmir Konak Alsancak Saat Kulesi meydanı",
        expected_components={
            "il": "İzmir",
            "ilce": "Konak",
            "mahalle": "Alsancak",
            "landmark": "Saat Kulesi",
            "description": "meydanı",
        },
        difficulty_level="medium",
        category="historical",
        expected_confidence_min=0.70,
        notes="Historic clock tower square",
    ),
]

# Performance testing samples (for benchmarks)
PERFORMANCE_SAMPLES = []


# Generate additional samples for performance testing
def generate_performance_samples(count: int = 1000) -> List[AddressSample]:
    """Generate large number of samples for performance testing"""

    # Templates for generation
    templates = [
        "{il} {ilce} {mahalle} Mahallesi {sokak} Sokağı No: {no}",
        "{il} {ilce} {mahalle} {cadde} Caddesi {no}/{daire}",
        "{il} {ilce} {mahalle} Mahallesi {bulvar} Bulvarı No: {no} Kat: {kat}",
        "{il} {ilce} {mahalle} {sokak} Sokağı {no} Daire: {daire}",
    ]

    # Sample components
    iller = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep"]
    ilceler = {
        "İstanbul": ["Kadıköy", "Beyoğlu", "Şişli", "Beşiktaş", "Üsküdar", "Fatih"],
        "Ankara": ["Çankaya", "Keçiören", "Yenimahalle", "Mamak", "Altındağ"],
        "İzmir": ["Konak", "Karşıyaka", "Bornova", "Alsancak", "Bayraklı"],
        "Bursa": ["Nilüfer", "Osmangazi", "Yıldırım", "Mudanya"],
        "Antalya": ["Muratpaşa", "Konyaaltı", "Kepez", "Aksu"],
    }

    mahalleler = ["Merkez", "Yeni", "Eski", "Güney", "Kuzey", "Doğu", "Batı", "Çamlık", "Gültepe"]
    sokaklar = ["Atatürk", "İnönü", "Cumhuriyet", "Barış", "Hürriyet", "Gazi", "Mimar Sinan"]
    caddeler = ["Atatürk", "Cumhuriyet", "İnönü", "Gazi", "Mimar Sinan", "Barbaros"]
    bulvarlar = ["Atatürk", "Cumhuriyet", "Dumlupınar", "Barış Manço", "Turgut Özal"]

    samples = []

    for i in range(count):
        template = random.choice(templates)
        il = random.choice(iller)
        ilce = random.choice(ilceler.get(il, ["Merkez"]))
        mahalle = random.choice(mahalleler)

        # Generate address based on template
        if "sokak" in template:
            sokak = random.choice(sokaklar)
            address = template.format(
                il=il,
                ilce=ilce,
                mahalle=mahalle,
                sokak=sokak,
                no=random.randint(1, 200),
                daire=random.randint(1, 50),
                kat=random.randint(1, 10),
            )
        elif "cadde" in template:
            cadde = random.choice(caddeler)
            address = template.format(
                il=il, ilce=ilce, mahalle=mahalle, cadde=cadde, no=random.randint(1, 200), daire=random.randint(1, 50)
            )
        else:  # bulvar
            bulvar = random.choice(bulvarlar)
            address = template.format(
                il=il, ilce=ilce, mahalle=mahalle, bulvar=bulvar, no=random.randint(1, 200), kat=random.randint(1, 10)
            )

        sample = AddressSample(
            input_address=address,
            expected_components={"il": il, "ilce": ilce, "mahalle": mahalle},
            difficulty_level="easy",
            category="generated",
            expected_confidence_min=0.60,
        )
        samples.append(sample)

    return samples


# Combine all sample categories
def get_test_addresses() -> List[AddressSample]:
    """Get standard test addresses for regular testing"""
    return RESIDENTIAL_ADDRESSES[:10] + COMMERCIAL_ADDRESSES[:5] + LANDMARK_ADDRESSES[:5]


def get_edge_case_data() -> List[AddressSample]:
    """Get edge case test data for comprehensive testing"""
    return EDGE_CASES


def get_all_samples() -> List[AddressSample]:
    """Get all address samples"""
    return (
        RESIDENTIAL_ADDRESSES + COMMERCIAL_ADDRESSES + LANDMARK_ADDRESSES + EDGE_CASES + RURAL_ADDRESSES + HISTORICAL_ADDRESSES
    )


def get_samples_by_category(category: str) -> List[AddressSample]:
    """Get samples by category"""
    all_samples = get_all_samples()
    return [s for s in all_samples if s.category == category]


def get_samples_by_difficulty(difficulty: str) -> List[AddressSample]:
    """Get samples by difficulty level"""
    all_samples = get_all_samples()
    return [s for s in all_samples if s.difficulty_level == difficulty]


def get_performance_test_samples(count: int = 1000) -> List[AddressSample]:
    """Get samples for performance testing"""
    if not PERFORMANCE_SAMPLES:
        PERFORMANCE_SAMPLES.extend(generate_performance_samples(count))
    return PERFORMANCE_SAMPLES[:count]


# Sample statistics
def get_sample_statistics() -> Dict[str, Any]:
    """Get statistics about available samples"""
    all_samples = get_all_samples()

    stats = {"total_samples": len(all_samples), "by_category": {}, "by_difficulty": {}, "expected_success_rate": 0}

    # Count by category
    for sample in all_samples:
        category = sample.category
        difficulty = sample.difficulty_level

        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1

    # Calculate expected success rate
    success_count = sum(1 for s in all_samples if s.expected_success)
    stats["expected_success_rate"] = success_count / len(all_samples) if all_samples else 0

    return stats


if __name__ == "__main__":
    # Print sample statistics
    stats = get_sample_statistics()
    print("📊 Address Sample Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Expected success rate: {stats['expected_success_rate']:.2%}")

    print(f"\n📂 By Category:")
    for category, count in stats["by_category"].items():
        print(f"  {category}: {count}")

    print(f"\n⚡ By Difficulty:")
    for difficulty, count in stats["by_difficulty"].items():
        print(f"  {difficulty}: {count}")

    # Generate performance samples
    perf_samples = get_performance_test_samples(100)
    print(f"\n🚀 Generated {len(perf_samples)} performance test samples")
