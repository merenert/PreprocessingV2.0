"""
Golden test examples for Turkish address normalization.
Realistic test cases based on actual system behavior.
"""

GOLDEN_TEST_CASES = [
    # Basic functionality tests
    {
        "id": "basic_001",
        "input": "İstanbul",
        "expected": {
            "normalized_address": "İstanbul",
        },
        "category": "basic",
        "description": "City only",
    },
    {
        "id": "basic_002",
        "input": "Ankara",
        "expected": {
            "normalized_address": "Ankara",
        },
        "category": "basic",
        "description": "Capital city",
    },
    {
        "id": "basic_003",
        "input": "İzmir",
        "expected": {
            "normalized_address": "İzmir",
        },
        "category": "basic",
        "description": "Coastal city",
    },
    {
        "id": "basic_004",
        "input": "İstanbul Beşiktaş",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş",
        },
        "category": "basic",
        "description": "City and district",
    },
    {
        "id": "basic_005",
        "input": "Ankara Çankaya",
        "expected": {
            "normalized_address": "Ankara Çankaya",
        },
        "category": "basic",
        "description": "Capital district",
    },
    # Case variations
    {
        "id": "case_001",
        "input": "istanbul",
        "expected": {
            "normalized_address": "İstanbul",
        },
        "category": "case",
        "description": "Lowercase city",
    },
    {
        "id": "case_002",
        "input": "ANKARA",
        "expected": {
            "normalized_address": "Ankara",
        },
        "category": "case",
        "description": "Uppercase city",
    },
    {
        "id": "case_003",
        "input": "istanbul beşiktaş",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş",
        },
        "category": "case",
        "description": "Mixed case normalization",
    },
    # Standard addresses
    {
        "id": "standard_001",
        "input": "İstanbul Beşiktaş Levent",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş Levent",
        },
        "category": "standard",
        "description": "City-district-neighborhood",
    },
    {
        "id": "standard_002",
        "input": "Ankara Çankaya Kızılay",
        "expected": {
            "normalized_address": "Ankara Çankaya Kızılay",
        },
        "category": "standard",
        "description": "Well-known area",
    },
    # Street addresses
    {
        "id": "street_001",
        "input": "İstanbul Beşiktaş Büyükdere Caddesi",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş Büyükdere Caddesi",
        },
        "category": "street",
        "description": "Address with street",
    },
    {
        "id": "street_002",
        "input": "Ankara Çankaya Atatürk Bulvarı",
        "expected": {
            "normalized_address": "Ankara Çankaya Atatürk Bulvarı",
        },
        "category": "street",
        "description": "Boulevard address",
    },
    # With numbers
    {
        "id": "number_001",
        "input": "İstanbul Beşiktaş No:100",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş No:100",
        },
        "category": "number",
        "description": "With building number",
    },
    {
        "id": "number_002",
        "input": "Ankara Çankaya 15",
        "expected": {
            "normalized_address": "Ankara Çankaya 15",
        },
        "category": "number",
        "description": "Plain number",
    },
    # Turkish characters
    {
        "id": "turkish_001",
        "input": "İzmir Bornova Ege Üniversitesi",
        "expected": {
            "normalized_address": "İzmir Bornova Ege Üniversitesi",
        },
        "category": "turkish",
        "description": "University with Turkish chars",
    },
    {
        "id": "turkish_002",
        "input": "Bursa Osmangazi Çekirge",
        "expected": {
            "normalized_address": "Bursa Osmangazi Çekirge",
        },
        "category": "turkish",
        "description": "Turkish neighborhood",
    },
    # Complex addresses
    {
        "id": "complex_001",
        "input": "İstanbul Beşiktaş Levent Büyükdere Caddesi No:100",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş Levent Büyükdere Caddesi No:100",
        },
        "category": "complex",
        "description": "Full address components",
    },
    {
        "id": "complex_002",
        "input": "Ankara Çankaya Gaziosmanpaşa Mahallesi Filistin Caddesi",
        "expected": {
            "normalized_address": (
                "Ankara Çankaya Gaziosmanpaşa Mahallesi Filistin Caddesi"
            ),
        },
        "category": "complex",
        "description": "Long formal address",
    },
    # Postal codes
    {
        "id": "postal_001",
        "input": "34394 İstanbul Beşiktaş",
        "expected": {
            "normalized_address": "34394 İstanbul Beşiktaş",
        },
        "category": "postal",
        "description": "With postal code",
    },
    {
        "id": "postal_002",
        "input": "İstanbul Beşiktaş 34394",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş 34394",
        },
        "category": "postal",
        "description": "Postal code at end",
    },
    # Landmarks
    {
        "id": "landmark_001",
        "input": "İstanbul Sultanahmet",
        "expected": {
            "normalized_address": "İstanbul Sultanahmet",
        },
        "category": "landmark",
        "description": "Historic area",
    },
    {
        "id": "landmark_002",
        "input": "İstanbul Galata Kulesi",
        "expected": {
            "normalized_address": "İstanbul Galata Kulesi",
        },
        "category": "landmark",
        "description": "Famous tower",
    },
    # Business addresses
    {
        "id": "business_001",
        "input": "İstanbul Maslak Plaza",
        "expected": {
            "normalized_address": "İstanbul Maslak Plaza",
        },
        "category": "business",
        "description": "Business center",
    },
    {
        "id": "business_002",
        "input": "Ankara Bilkent Üniversitesi",
        "expected": {
            "normalized_address": "Ankara Bilkent Üniversitesi",
        },
        "category": "business",
        "description": "University",
    },
    # Transportation
    {
        "id": "transport_001",
        "input": "İstanbul Atatürk Havalimanı",
        "expected": {
            "normalized_address": "İstanbul Atatürk Havalimanı",
        },
        "category": "transport",
        "description": "Airport",
    },
    {
        "id": "transport_002",
        "input": "Ankara Gar",
        "expected": {
            "normalized_address": "Ankara Gar",
        },
        "category": "transport",
        "description": "Train station",
    },
    # Special formatting
    {
        "id": "format_001",
        "input": "İstanbul / Beşiktaş",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş",
        },
        "category": "format",
        "description": "Special delimiters",
    },
    {
        "id": "format_002",
        "input": "Ankara - Çankaya",
        "expected": {
            "normalized_address": "Ankara Çankaya",
        },
        "category": "format",
        "description": "Dash separator",
    },
    # Medical facilities
    {
        "id": "medical_001",
        "input": "İstanbul Şişli Hastanesi",
        "expected": {
            "normalized_address": "İstanbul Şişli Hastanesi",
        },
        "category": "medical",
        "description": "Hospital",
    },
    {
        "id": "medical_002",
        "input": "Ankara Hacettepe Tıp",
        "expected": {
            "normalized_address": "Ankara Hacettepe Tıp",
        },
        "category": "medical",
        "description": "Medical school",
    },
    # Government
    {
        "id": "government_001",
        "input": "Ankara Belediyesi",
        "expected": {
            "normalized_address": "Ankara Belediyesi",
        },
        "category": "government",
        "description": "Municipality",
    },
    {
        "id": "government_002",
        "input": "İstanbul Valilik",
        "expected": {
            "normalized_address": "İstanbul Valilik",
        },
        "category": "government",
        "description": "Governor office",
    },
    # Sports
    {
        "id": "sports_001",
        "input": "İstanbul Beşiktaş Stadyumu",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş Stadyumu",
        },
        "category": "sports",
        "description": "Football stadium",
    },
    {
        "id": "sports_002",
        "input": "Ankara Spor Kompleksi",
        "expected": {
            "normalized_address": "Ankara Spor Kompleksi",
        },
        "category": "sports",
        "description": "Sports complex",
    },
    # Shopping
    {
        "id": "shopping_001",
        "input": "İstanbul Şişli AVM",
        "expected": {
            "normalized_address": "İstanbul Şişli AVM",
        },
        "category": "shopping",
        "description": "Shopping mall",
    },
    {
        "id": "shopping_002",
        "input": "Ankara Çayyolu Armada",
        "expected": {
            "normalized_address": "Ankara Çayyolu Armada",
        },
        "category": "shopping",
        "description": "Mall complex",
    },
    # Cultural
    {
        "id": "cultural_001",
        "input": "İstanbul Beyoğlu Tiyatro",
        "expected": {
            "normalized_address": "İstanbul Beyoğlu Tiyatro",
        },
        "category": "cultural",
        "description": "Theater",
    },
    {
        "id": "cultural_002",
        "input": "Ankara Opera",
        "expected": {
            "normalized_address": "Ankara Opera",
        },
        "category": "cultural",
        "description": "Opera house",
    },
    # Mixed content
    {
        "id": "mixed_001",
        "input": "İstanbul Kadıköy 19 Mayıs",
        "expected": {
            "normalized_address": "İstanbul Kadıköy 19 Mayıs",
        },
        "category": "mixed",
        "description": "Historical date",
    },
    {
        "id": "mixed_002",
        "input": "Ankara Bahçelievler 7. Cadde",
        "expected": {
            "normalized_address": "Ankara Bahçelievler 7. Cadde",
        },
        "category": "mixed",
        "description": "Numbered street",
    },
    # Abbreviations (realistic)
    {
        "id": "abbrev_001",
        "input": "İst. Beş.",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş",
        },
        "category": "abbrev",
        "description": "Common abbreviations",
    },
    {
        "id": "abbrev_002",
        "input": "Ank. Çnk.",
        "expected": {
            "normalized_address": "Ankara Çankaya",
        },
        "category": "abbrev",
        "description": "District abbreviations",
    },
    # Typos and variations
    {
        "id": "typo_001",
        "input": "Istambul Besiktas",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş",
        },
        "category": "typo",
        "description": "Missing Turkish chars",
    },
    {
        "id": "typo_002",
        "input": "Ankaara Cankayaa",
        "expected": {
            "normalized_address": "Ankara Çankaya",
        },
        "category": "typo",
        "description": "Extra letters",
    },
    # Formal designations
    {
        "id": "formal_001",
        "input": "İstanbul İli Beşiktaş İlçesi",
        "expected": {
            "normalized_address": "İstanbul Beşiktaş",
        },
        "category": "formal",
        "description": "Province/district formal",
    },
    {
        "id": "formal_002",
        "input": "Ankara İli Çankaya İlçesi",
        "expected": {
            "normalized_address": "Ankara Çankaya",
        },
        "category": "formal",
        "description": "Official designation",
    },
    # Residential
    {
        "id": "residential_001",
        "input": "İstanbul Ataşehir Residences",
        "expected": {
            "normalized_address": "İstanbul Ataşehir Residences",
        },
        "category": "residential",
        "description": "Housing complex",
    },
    {
        "id": "residential_002",
        "input": "Ankara Çayyolu Sitesi",
        "expected": {
            "normalized_address": "Ankara Çayyolu Sitesi",
        },
        "category": "residential",
        "description": "Residential site",
    },
    # Final comprehensive test
    {
        "id": "comprehensive_001",
        "input": (
            "34394 İstanbul İli Beşiktaş İlçesi Levent Mahallesi "
            "Büyükdere Caddesi No:100"
        ),
        "expected": {
            "normalized_address": (
                "34394 İstanbul Beşiktaş Levent Büyükdere Caddesi No:100"
            ),
        },
        "category": "comprehensive",
        "description": "Full formal address",
    },
]
