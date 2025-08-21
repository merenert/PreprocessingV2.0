# CSV Sözleşmesi - Normalize Adres Çıktısı

## Genel Kurallar
- **Kodlama**: UTF-8 (BOM olmadan)
- **Ayırıcı**: `,` (virgül)
- **Satır sonu**: `\n` (LF)
- **Header**: Zorunlu (ilk satır)
- **Format**: Tek satır = tek adres

## Alan Sırası ve Tanımları

| Sıra | Alan Adı | Tip | Açıklama | Örnek |
|------|----------|-----|----------|-------|
| 1 | country | string/null | ISO 3166-1 alpha-2 ülke kodu | TR |
| 2 | city | string/null | İl adı | İstanbul |
| 3 | district | string/null | İlçe adı | Kadıköy |
| 4 | neighborhood | string/null | Mahalle adı | Moda Mah. |
| 5 | street | string/null | Sokak/Cadde adı | Bahariye Cad. |
| 6 | building | string/null | Bina adı | Moda Plaza |
| 7 | block | string/null | Blok bilgisi | A |
| 8 | number | string/null | Kapı/Dış kapı no | 12 |
| 9 | entrance | string/null | Giriş | 2 |
| 10 | floor | string/null | Kat | 5 |
| 11 | apartment | string/null | Daire | 15 |
| 12 | postcode | string/null | Posta kodu (5 haneli) | 34710 |
| 13 | relation | string/null | Mekansal ilişki | karsisi |
| 14 | explanation_raw | string | Ham giriş metni | "İstanbul Kadıköy..." |
| 15 | normalized_address | string | Normalize adres | "TR, İstanbul, Kadıköy..." |
| 16 | confidence | number | Güven skoru (0-1) | 0.95 |
| 17 | method | string | Yöntem | ml |
| 18 | warnings | string | Uyarılar (JSON array string) | "[]" |

## Null Değerler
- Boş alanlar: boş string olarak yazılır
- `null` değerler: boş string olarak yazılır

## Özel Karakterler
- Virgül içeren değerler: çift tırnak içinde
- Çift tırnak içeren değerler: `""` olarak escape edilir
- Satır sonu karakterleri: metinden temizlenir

## Örnek CSV

```csv
country,city,district,neighborhood,street,building,block,number,entrance,floor,apartment,postcode,relation,explanation_raw,normalized_address,confidence,method,warnings
TR,İstanbul,Kadıköy,Moda Mah.,Bahariye Cad.,,A,12,2,5,15,34710,,"İstanbul, Kadıköy, Moda Mah., Bahariye Cad. No:12 D:5","TR, İstanbul, Kadıköy, Moda Mah., Bahariye Cad., A Blok, No:12, Giriş:2, Kat:5, Daire:15, 34710",0.95,ml,"[]"
TR,Ankara,Çankaya,Kızılay Mah.,Atatürk Bulvarı,,,100,,,,,,"Ankara, Çankaya, Kızılay Mah., Atatürk Bulvarı No:100","TR, Ankara, Çankaya, Kızılay Mah., Atatürk Bulvarı, No:100",0.87,pattern,"[""Posta kodu bulunamadı""]"
```

## Validasyon
- Her satır aynı sayıda alan içermeli
- Zorunlu alanlar: `explanation_raw`, `normalized_address`, `confidence`, `method`
- `confidence` 0-1 arasında olmalı
- `method` enum değerlerinden biri olmalı: ml, pattern, fallback
