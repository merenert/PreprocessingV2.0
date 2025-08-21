# Address Normalizer (addrnorm)

> Türkçe adresleri normalize eden, modüler Python monorepo.

## Mimarî (Mermaid)
```mermaid
graph TD
    A[raw_addresses_tr.txt] -->|Preprocess| B(addrnorm.preprocess)
    B --> C(addrnorm.patterns)
    C --> D(addrnorm.ml)
    D --> E(addrnorm.fallback)
    E --> F(addrnorm.validate)
    F --> G(addrnorm.pipeline)
    G --> H(addrnorm.api)
    H --> I[output.address.json]
    G --> J(addrnorm.explainer)
    G --> K(addrnorm.utils)
```

## Hızlı Başlangıç
```bash
# Kurulum
make install

# Test
make test

# API'yi başlat
make run-api
```

## Örnek CLI Komutu
```bash
python -m src.addrnorm.preprocess.cli --input data/examples/raw_addresses_tr.txt --output data/examples/normalized.json
```

## Çıktı Şeması
- Bkz: `schemas/output.address.json`

## Lisans
MIT
