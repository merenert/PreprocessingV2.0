# Turkish Address Explanation Processing - Simple Text Module

## Overview

The `addrnorm.explanation` module provides simple text processing for Turkish address explanations. It focuses on cleaning and validating text without complex parsing.

## 🎯 Module Features

### Simple Text Processing
- Text cleaning and normalization
- Whitespace management
- Input validation
- Backward compatibility

## 🚀 Usage Examples

### Basic Usage
```python
from addrnorm.explanation import process_explanation

# Simple text cleaning
result = process_explanation("Migros yanı")
print(result)  # "Migros yanı"

# Handle whitespace
result = process_explanation("  Amorium Hotel   karşısı  ")
print(result)  # "Amorium Hotel karşısı"
```

### Batch Processing
```python
explanations = [
    "Migros yanı",
    "  McDonald's karşısı  ",
    "Hastane önü",
    "   Okul arkası   "
]

cleaned_explanations = [process_explanation(text) for text in explanations]
for original, cleaned in zip(explanations, cleaned_explanations):
    print(f"'{original}' → '{cleaned}'")
```

### Integration with Main System
```python
from addrnorm import AddressNormalizer
from addrnorm.explanation import process_explanation

normalizer = AddressNormalizer()

def normalize_with_explanation(address_text):
    # Clean explanation text
    explanation_cleaned = process_explanation(address_text)

    # Use cleaned text as explanation_raw
    result = normalizer.normalize(address_text)
    if isinstance(result, dict):
        result["explanation_raw"] = explanation_cleaned

    return result
```

## 🔧 API Reference

### Functions

#### process_explanation(text: str) -> str
Clean and validate explanation text.

**Parameters:**
- `text` (str): Raw explanation text

**Returns:**
- `str`: Cleaned explanation text

**Example:**
```python
cleaned = process_explanation("  Migros   yanı  ")
# Returns: "Migros yanı"
```

#### parse_explanation(text: str) -> str
Legacy compatibility function. Same as `process_explanation`.

**Parameters:**
- `text` (str): Raw explanation text

**Returns:**
- `str`: Cleaned explanation text

## 🧪 Testing

### Running Tests
```bash
# Run explanation processing tests
python -m pytest tests/test_explanation_simple.py -v

# Run with coverage
python -m pytest tests/test_explanation_simple.py --cov=src/addrnorm/explanation
```

### Test Coverage
- Basic text processing: 8 test cases
- Edge cases: whitespace, empty input, non-string input
- Turkish character support
- Backward compatibility

## 📈 Performance

- **Processing Speed**: <1ms per text
- **Memory Usage**: Minimal
- **No Dependencies**: Uses only built-in Python modules

## 🔄 Migration from Complex Parser

The explanation module has been simplified to focus on basic text processing:

### Before (Complex Parser)
```python
from addrnorm.explanation import ExplanationParser

parser = ExplanationParser()
result = parser.parse("Migros yanı")
# Returned complex object with landmarks, relations, confidence, etc.
```

### After (Simple Processing)
```python
from addrnorm.explanation import process_explanation

cleaned = process_explanation("Migros yanı")
# Returns simple cleaned string: "Migros yanı"
```

This change simplifies the system while maintaining the essential functionality of cleaning explanation text for the `explanation_raw` field.
