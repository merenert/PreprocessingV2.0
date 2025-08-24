# Migration Guide: Legacy to Enhanced Output Format

## Overview
This guide helps you migrate from the legacy address normalization output format to the new enhanced format.

## Key Changes

### Legacy Format
```json
{
  "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
  "confidence": 0.85,
  "method": "pattern",
  "il": "İstanbul",
  "ilce": "Kadıköy",
  "mahalle": "Moda",
  "sokak": "Bahariye Caddesi",
  "bina_no": "15"
}
```

### Enhanced Format
```json
{
  "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
  "extracted_fields": {
    "il": "İstanbul",
    "ilce": "Kadıköy",
    "mahalle": "Moda",
    "sokak": "Bahariye Caddesi",
    "bina_no": "15"
  },
  "explanation": {
    "raw": "Original address input",
    "parsed": { "type": "standard" },
    "processing_steps": ["validation", "extraction"],
    "method_details": { "primary_method": "pattern" }
  },
  "confidence_scores": {
    "pattern": 0.95,
    "ml": 0.75,
    "overall": 0.85
  },
  "quality_metrics": {
    "completeness": 0.9,
    "consistency": 0.85,
    "accuracy": 0.88,
    "usability": 0.82
  },
  "processing_method": "pattern_primary",
  "validation_status": "passed"
}
```

## Migration Steps

### 1. Update Field Access
- **Before**: `result["il"]`
- **After**: `result["extracted_fields"]["il"]`

### 2. Update Confidence Handling
- **Before**: `result["confidence"]`
- **After**: `result["confidence_scores"]["overall"]`

### 3. Update Method Detection
- **Before**: `result["method"]`
- **After**: `result["processing_method"]`

### 4. Add Quality Assessment
- **New**: `result["quality_metrics"]["overall_score"]`
- **New**: `result["validation_status"]`

## Backward Compatibility

The system supports legacy format output by setting `output_format="legacy_json"`:

```python
formatter = EnhancedFormatter(enable_legacy_compatibility=True)
result = formatter.format_result(data, original, context, OutputFormat.LEGACY_JSON)
```

## Code Examples

### Before (Legacy)
```python
def process_address(address):
    result = normalize_address(address)
    return {
        'address': result['normalized_address'],
        'confidence': result['confidence'],
        'city': result.get('il'),
        'district': result.get('ilce')
    }
```

### After (Enhanced)
```python
def process_address(address):
    result = normalize_address_enhanced(address)
    return {
        'address': result['normalized_address'],
        'confidence': result['confidence_scores']['overall'],
        'quality': result['quality_metrics']['overall_score'],
        'city': result['extracted_fields'].get('il'),
        'district': result['extracted_fields'].get('ilce'),
        'validation': result['validation_status']
    }
```

## Best Practices

1. **Always check validation_status** before using results
2. **Use quality_metrics** to filter low-quality results
3. **Leverage explanation** for debugging and user feedback
4. **Monitor confidence_scores** for performance insights
5. **Use appropriate output_format** for your use case

## Support

For additional support during migration, please refer to the API documentation or contact the development team.
