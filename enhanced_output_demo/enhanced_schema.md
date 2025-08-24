# Enhanced Address Normalization Output Schema

## Overview
This document describes the enhanced output format for address normalization results.

## Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `normalized_address` | string | Yes | Final normalized address string |
| `extracted_fields` | object | No | Extracted address components |
| `explanation` | object | No | Processing explanation details |
| `confidence_scores` | object | Yes | Multi-level confidence scores |
| `quality_metrics` | object | Yes | Quality assessment metrics |
| `processing_method` | string | No | Processing method used |
| `validation_status` | string | No | Validation status |

## Example Output

```json
{
  "normalized_address": "\u0130stanbul Kad\u0131k\u00f6y Moda Mahallesi Bahariye Caddesi No: 15",
  "explanation": {
    "raw": "Amorium Hotel kar\u015f\u0131s\u0131",
    "parsed": {
      "type": "landmark",
      "name": "Amorium Hotel",
      "relation": "kar\u015f\u0131s\u0131"
    }
  },
  "confidence_scores": {
    "pattern": 0.95,
    "ml": 0.87,
    "overall": 0.91
  },
  "quality_metrics": {
    "completeness": 0.9,
    "consistency": 0.85
  },
  "processing_method": "pattern_primary",
  "validation_status": "passed"
}
```
