# Turkish Address Normalization REST API

Minimal REST API layer for Turkish address normalization using FastAPI.

## Features

- **FastAPI Framework**: Modern, fast web framework with automatic API docs
- **Pydantic Validation**: Request/response schema validation
- **Single & Batch Processing**: Handle single addresses or batch requests
- **Health Check**: Service health monitoring endpoint
- **Processing Statistics**: Detailed stats including pattern_score, ner_used, fallback_used
- **Environment Configuration**: Configurable via environment variables

## API Endpoints

### Health Check
```
GET /healthz
```

Returns service health status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "pipeline": "healthy",
    "ml_model": "available",
    "pattern_matcher": "available",
    "geo_validator": "available"
  }
}
```

### Address Normalization
```
POST /normalize
```

Normalize Turkish address(es). Accepts single or batch requests.

**Single Address Request:**
```json
{
  "text": "Beşiktaş İlçesi Levent Mahallesi Büyükdere Caddesi No:100 Daire:5 İstanbul"
}
```

**Batch Address Request:**
```json
{
  "texts": [
    "Kadıköy İlçesi Moda Mahallesi Caferağa Mah. Şair Nedim Cad. No:15 İstanbul",
    "Şişli Mecidiyeköy Büyükdere Cad. No:78 K:5 D:12 İstanbul",
    "Ankara Çankaya Kızılay Meydanı No:1"
  ]
}
```

**Single Address Response:**
```json
{
  "success": true,
  "address": {
    "city": "İstanbul",
    "district": "Beşiktaş",
    "neighborhood": "Levent",
    "street": "Büyükdere Caddesi",
    "number": "100",
    "apartment": "5",
    "normalized_address": "İstanbul Beşiktaş Levent Büyükdere Caddesi No:100 Daire:5",
    "explanation_parsed": {
      "confidence": 0.85,
      "method": "pattern",
      "warnings": []
    }
  },
  "stats": {
    "pattern_score": 0.85,
    "ner_used": false,
    "fallback_used": false,
    "processing_method": "pattern_high",
    "confidence": 0.85,
    "processing_time_ms": 12.5
  },
  "error": null
}
```

**Batch Address Response:**
```json
{
  "results": [
    {
      "success": true,
      "address": { /* AddressOut object */ },
      "stats": { /* ProcessingStats object */ },
      "error": null
    }
  ],
  "total_count": 3,
  "success_count": 3,
  "error_count": 0
}
```

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn
```

### 2. Start Server

```bash
# Using uvicorn directly
uvicorn addrnorm.api.server:app --host 0.0.0.0 --port 8000

# Or using the launcher script
python run_api.py
```

### 3. Test API

```bash
# Health check
curl http://localhost:8000/healthz

# Single address normalization
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"text": "İstanbul Beşiktaş Levent Büyükdere Cad. No:100 D:5"}'

# Batch normalization
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"texts": ["İstanbul Beşiktaş Levent", "Ankara Çankaya Kızılay"]}'
```

### 4. Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI documentation
Visit http://localhost:8000/redoc for ReDoc documentation

## Environment Configuration

Configure the API using environment variables:

```bash
# Server settings
export PORT=8000              # Server port (default: 8000)
export WORKERS=1              # Number of workers (default: 1)
export TIMEOUT=30             # Request timeout in seconds (default: 30)

# Pipeline settings
export ML_MODEL_PATH="models/turkish_address_ner_improved"  # ML model path
export GEO_DATA_DIR="/path/to/geo/data"                     # Geographic data directory

# Start server
uvicorn addrnorm.api.server:app --host 0.0.0.0 --port $PORT --workers $WORKERS
```

## Processing Statistics

Each response includes detailed processing statistics:

- **pattern_score**: Pattern matching confidence score (0.0-1.0)
- **ner_used**: Whether ML NER was used in processing
- **fallback_used**: Whether fallback method was used
- **processing_method**: Primary method used (pattern_high, pattern_medium, ml, fallback)
- **confidence**: Overall confidence score (0.0-1.0)
- **processing_time_ms**: Processing time in milliseconds

## Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid request format or validation errors
- **422 Unprocessable Entity**: Pydantic validation errors
- **500 Internal Server Error**: Processing errors or server issues
- **503 Service Unavailable**: Pipeline not initialized

## Development

### Running Tests

```bash
# Start API server
uvicorn addrnorm.api.server:app --host 0.0.0.0 --port 8000

# Run test script (in another terminal)
python test_api.py
```

### Deployment

For production deployment:

```bash
# Using Gunicorn with Uvicorn workers
gunicorn addrnorm.api.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker (if Dockerfile exists)
docker build -t addrnorm-api .
docker run -p 8000:8000 addrnorm-api
```

## Architecture

The API layer is built on top of the existing pipeline:

```
FastAPI Server
├── Pydantic Models (request/response validation)
├── AddressNormalizationPipeline
│   ├── Pattern Matching
│   ├── ML NER Inference
│   ├── Fallback Processing
│   └── Geographic Validation
└── Error Handling & Statistics
```

## Performance

- **Single Address**: ~10-50ms per request
- **Batch Processing**: Efficient for multiple addresses
- **Memory Usage**: ~200-500MB depending on models loaded
- **Concurrency**: Supports multiple concurrent requests

## License

MIT License - Same as the main project.
