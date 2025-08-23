"""
FastAPI server for Turkish address normalization REST API.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..pipeline.run import (
    AddressNormalizationPipeline,
    PipelineConfig,
    ProcessingResult,
)
from .models import (
    AddressRequest,
    AddressResponse,
    BatchAddressRequest,
    BatchAddressResponse,
    HealthResponse,
)

# Environment variables with defaults
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

# Application version
VERSION = "2.0.0"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Turkish Address Normalization API",
    description="REST API for normalizing Turkish addresses using ML and "
    "rule-based methods",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: AddressNormalizationPipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on application startup."""
    global pipeline

    logger.info("Initializing address normalization pipeline...")

    try:
        config = PipelineConfig(
            log_level="INFO",
            enable_validation=True,
            # Configure based on environment
            ml_model_path=os.getenv(
                "ML_MODEL_PATH", "models/turkish_address_ner_improved"
            ),
            geo_data_dir=os.getenv("GEO_DATA_DIR", None),
        )

        pipeline = AddressNormalizationPipeline(config)
        logger.info("Pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application...")


def _result_to_dict(result: ProcessingResult) -> dict:
    """Convert ProcessingResult to dictionary for API response."""
    return {
        "raw_input": result.raw_input,
        "success": result.success,
        "address_out": result.address_out.to_dict() if result.address_out else None,
        "error": result.error,
        "processing_method": result.processing_method,
        "confidence": result.confidence,
        "processing_time_ms": result.processing_time_ms,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""

    components = {
        "pipeline": pipeline is not None,
        "ml_model": pipeline.ml_inference is not None if pipeline else False,
        "pattern_matcher": pipeline.pattern_matcher is not None if pipeline else False,
        "geo_validator": pipeline.geo_validator is not None if pipeline else False,
    }

    return HealthResponse(
        status="healthy" if pipeline else "unhealthy",
        version=VERSION,
        components=components,
    )


@app.post("/normalize", response_model=AddressResponse)
async def normalize_address(request: AddressRequest):
    """
    Normalize a single Turkish address.

    Accepts: {"text": "address string"}
    Returns: ProcessingResult structure
    """

    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        logger.info(f"Processing single address: {request.text[:50]}...")

        result = pipeline.process_single(request.text)
        return _result_to_dict(result)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}",
        )


@app.post("/normalize/batch", response_model=BatchAddressResponse)
async def normalize_batch(request: BatchAddressRequest):
    """
    Normalize multiple Turkish addresses.

    Accepts: {"texts": ["address1", "address2", ...]}
    Returns: Batch results with summary
    """

    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        logger.info(f"Processing batch of {len(request.texts)} addresses...")

        results = []
        success_count = 0

        for text in request.texts:
            result = pipeline.process_single(text)
            address_response = _result_to_dict(result)
            results.append(address_response)

            if result.success:
                success_count += 1

        return BatchAddressResponse(
            results=results,
            total_count=len(request.texts),
            success_count=success_count,
            error_count=len(request.texts) - success_count,
        )

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""

    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


def main():
    """Run the server."""

    logger.info(f"Starting server on port {PORT} with {WORKERS} workers")

    uvicorn.run(
        "addrnorm.api.server:app",
        host="0.0.0.0",
        port=PORT,
        workers=WORKERS,
        timeout_keep_alive=TIMEOUT,
        reload=False,  # Set to True for development
        log_level="info",
    )


if __name__ == "__main__":
    main()
