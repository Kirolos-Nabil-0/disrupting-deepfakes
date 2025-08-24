"""
FastAPI application entry point for the Deepfake Protection API.

This module sets up the FastAPI application with all necessary middleware,
routers, and startup/shutdown event handlers.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import ProtectionError
from app.core.model_manager import ModelManager
from app.core.image_processor import ImageProcessor
from app.services.protection_service import ImageProtectionService
from app.api.v1.router import api_router
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware


# Initialize settings and logging
settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Global instances
model_manager: ModelManager = None
protection_service: ImageProtectionService = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Model loading during startup
    - Resource cleanup during shutdown
    """
    global model_manager
    
    logger.info("Starting Deepfake Protection API...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(
            models_dir=settings.models_dir,
            device=settings.device,
            cache_models=settings.cache_models
        )
        
        # Initialize image processor and protection service
        image_processor = ImageProcessor(
            max_size_mb=settings.max_image_size_mb,
            max_dimension=settings.max_image_dimension,
            supported_formats=settings.supported_formats
        )
        
        protection_service = ImageProtectionService(
            model_manager=model_manager,
            image_processor=image_processor
        )
        
        # Load default models
        if settings.preload_models:
            await model_manager.load_default_models()
            
        app.state.model_manager = model_manager
        app.state.protection_service = protection_service
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Deepfake Protection API...")
    if model_manager:
        await model_manager.cleanup()
    if protection_service:
        # Clean up any remaining batch tasks
        protection_service.cleanup_completed_tasks(max_age_hours=0)
    logger.info("API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Deepfake Protection API",
    description="API for protecting images against deepfake manipulation using adversarial perturbations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include API routers
app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(ProtectionError)
async def protection_error_handler(request: Request, exc: ProtectionError):
    """Handle protection-specific errors."""
    logger.error(f"Protection error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.__class__.__name__.upper(),
                "message": str(exc),
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": str(asyncio.get_event_loop().time())
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": str(asyncio.get_event_loop().time())
            }
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": len(model_manager.loaded_models) if model_manager else 0
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )