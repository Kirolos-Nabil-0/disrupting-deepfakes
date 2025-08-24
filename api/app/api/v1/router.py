"""
API Router for the Deepfake Protection API v1.

This module defines all the API endpoints for image protection,
model management, and system monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List

from app.models.schemas import (
    ProtectionRequest,
    ProtectionResponse,
    BatchProtectionRequest,
    BatchTaskResponse,
    BatchStatusResponse,
    ModelsResponse,
    AttackMethodsResponse,
    AttackMethodInfo,
    HealthResponse,
    ErrorResponse
)
from app.services.protection_service import ImageProtectionService
from app.core.model_manager import ModelManager
from app.core.attacks import AttackFactory
from app.core.config import get_settings
from app.core.exceptions import ProtectionError, get_http_status_code
from app.core.logging import request_logger, get_logger

logger = get_logger(__name__)
settings = get_settings()

# Create API router
api_router = APIRouter(
    prefix="/api/v1",
    tags=["Deepfake Protection API v1"]
)

# Dependency to get protection service
def get_protection_service(request: Request) -> ImageProtectionService:
    """Get the protection service from app state."""
    return request.app.state.protection_service

# Dependency to get model manager
def get_model_manager(request: Request) -> ModelManager:
    """Get the model manager from app state."""
    return request.app.state.model_manager


@api_router.post(
    "/protect/image",
    response_model=ProtectionResponse,
    summary="Protect Single Image",
    description="Apply adversarial protection to a single image to prevent deepfake manipulation"
)
async def protect_image(
    request: ProtectionRequest,
    protection_service: ImageProtectionService = Depends(get_protection_service)
) -> ProtectionResponse:
    """
    Protect a single image with adversarial perturbations.
    
    This endpoint applies the specified adversarial attack method to the input image
    using the selected model to generate imperceptible perturbations that protect
    against deepfake manipulation.
    
    - **image**: Base64 encoded image data (JPEG, PNG, WebP supported)
    - **model_type**: Deepfake model to use for protection (stargan, ganimation, etc.)
    - **attack_method**: Adversarial attack method (fgsm, pgd, ifgsm)
    - **attack_params**: Attack parameters (epsilon, iterations, step_size)
    - **target_attributes**: Target attributes for facial manipulation models
    - **output_format**: Output image format (jpeg, png, webp)
    
    Returns the protected image along with quality metrics and processing information.
    """
    try:
        logger.info(
            "Single image protection requested",
            model_type=request.model_type.value,
            attack_method=request.attack_method.value
        )
        
        result = await protection_service.protect_single_image(request)
        
        logger.info(
            "Single image protection completed successfully",
            processing_time_ms=result.processing_time_ms
        )
        
        return result
        
    except ProtectionError as e:
        logger.error(f"Protection error: {e}")
        raise HTTPException(
            status_code=get_http_status_code(e),
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in protect_image: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during image protection"
            }
        )


@api_router.post(
    "/protect/batch",
    response_model=BatchTaskResponse,
    summary="Protect Batch of Images",
    description="Process multiple images for protection in batch mode"
)
async def protect_batch(
    request: BatchProtectionRequest,
    background_tasks: BackgroundTasks,
    protection_service: ImageProtectionService = Depends(get_protection_service)
) -> BatchTaskResponse:
    """
    Process multiple images for protection in batch mode.
    
    This endpoint accepts multiple images and processes them using the same
    protection configuration. Can be processed synchronously or asynchronously.
    
    - **images**: List of base64 encoded images (max 10 images)
    - **protection_config**: Shared configuration for all images
    - **async_processing**: Whether to process asynchronously (default: false)
    
    For synchronous processing, the response will include task completion status.
    For asynchronous processing, use the returned task_id to check progress.
    """
    try:
        logger.info(
            "Batch protection requested",
            image_count=len(request.images),
            async_processing=request.async_processing,
            model_type=request.protection_config.model_type.value
        )
        
        # Validate batch size
        if len(request.images) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "BATCH_SIZE_EXCEEDED",
                    "message": f"Batch size {len(request.images)} exceeds maximum {settings.max_batch_size}"
                }
            )
        
        result = await protection_service.start_batch_protection(request)
        
        logger.info(
            "Batch protection task created",
            task_id=result.task_id,
            status=result.status.value
        )
        
        return result
        
    except ProtectionError as e:
        logger.error(f"Batch protection error: {e}")
        raise HTTPException(
            status_code=get_http_status_code(e),
            detail=e.to_dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in protect_batch: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during batch protection"
            }
        )


@api_router.get(
    "/protect/status/{task_id}",
    response_model=BatchStatusResponse,
    summary="Get Batch Task Status",
    description="Retrieve the status and results of a batch protection task"
)
async def get_batch_status(
    task_id: str,
    protection_service: ImageProtectionService = Depends(get_protection_service)
) -> BatchStatusResponse:
    """
    Get the status of a batch protection task.
    
    This endpoint returns the current status, progress, and results (if completed)
    for a batch protection task.
    
    - **task_id**: Unique identifier for the batch task
    
    Returns progress information and results when the task is completed.
    """
    try:
        logger.info(f"Batch status requested for task: {task_id}")
        
        result = protection_service.get_batch_status(task_id)
        
        logger.debug(
            "Batch status retrieved",
            task_id=task_id,
            status=result.status.value,
            progress=result.progress.percentage
        )
        
        return result
        
    except ProtectionError as e:
        logger.error(f"Batch status error: {e}")
        raise HTTPException(
            status_code=get_http_status_code(e),
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_batch_status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while retrieving batch status"
            }
        )


@api_router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List Available Models",
    description="Get information about available deepfake protection models"
)
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
) -> ModelsResponse:
    """
    List all available deepfake protection models.
    
    Returns information about each model including:
    - Loading status
    - Capabilities
    - Supported attack methods
    - Memory usage
    - Version information
    """
    try:
        logger.info("Models list requested")
        
        models = model_manager.get_loaded_models()
        
        logger.debug(f"Listed {len(models)} models")
        
        return ModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MODEL_LIST_ERROR",
                "message": "Failed to retrieve model information"
            }
        )


@api_router.post(
    "/models/{model_name}/load",
    summary="Load Model",
    description="Load a specific model into memory"
)
async def load_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
) -> JSONResponse:
    """
    Load a specific model into memory.
    
    This endpoint loads the specified model if it's not already loaded.
    Useful for preloading models to reduce first-request latency.
    
    - **model_name**: Name of the model to load (stargan, ganimation, etc.)
    """
    try:
        from app.models.schemas import ModelType
        
        # Validate model name
        try:
            model_type = ModelType(model_name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_MODEL_NAME",
                    "message": f"Invalid model name: {model_name}",
                    "available_models": [m.value for m in ModelType]
                }
            )
        
        logger.info(f"Model load requested: {model_name}")
        
        wrapper = await model_manager.load_model(model_type)
        
        logger.info(f"Model loaded successfully: {model_name}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Model {model_name} loaded successfully",
                "model_info": wrapper.to_info().dict()
            }
        )
        
    except ProtectionError as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(
            status_code=get_http_status_code(e),
            detail=e.to_dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MODEL_LOAD_ERROR",
                "message": f"Failed to load model {model_name}"
            }
        )


@api_router.post(
    "/models/{model_name}/unload",
    summary="Unload Model",
    description="Unload a specific model from memory"
)
async def unload_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
) -> JSONResponse:
    """
    Unload a specific model from memory.
    
    This endpoint unloads the specified model to free up memory resources.
    
    - **model_name**: Name of the model to unload
    """
    try:
        from app.models.schemas import ModelType
        
        # Validate model name
        try:
            model_type = ModelType(model_name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_MODEL_NAME",
                    "message": f"Invalid model name: {model_name}"
                }
            )
        
        logger.info(f"Model unload requested: {model_name}")
        
        await model_manager.unload_model(model_type)
        
        logger.info(f"Model unloaded successfully: {model_name}")
        
        return JSONResponse(
            status_code=200,
            content={"message": f"Model {model_name} unloaded successfully"}
        )
        
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MODEL_UNLOAD_ERROR",
                "message": f"Failed to unload model {model_name}"
            }
        )


@api_router.get(
    "/config/attack-methods",
    response_model=AttackMethodsResponse,
    summary="List Attack Methods",
    description="Get information about available adversarial attack methods"
)
async def list_attack_methods() -> AttackMethodsResponse:
    """
    List all available adversarial attack methods.
    
    Returns detailed information about each attack method including:
    - Parameter definitions
    - Valid ranges
    - Default values
    - Descriptions
    """
    try:
        logger.info("Attack methods list requested")
        
        methods_info = AttackFactory.get_available_methods()
        
        attack_methods = []
        for name, info in methods_info.items():
            attack_method = AttackMethodInfo(
                name=name,
                description=info["description"],
                parameters=info["parameters"]
            )
            attack_methods.append(attack_method)
        
        logger.debug(f"Listed {len(attack_methods)} attack methods")
        
        return AttackMethodsResponse(attack_methods=attack_methods)
        
    except Exception as e:
        logger.error(f"Error listing attack methods: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ATTACK_METHODS_ERROR",
                "message": "Failed to retrieve attack methods information"
            }
        )


@api_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Get the health status of the API service"
)
async def health_check(
    protection_service: ImageProtectionService = Depends(get_protection_service)
) -> HealthResponse:
    """
    Perform a comprehensive health check of the API service.
    
    Returns information about:
    - Service status
    - Loaded models
    - GPU availability
    - Memory usage
    - Batch processing status
    """
    try:
        logger.debug("Health check requested")
        
        health_info = await protection_service.health_check()
        
        # Create health response
        response = HealthResponse(
            status="healthy" if health_info.get("service_status") == "healthy" else "unhealthy",
            version=settings.version,
            models_loaded=health_info.get("model_manager", {}).get("loaded_models", 0),
            gpu_available=health_info.get("model_manager", {}).get("gpu_available", False),
            memory_usage=health_info.get("model_manager", {}).get("memory_usage")
        )
        
        logger.debug("Health check completed", status=response.status)
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.version,
            models_loaded=0,
            gpu_available=False,
            memory_usage=None
        )