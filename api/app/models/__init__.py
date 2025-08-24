"""
Data models package for the Deepfake Protection API.
"""

from .schemas import (
    # Enums
    ModelType,
    AttackMethod,
    ImageFormat,
    TaskStatus,
    ModelStatus,
    
    # Request models
    ProtectionRequest,
    BatchProtectionRequest,
    AttackParams,
    ProtectionConfig,
    
    # Response models
    ProtectionResponse,
    BatchTaskResponse,
    BatchStatusResponse,
    ProtectionMetrics,
    BatchImageResult,
    BatchTaskProgress,
    
    # Model management
    ModelInfo,
    ModelsResponse,
    AttackMethodInfo,
    AttackMethodsResponse,
    
    # Health and errors
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
)

__all__ = [
    # Enums
    "ModelType",
    "AttackMethod", 
    "ImageFormat",
    "TaskStatus",
    "ModelStatus",
    
    # Request models
    "ProtectionRequest",
    "BatchProtectionRequest",
    "AttackParams",
    "ProtectionConfig",
    
    # Response models
    "ProtectionResponse",
    "BatchTaskResponse", 
    "BatchStatusResponse",
    "ProtectionMetrics",
    "BatchImageResult",
    "BatchTaskProgress",
    
    # Model management
    "ModelInfo",
    "ModelsResponse",
    "AttackMethodInfo",
    "AttackMethodsResponse",
    
    # Health and errors
    "HealthResponse",
    "ErrorResponse",
    "ErrorDetail",
]