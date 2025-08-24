"""
Pydantic data models for the Deepfake Protection API.

This module defines all the request/response schemas, enums, and validation
models used throughout the API.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from pydantic import BaseModel, Field, validator, ConfigDict
import base64


class ModelType(str, Enum):
    """Available deepfake model types."""
    STARGAN = "stargan"
    GANIMATION = "ganimation"
    PIX2PIXHD = "pix2pixhd"
    CYCLEGAN = "cyclegan"


class AttackMethod(str, Enum):
    """Available adversarial attack methods."""
    FGSM = "fgsm"
    PGD = "pgd"
    IFGSM = "ifgsm"


class ImageFormat(str, Enum):
    """Supported image output formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class TaskStatus(str, Enum):
    """Task processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStatus(str, Enum):
    """Model loading status."""
    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"


# Attack Parameters Models
class AttackParams(BaseModel):
    """Parameters for adversarial attack configuration."""
    model_config = ConfigDict(extra='forbid')
    
    epsilon: float = Field(
        default=0.05, 
        ge=0.01, 
        le=0.1,
        description="Maximum perturbation magnitude (L-infinity norm)"
    )
    iterations: int = Field(
        default=10, 
        ge=1, 
        le=50,
        description="Number of attack iterations"
    )
    step_size: float = Field(
        default=0.01, 
        ge=0.001, 
        le=0.05,
        description="Step size for iterative attacks"
    )
    random_start: bool = Field(
        default=True,
        description="Whether to use random initialization"
    )
    
    @validator('step_size')
    def step_size_must_be_smaller_than_epsilon(cls, v, values):
        if 'epsilon' in values and v > values['epsilon']:
            raise ValueError('step_size must be smaller than or equal to epsilon')
        return v


# Protection Configuration Models
class ProtectionConfig(BaseModel):
    """Configuration for image protection."""
    model_config = ConfigDict(extra='forbid')
    
    model_type: ModelType = Field(default=ModelType.STARGAN)
    attack_method: AttackMethod = Field(default=AttackMethod.PGD)
    attack_params: AttackParams = Field(default_factory=AttackParams)
    target_attributes: Optional[List[str]] = Field(
        default=None,
        description="Target attributes for facial manipulation models"
    )
    output_format: ImageFormat = Field(default=ImageFormat.JPEG)


# Request Models
class ProtectionRequest(BaseModel):
    """Request model for single image protection."""
    model_config = ConfigDict(extra='forbid')
    
    image: str = Field(
        ..., 
        description="Base64 encoded image data",
        min_length=1
    )
    model_type: ModelType = Field(default=ModelType.STARGAN)
    attack_method: AttackMethod = Field(default=AttackMethod.PGD)
    attack_params: AttackParams = Field(default_factory=AttackParams)
    target_attributes: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="Target attributes for facial manipulation models"
    )
    output_format: ImageFormat = Field(default=ImageFormat.JPEG)
    
    @validator('image')
    def validate_base64_image(cls, v):
        """Validate that the image is proper base64 encoded data."""
        try:
            # Try to decode base64
            decoded = base64.b64decode(v)
            # Check minimum size (should be at least a few KB for real images)
            if len(decoded) < 1024:
                raise ValueError("Image data too small")
            return v
        except Exception:
            raise ValueError("Invalid base64 image data")


class BatchProtectionRequest(BaseModel):
    """Request model for batch image protection."""
    model_config = ConfigDict(extra='forbid')
    
    images: List[str] = Field(
        ..., 
        min_items=1,
        max_items=10,
        description="List of base64 encoded images"
    )
    protection_config: ProtectionConfig
    async_processing: bool = Field(
        default=False,
        description="Whether to process asynchronously"
    )
    
    @validator('images')
    def validate_images(cls, v):
        """Validate all images in the batch."""
        for i, image in enumerate(v):
            try:
                decoded = base64.b64decode(image)
                if len(decoded) < 1024:
                    raise ValueError(f"Image {i} data too small")
            except Exception:
                raise ValueError(f"Invalid base64 data for image {i}")
        return v


# Response Models
class ProtectionMetrics(BaseModel):
    """Metrics for evaluating protection quality."""
    l2_norm: float = Field(description="L2 norm of perturbation")
    linf_norm: float = Field(description="L-infinity norm of perturbation")
    ssim: float = Field(description="Structural similarity index", ge=0, le=1)
    psnr: float = Field(description="Peak signal-to-noise ratio")


class ProtectionResponse(BaseModel):
    """Response model for single image protection."""
    protected_image: str = Field(description="Base64 encoded protected image")
    perturbation_map: Optional[str] = Field(
        default=None,
        description="Base64 encoded perturbation visualization"
    )
    protection_metrics: ProtectionMetrics
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    model_used: str = Field(description="Model type used for protection")
    attack_applied: str = Field(description="Attack method applied")


class BatchTaskProgress(BaseModel):
    """Progress information for batch tasks."""
    processed: int = Field(description="Number of images processed")
    total: int = Field(description="Total number of images")
    percentage: float = Field(description="Completion percentage", ge=0, le=100)


class BatchImageResult(BaseModel):
    """Result for a single image in batch processing."""
    image_id: int = Field(description="Index of the image in the batch")
    protected_image: str = Field(description="Base64 encoded protected image")
    metrics: ProtectionMetrics
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class BatchTaskResponse(BaseModel):
    """Response model for batch protection tasks."""
    task_id: str = Field(description="Unique task identifier")
    status: TaskStatus
    estimated_completion_ms: Optional[int] = Field(
        default=None,
        description="Estimated completion time in milliseconds"
    )
    images_count: int = Field(description="Number of images in the batch")


class BatchStatusResponse(BaseModel):
    """Response model for batch task status."""
    task_id: str
    status: TaskStatus
    progress: BatchTaskProgress
    results: Optional[List[BatchImageResult]] = Field(
        default=None,
        description="Results available when status is completed"
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Total processing time when completed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task failed"
    )


# Model Management Models
class ModelInfo(BaseModel):
    """Information about a loaded model."""
    name: str
    status: ModelStatus
    version: str
    capabilities: List[str] = Field(description="Model capabilities")
    supported_attacks: List[str] = Field(description="Supported attack methods")
    memory_usage_mb: Optional[int] = Field(default=None, description="Memory usage in MB")
    loaded_at: Optional[datetime] = Field(default=None, description="When the model was loaded")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    models: List[ModelInfo]


class AttackMethodInfo(BaseModel):
    """Information about an attack method."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = Field(
        description="Parameter definitions with type, range, and default values"
    )


class AttackMethodsResponse(BaseModel):
    """Response model for available attack methods."""
    attack_methods: List[AttackMethodInfo]


# Health Check Models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: int
    gpu_available: bool
    memory_usage: Optional[Dict[str, float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Error Models
class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail