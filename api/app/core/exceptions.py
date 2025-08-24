"""
Custom exceptions for the Deepfake Protection API.

This module defines all custom exceptions used throughout the application
with proper error categorization and handling.
"""

from typing import Optional, Dict, Any


class ProtectionError(Exception):
    """Base exception for all protection-related errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ModelError(ProtectionError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    
    def __init__(
        self, 
        model_type: str, 
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type
        message = message or f"Failed to load model: {model_type}"
        details = details or {"model_type": model_type}
        super().__init__(message, "MODEL_LOAD_FAILED", details)


class ModelNotFoundError(ModelError):
    """Exception raised when a requested model is not found."""
    
    def __init__(
        self, 
        model_type: str,
        available_models: Optional[list] = None
    ):
        self.model_type = model_type
        message = f"Model not found: {model_type}"
        details = {"model_type": model_type}
        if available_models:
            details["available_models"] = available_models
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(message, "MODEL_NOT_FOUND", details)


class ModelNotLoadedError(ModelError):
    """Exception raised when trying to use an unloaded model."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        message = f"Model not loaded: {model_type}"
        details = {"model_type": model_type}
        super().__init__(message, "MODEL_NOT_LOADED", details)


class ImageProcessingError(ProtectionError):
    """Base exception for image processing errors."""
    pass


class InvalidImageError(ImageProcessingError):
    """Exception raised when image data is invalid."""
    
    def __init__(
        self, 
        message: Optional[str] = None,
        image_details: Optional[Dict[str, Any]] = None
    ):
        message = message or "Invalid image data"
        details = image_details or {}
        super().__init__(message, "INVALID_IMAGE", details)


class ImageTooLargeError(ImageProcessingError):
    """Exception raised when image exceeds size limits."""
    
    def __init__(
        self, 
        size_mb: float, 
        max_size_mb: float,
        dimensions: Optional[tuple] = None
    ):
        self.size_mb = size_mb
        self.max_size_mb = max_size_mb
        message = f"Image too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        details = {"size_mb": size_mb, "max_size_mb": max_size_mb}
        if dimensions:
            details["dimensions"] = dimensions
        super().__init__(message, "IMAGE_TOO_LARGE", details)


class UnsupportedImageFormatError(ImageProcessingError):
    """Exception raised for unsupported image formats."""
    
    def __init__(
        self, 
        format_name: str, 
        supported_formats: Optional[list] = None
    ):
        self.format_name = format_name
        message = f"Unsupported image format: {format_name}"
        details = {"format": format_name}
        if supported_formats:
            details["supported_formats"] = supported_formats
            message += f". Supported formats: {', '.join(supported_formats)}"
        super().__init__(message, "UNSUPPORTED_FORMAT", details)


class AttackError(ProtectionError):
    """Base exception for adversarial attack errors."""
    pass


class AttackGenerationError(AttackError):
    """Exception raised when adversarial attack generation fails."""
    
    def __init__(
        self, 
        attack_method: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.attack_method = attack_method
        message = message or f"Failed to generate adversarial attack: {attack_method}"
        details = details or {"attack_method": attack_method}
        super().__init__(message, "ATTACK_GENERATION_FAILED", details)


class InvalidAttackParametersError(AttackError):
    """Exception raised for invalid attack parameters."""
    
    def __init__(
        self, 
        parameter_name: str,
        parameter_value: Any,
        valid_range: Optional[str] = None
    ):
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        message = f"Invalid attack parameter: {parameter_name}={parameter_value}"
        details = {
            "parameter_name": parameter_name,
            "parameter_value": parameter_value
        }
        if valid_range:
            details["valid_range"] = valid_range
            message += f" (valid range: {valid_range})"
        super().__init__(message, "INVALID_ATTACK_PARAMETERS", details)


class GPUError(ProtectionError):
    """Base exception for GPU-related errors."""
    pass


class GPUMemoryError(GPUError):
    """Exception raised when GPU runs out of memory."""
    
    def __init__(
        self, 
        required_memory: Optional[float] = None,
        available_memory: Optional[float] = None
    ):
        message = "Insufficient GPU memory"
        details = {}
        if required_memory:
            details["required_memory_gb"] = required_memory
            message += f" (required: {required_memory:.1f}GB"
        if available_memory:
            details["available_memory_gb"] = available_memory
            message += f", available: {available_memory:.1f}GB"
        if required_memory or available_memory:
            message += ")"
        super().__init__(message, "GPU_MEMORY_ERROR", details)


class GPUNotAvailableError(GPUError):
    """Exception raised when GPU is not available but required."""
    
    def __init__(self):
        message = "GPU not available but required for operation"
        super().__init__(message, "GPU_NOT_AVAILABLE")


class ValidationError(ProtectionError):
    """Exception raised for input validation errors."""
    
    def __init__(
        self, 
        field_name: str,
        field_value: Any,
        validation_message: str
    ):
        self.field_name = field_name
        self.field_value = field_value
        message = f"Validation error for field '{field_name}': {validation_message}"
        details = {
            "field_name": field_name,
            "field_value": str(field_value),
            "validation_message": validation_message
        }
        super().__init__(message, "VALIDATION_ERROR", details)


class RateLimitError(ProtectionError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self, 
        limit: int,
        window: str,
        retry_after: Optional[int] = None
    ):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        message = f"Rate limit exceeded: {limit} requests per {window}"
        details = {"limit": limit, "window": window}
        if retry_after:
            details["retry_after_seconds"] = retry_after
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class BatchProcessingError(ProtectionError):
    """Exception raised during batch processing."""
    
    def __init__(
        self, 
        task_id: str,
        processed_count: int,
        total_count: int,
        message: Optional[str] = None
    ):
        self.task_id = task_id
        self.processed_count = processed_count
        self.total_count = total_count
        message = message or f"Batch processing failed for task {task_id}"
        details = {
            "task_id": task_id,
            "processed_count": processed_count,
            "total_count": total_count
        }
        super().__init__(message, "BATCH_PROCESSING_FAILED", details)


class TaskNotFoundError(ProtectionError):
    """Exception raised when a batch task is not found."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        message = f"Task not found: {task_id}"
        details = {"task_id": task_id}
        super().__init__(message, "TASK_NOT_FOUND", details)


class ConfigurationError(ProtectionError):
    """Exception raised for configuration errors."""
    
    def __init__(
        self, 
        config_key: str,
        config_value: Any,
        message: Optional[str] = None
    ):
        self.config_key = config_key
        self.config_value = config_value
        message = message or f"Invalid configuration: {config_key}={config_value}"
        details = {
            "config_key": config_key,
            "config_value": str(config_value)
        }
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ResourceNotFoundError(ProtectionError):
    """Exception raised when a required resource is not found."""
    
    def __init__(
        self, 
        resource_type: str,
        resource_id: str,
        message: Optional[str] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = message or f"{resource_type} not found: {resource_id}"
        details = {
            "resource_type": resource_type,
            "resource_id": resource_id
        }
        super().__init__(message, "RESOURCE_NOT_FOUND", details)


class ServiceUnavailableError(ProtectionError):
    """Exception raised when service is temporarily unavailable."""
    
    def __init__(
        self, 
        service_name: str,
        retry_after: Optional[int] = None,
        message: Optional[str] = None
    ):
        self.service_name = service_name
        self.retry_after = retry_after
        message = message or f"Service temporarily unavailable: {service_name}"
        details = {"service_name": service_name}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, "SERVICE_UNAVAILABLE", details)


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAPPING = {
    ValidationError: 400,
    InvalidImageError: 400,
    InvalidAttackParametersError: 400,
    UnsupportedImageFormatError: 400,
    ImageTooLargeError: 413,
    ModelNotFoundError: 404,
    TaskNotFoundError: 404,
    ResourceNotFoundError: 404,
    RateLimitError: 429,
    ServiceUnavailableError: 503,
    GPUMemoryError: 503,
    GPUNotAvailableError: 503,
    ModelLoadError: 503,
    AttackGenerationError: 500,
    BatchProcessingError: 500,
    ImageProcessingError: 500,
    ModelError: 500,
    ConfigurationError: 500,
    ProtectionError: 500,
}


def get_http_status_code(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    for exc_type, status_code in EXCEPTION_STATUS_MAPPING.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default to internal server error