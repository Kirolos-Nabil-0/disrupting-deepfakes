"""
Tasks package for Celery background processing.
"""

from .batch_processing import (
    process_batch_images,
    cleanup_completed_tasks,
    get_batch_task_status,
    retry_failed_batch_processing
)
from .model_management import (
    preload_models,
    health_check_models,
    cleanup_model_cache,
    benchmark_model_performance
)

__all__ = [
    # Batch processing tasks
    "process_batch_images",
    "cleanup_completed_tasks", 
    "get_batch_task_status",
    "retry_failed_batch_processing",
    
    # Model management tasks
    "preload_models",
    "health_check_models",
    "cleanup_model_cache",
    "benchmark_model_performance",
]