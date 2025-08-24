"""
Celery tasks for batch image processing.

This module contains Celery tasks for handling batch image protection
operations asynchronously.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta

from celery import Task
from celery.exceptions import Retry

from app.core.celery_app import celery_app
from app.core.model_manager import ModelManager
from app.core.image_processor import ImageProcessor
from app.services.protection_service import ImageProtectionService
from app.models.schemas import (
    ProtectionConfig,
    ProtectionRequest,
    TaskStatus,
    BatchImageResult,
    ProtectionMetrics
)
from app.core.config import get_settings
from app.core.logging import get_logger, batch_logger

logger = get_logger(__name__)
settings = get_settings()


class AsyncBatchTask(Task):
    """Base task class with async support."""
    
    def __call__(self, *args, **kwargs):
        """Execute the task with async support."""
        if asyncio.iscoroutinefunction(self.run):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.run(*args, **kwargs))
            finally:
                loop.close()
        else:
            return self.run(*args, **kwargs)


@celery_app.task(bind=True, base=AsyncBatchTask, name="process_batch_images")
async def process_batch_images(
    self,
    task_id: str,
    images: List[str],
    protection_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a batch of images for protection.
    
    Args:
        task_id: Unique task identifier
        images: List of base64 encoded images
        protection_config: Protection configuration dictionary
        
    Returns:
        Task result with processed images and metrics
    """
    start_time = batch_logger.log_batch_start(
        task_id,
        len(images),
        protection_config["model_type"],
        protection_config["attack_method"]
    )
    
    try:
        # Initialize components
        model_manager = ModelManager(
            models_dir=settings.models_dir,
            device=settings.device,
            cache_models=True
        )
        
        image_processor = ImageProcessor(
            max_size_mb=settings.max_image_size_mb,
            max_dimension=settings.max_image_dimension,
            supported_formats=settings.supported_formats
        )
        
        protection_service = ImageProtectionService(
            model_manager=model_manager,
            image_processor=image_processor
        )
        
        # Convert config dict to ProtectionConfig
        config = ProtectionConfig(**protection_config)
        
        # Update task state
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                "processed": 0,
                "total": len(images),
                "percentage": 0,
                "current_image": 0
            }
        )
        
        results = []
        successful = 0
        failed = 0
        
        # Process each image
        for i, image_data in enumerate(images):
            try:
                # Create protection request
                request = ProtectionRequest(
                    image=image_data,
                    model_type=config.model_type,
                    attack_method=config.attack_method,
                    attack_params=config.attack_params,
                    target_attributes=config.target_attributes,
                    output_format=config.output_format
                )
                
                # Process image
                protection_result = await protection_service.protect_single_image(request)
                
                # Create batch result
                batch_result = BatchImageResult(
                    image_id=i,
                    protected_image=protection_result.protected_image,
                    metrics=protection_result.protection_metrics
                )
                results.append(batch_result.dict())
                successful += 1
                
                logger.info(f"Batch image {i} processed successfully", task_id=task_id)
                
            except Exception as e:
                # Handle image processing error
                error_result = BatchImageResult(
                    image_id=i,
                    protected_image="",
                    metrics=ProtectionMetrics(l2_norm=0, linf_norm=0, ssim=0, psnr=0),
                    error=str(e)
                )
                results.append(error_result.dict())
                failed += 1
                
                logger.error(f"Batch image {i} processing failed", task_id=task_id, error=str(e))
            
            # Update progress
            processed = i + 1
            percentage = (processed / len(images)) * 100
            
            self.update_state(
                state=TaskStatus.PROCESSING.value,
                meta={
                    "processed": processed,
                    "total": len(images),
                    "percentage": round(percentage, 1),
                    "current_image": i,
                    "successful": successful,
                    "failed": failed
                }
            )
            
            # Log progress periodically
            if processed % max(1, len(images) // 10) == 0:
                current_time = asyncio.get_event_loop().time()
                duration_ms = (current_time - start_time) * 1000
                batch_logger.log_batch_progress(task_id, processed, len(images), duration_ms)
        
        # Clean up resources
        await model_manager.cleanup()
        
        # Log completion
        batch_logger.log_batch_end(
            task_id,
            start_time,
            successful,
            failed,
            len(images)
        )
        
        # Return final results
        return {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED.value,
            "results": results,
            "summary": {
                "total": len(images),
                "successful": successful,
                "failed": failed,
                "processing_time_ms": int((asyncio.get_event_loop().time() - start_time) * 1000)
            }
        }
        
    except Exception as e:
        # Handle task-level errors
        logger.error(f"Batch processing task failed", task_id=task_id, error=str(e))
        
        batch_logger.log_batch_end(
            task_id,
            start_time,
            successful if 'successful' in locals() else 0,
            failed if 'failed' in locals() else 0,
            len(images),
            error=str(e)
        )
        
        # Update task state to failed
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                "error": str(e),
                "processed": successful + failed if 'successful' in locals() and 'failed' in locals() else 0,
                "total": len(images)
            }
        )
        
        raise


@celery_app.task(name="cleanup_completed_tasks")
def cleanup_completed_tasks(max_age_hours: int = 24) -> Dict[str, int]:
    """
    Clean up completed batch tasks older than specified age.
    
    Args:
        max_age_hours: Maximum age of tasks to keep
        
    Returns:
        Cleanup statistics
    """
    try:
        from celery.result import AsyncResult
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        total_checked = 0
        
        # Get all task results from backend
        # Note: This is a simplified implementation
        # In practice, you'd need to track task IDs more systematically
        
        logger.info(f"Starting cleanup of tasks older than {max_age_hours} hours")
        
        # This would need to be implemented based on your result backend
        # For Redis, you might query keys matching a pattern
        # For database backends, you'd query the results table
        
        logger.info(
            f"Cleanup completed",
            cleaned_count=cleaned_count,
            total_checked=total_checked
        )
        
        return {
            "cleaned_count": cleaned_count,
            "total_checked": total_checked,
            "cutoff_time": cutoff_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task cleanup failed: {e}")
        return {
            "error": str(e),
            "cleaned_count": 0,
            "total_checked": 0
        }


@celery_app.task(name="get_batch_task_status")
def get_batch_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a batch processing task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status information
    """
    try:
        from celery.result import AsyncResult
        
        # Get task result
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state == "PENDING":
            response = {
                "state": TaskStatus.QUEUED.value,
                "processed": 0,
                "total": 0,
                "percentage": 0
            }
        elif result.state == "PROGRESS":
            response = {
                "state": TaskStatus.PROCESSING.value,
                **result.info
            }
        elif result.state == "SUCCESS":
            response = {
                "state": TaskStatus.COMPLETED.value,
                **result.result
            }
        elif result.state == "FAILURE":
            response = {
                "state": TaskStatus.FAILED.value,
                "error": str(result.info),
                "processed": getattr(result.info, "processed", 0) if hasattr(result.info, "processed") else 0,
                "total": getattr(result.info, "total", 0) if hasattr(result.info, "total") else 0
            }
        else:
            response = {
                "state": result.state,
                "info": result.info
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        return {
            "state": TaskStatus.FAILED.value,
            "error": f"Failed to retrieve task status: {e}"
        }


# Task retry configuration
@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def retry_failed_batch_processing(self, task_id: str, images: List[str], config: Dict[str, Any]):
    """
    Retry failed batch processing with exponential backoff.
    
    This task can be used to retry failed batch operations with
    automatic retry logic and exponential backoff.
    """
    try:
        # Delegate to main processing task
        return process_batch_images.delay(task_id, images, config)
        
    except Exception as e:
        logger.error(f"Retry attempt failed for task {task_id}: {e}")
        raise self.retry(countdown=60 * (2 ** self.request.retries))