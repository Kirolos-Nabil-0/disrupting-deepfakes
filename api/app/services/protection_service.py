"""
Image Protection Service - Main business logic layer.

This module orchestrates the entire image protection workflow,
combining model management, attack generation, and image processing.
"""

import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import torch
from PIL import Image

from app.models.schemas import (
    ProtectionRequest,
    ProtectionResponse,
    BatchProtectionRequest,
    BatchTaskResponse,
    BatchStatusResponse,
    BatchImageResult,
    BatchTaskProgress,
    ProtectionMetrics,
    ProtectionConfig,
    ModelType,
    AttackMethod,
    TaskStatus
)
from app.core.model_manager import ModelManager
from app.core.attacks import ModelSpecificAttackEngine
from app.core.image_processor import ImageProcessor
from app.core.exceptions import (
    ProtectionError,
    ModelNotLoadedError,
    ImageProcessingError,
    AttackGenerationError,
    BatchProcessingError
)
from app.core.logging import get_logger, batch_logger

logger = get_logger(__name__)


class ImageProtectionService:
    """Main service for image protection operations."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        image_processor: Optional[ImageProcessor] = None
    ):
        self.model_manager = model_manager
        self.image_processor = image_processor or ImageProcessor()
        self.attack_engine = ModelSpecificAttackEngine()
        
        # Task storage for batch processing
        self.batch_tasks: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ImageProtectionService initialized")
    
    async def protect_single_image(self, request: ProtectionRequest) -> ProtectionResponse:
        """
        Protect a single image with adversarial perturbations.
        
        Args:
            request: Protection request with image and parameters
            
        Returns:
            Protection response with protected image and metrics
            
        Raises:
            ProtectionError: If protection fails
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Starting single image protection",
                model_type=request.model_type.value,
                attack_method=request.attack_method.value
            )
            
            # Step 1: Decode and validate image
            original_image = self.image_processor.decode_base64_image(request.image)
            
            # Step 2: Load model if not already loaded
            model_wrapper = await self.model_manager.load_model(request.model_type)
            
            # Step 3: Preprocess image
            input_tensor = self.image_processor.preprocess_image(original_image)
            input_tensor = input_tensor.to(model_wrapper.device)
            
            # Step 4: Generate adversarial protection
            protected_tensor, perturbation_tensor = await self.attack_engine.generate_protection(
                images=input_tensor,
                model_wrapper=model_wrapper,
                attack_method=request.attack_method,
                attack_params=request.attack_params,
                target_attributes=request.target_attributes
            )
            
            # Step 5: Postprocess images
            protected_tensor = self.image_processor.postprocess_image(protected_tensor)
            original_tensor = self.image_processor.postprocess_image(input_tensor)
            
            # Step 6: Calculate protection metrics
            metrics = self.image_processor.calculate_protection_metrics(
                original_tensor,
                protected_tensor
            )
            
            # Step 7: Convert to output format
            protected_image_b64 = self.image_processor.encode_image_to_base64(
                protected_tensor,
                request.output_format
            )
            
            # Step 8: Create perturbation visualization (optional)
            perturbation_viz = None
            if perturbation_tensor is not None:
                viz_tensor = self.image_processor.create_perturbation_visualization(
                    perturbation_tensor
                )
                viz_tensor = self.image_processor.postprocess_image(viz_tensor)
                perturbation_viz = self.image_processor.encode_image_to_base64(
                    viz_tensor,
                    request.output_format
                )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = ProtectionResponse(
                protected_image=protected_image_b64,
                perturbation_map=perturbation_viz,
                protection_metrics=ProtectionMetrics(**metrics),
                processing_time_ms=processing_time_ms,
                model_used=request.model_type.value,
                attack_applied=request.attack_method.value
            )
            
            logger.info(
                "Single image protection completed",
                processing_time_ms=processing_time_ms,
                l2_norm=metrics["l2_norm"],
                ssim=metrics["ssim"]
            )
            
            return response
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Single image protection failed",
                error=str(e),
                processing_time_ms=processing_time_ms
            )
            
            if isinstance(e, ProtectionError):
                raise
            else:
                raise ProtectionError(f"Image protection failed: {e}")
    
    async def start_batch_protection(self, request: BatchProtectionRequest) -> BatchTaskResponse:
        """
        Start batch image protection.
        
        Args:
            request: Batch protection request
            
        Returns:
            Task information for tracking progress
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Store task information
        task_info = {
            "task_id": task_id,
            "status": TaskStatus.QUEUED,
            "images": request.images,
            "config": request.protection_config,
            "created_at": datetime.utcnow(),
            "total_images": len(request.images),
            "processed_images": 0,
            "results": [],
            "errors": []
        }
        
        self.batch_tasks[task_id] = task_info
        
        logger.info(
            "Batch protection task created",
            task_id=task_id,
            image_count=len(request.images),
            model_type=request.protection_config.model_type.value
        )
        
        # If async processing is requested, return immediately
        if request.async_processing:
            # In a real implementation, this would be handled by Celery or similar
            # For now, we'll simulate async processing
            return BatchTaskResponse(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                estimated_completion_ms=len(request.images) * 2000,  # Estimate 2s per image
                images_count=len(request.images)
            )
        else:
            # Process synchronously
            await self._process_batch_task(task_id)
            
            return BatchTaskResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                estimated_completion_ms=None,
                images_count=len(request.images)
            )
    
    async def _process_batch_task(self, task_id: str) -> None:
        """
        Process a batch protection task.
        
        Args:
            task_id: ID of the task to process
        """
        if task_id not in self.batch_tasks:
            raise BatchProcessingError(task_id, 0, 0, "Task not found")
        
        task_info = self.batch_tasks[task_id]
        task_info["status"] = TaskStatus.PROCESSING
        
        start_time = batch_logger.log_batch_start(
            task_id,
            task_info["total_images"],
            task_info["config"].model_type.value,
            task_info["config"].attack_method.value
        )
        
        successful = 0
        failed = 0
        
        try:
            # Load model once for the entire batch
            model_wrapper = await self.model_manager.load_model(
                task_info["config"].model_type
            )
            
            for i, image_data in enumerate(task_info["images"]):
                try:
                    # Create individual protection request
                    protection_request = ProtectionRequest(
                        image=image_data,
                        model_type=task_info["config"].model_type,
                        attack_method=task_info["config"].attack_method,
                        attack_params=task_info["config"].attack_params,
                        target_attributes=task_info["config"].target_attributes,
                        output_format=task_info["config"].output_format
                    )
                    
                    # Process single image
                    result = await self.protect_single_image(protection_request)
                    
                    # Store result
                    batch_result = BatchImageResult(
                        image_id=i,
                        protected_image=result.protected_image,
                        metrics=result.protection_metrics
                    )
                    task_info["results"].append(batch_result)
                    successful += 1
                    
                except Exception as e:
                    # Store error
                    error_result = BatchImageResult(
                        image_id=i,
                        protected_image="",
                        metrics=ProtectionMetrics(l2_norm=0, linf_norm=0, ssim=0, psnr=0),
                        error=str(e)
                    )
                    task_info["results"].append(error_result)
                    failed += 1
                    
                    logger.error(f"Batch image {i} processing failed", error=str(e))
                
                # Update progress
                task_info["processed_images"] = i + 1
                
                # Log progress every 10% or every 5 images
                if (i + 1) % max(1, task_info["total_images"] // 10) == 0 or (i + 1) % 5 == 0:
                    current_time = time.time()
                    duration_ms = (current_time - start_time) * 1000
                    batch_logger.log_batch_progress(
                        task_id,
                        i + 1,
                        task_info["total_images"],
                        duration_ms
                    )
            
            # Mark as completed
            task_info["status"] = TaskStatus.COMPLETED
            task_info["completed_at"] = datetime.utcnow()
            
            batch_logger.log_batch_end(
                task_id,
                start_time,
                successful,
                failed,
                task_info["total_images"]
            )
            
            logger.info(
                "Batch processing completed",
                task_id=task_id,
                successful=successful,
                failed=failed,
                total=task_info["total_images"]
            )
            
        except Exception as e:
            task_info["status"] = TaskStatus.FAILED
            task_info["error"] = str(e)
            task_info["completed_at"] = datetime.utcnow()
            
            batch_logger.log_batch_end(
                task_id,
                start_time,
                successful,
                failed,
                task_info["total_images"],
                error=str(e)
            )
            
            logger.error(f"Batch processing failed", task_id=task_id, error=str(e))
            raise BatchProcessingError(task_id, successful, task_info["total_images"], str(e))
    
    def get_batch_status(self, task_id: str) -> BatchStatusResponse:
        """
        Get the status of a batch protection task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Current task status and results
            
        Raises:
            BatchProcessingError: If task not found
        """
        if task_id not in self.batch_tasks:
            raise BatchProcessingError(task_id, 0, 0, "Task not found")
        
        task_info = self.batch_tasks[task_id]
        
        # Calculate progress
        progress = BatchTaskProgress(
            processed=task_info["processed_images"],
            total=task_info["total_images"],
            percentage=round(
                (task_info["processed_images"] / task_info["total_images"]) * 100, 1
            ) if task_info["total_images"] > 0 else 0
        )
        
        # Calculate processing time if completed
        processing_time_ms = None
        if "completed_at" in task_info:
            duration = task_info["completed_at"] - task_info["created_at"]
            processing_time_ms = int(duration.total_seconds() * 1000)
        
        # Return results only if completed
        results = None
        if task_info["status"] == TaskStatus.COMPLETED:
            results = task_info["results"]
        
        return BatchStatusResponse(
            task_id=task_id,
            status=task_info["status"],
            progress=progress,
            results=results,
            processing_time_ms=processing_time_ms,
            error=task_info.get("error")
        )
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed tasks older than specified age.
        
        Args:
            max_age_hours: Maximum age of tasks to keep
            
        Returns:
            Number of tasks cleaned up
        """
        current_time = datetime.utcnow()
        cleaned_count = 0
        
        for task_id in list(self.batch_tasks.keys()):
            task_info = self.batch_tasks[task_id]
            
            # Check if task is completed and old enough
            if (task_info["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                "completed_at" in task_info):
                
                age = current_time - task_info["completed_at"]
                if age.total_seconds() > max_age_hours * 3600:
                    del self.batch_tasks[task_id]
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old batch tasks")
        
        return cleaned_count
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the protection service.
        
        Returns:
            Health status information
        """
        try:
            # Check model manager health
            model_health = await self.model_manager.health_check()
            
            # Check batch processing status
            active_tasks = sum(
                1 for task in self.batch_tasks.values()
                if task["status"] in [TaskStatus.QUEUED, TaskStatus.PROCESSING]
            )
            
            completed_tasks = sum(
                1 for task in self.batch_tasks.values()
                if task["status"] == TaskStatus.COMPLETED
            )
            
            failed_tasks = sum(
                1 for task in self.batch_tasks.values()
                if task["status"] == TaskStatus.FAILED
            )
            
            return {
                "service_status": "healthy",
                "model_manager": model_health,
                "batch_processing": {
                    "active_tasks": active_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "total_tasks": len(self.batch_tasks)
                },
                "image_processor": {
                    "max_size_mb": self.image_processor.max_size_mb,
                    "max_dimension": self.image_processor.max_dimension,
                    "supported_formats": self.image_processor.supported_formats
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "service_status": "unhealthy",
                "error": str(e)
            }