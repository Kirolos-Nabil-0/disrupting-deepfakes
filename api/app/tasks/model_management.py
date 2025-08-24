"""
Celery tasks for model management operations.

This module contains Celery tasks for handling model loading,
unloading, and health check operations.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime

from app.core.celery_app import celery_app
from app.core.model_manager import ModelManager
from app.models.schemas import ModelType, ModelStatus
from app.core.config import get_settings
from app.core.logging import get_logger, model_logger

logger = get_logger(__name__)
settings = get_settings()


@celery_app.task(name="preload_models")
def preload_models(model_types: List[str] = None) -> Dict[str, Any]:
    """
    Preload specified models into memory.
    
    Args:
        model_types: List of model types to load (defaults to all)
        
    Returns:
        Loading results for each model
    """
    async def _preload_models():
        try:
            model_manager = ModelManager(
                models_dir=settings.models_dir,
                device=settings.device,
                cache_models=True
            )
            
            # Use specified models or default to all
            if model_types is None:
                models_to_load = [ModelType.STARGAN, ModelType.GANIMATION]
            else:
                models_to_load = [ModelType(mt) for mt in model_types]
            
            results = {}
            
            for model_type in models_to_load:
                try:
                    start_time = model_logger.log_model_load_start(
                        model_type.value,
                        model_manager.device
                    )
                    
                    wrapper = await model_manager.load_model(model_type)
                    
                    model_logger.log_model_load_end(
                        model_type.value,
                        model_manager.device,
                        start_time,
                        success=True,
                        memory_usage_mb=wrapper.memory_usage_mb
                    )
                    
                    results[model_type.value] = {
                        "status": "success",
                        "memory_usage_mb": wrapper.memory_usage_mb,
                        "loaded_at": wrapper.loaded_at.isoformat()
                    }
                    
                except Exception as e:
                    model_logger.log_model_load_end(
                        model_type.value,
                        model_manager.device,
                        start_time if 'start_time' in locals() else 0,
                        success=False,
                        error=str(e)
                    )
                    
                    results[model_type.value] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            await model_manager.cleanup()
            
            logger.info(
                "Model preloading completed",
                results=results,
                successful=sum(1 for r in results.values() if r["status"] == "success"),
                failed=sum(1 for r in results.values() if r["status"] == "failed")
            )
            
            return {
                "task": "preload_models",
                "completed_at": datetime.utcnow().isoformat(),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
            return {
                "task": "preload_models",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e),
                "results": {}
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_preload_models())
    finally:
        loop.close()


@celery_app.task(name="health_check_models")
def health_check_models() -> Dict[str, Any]:
    """
    Perform health checks on loaded models.
    
    Returns:
        Health check results for all models
    """
    async def _health_check():
        try:
            model_manager = ModelManager(
                models_dir=settings.models_dir,
                device=settings.device,
                cache_models=True
            )
            
            health_info = await model_manager.health_check()
            
            logger.info("Model health check completed", health_info=health_info)
            
            return {
                "task": "health_check_models",
                "completed_at": datetime.utcnow().isoformat(),
                "health_info": health_info,
                "status": "healthy" if health_info.get("loaded_models", 0) > 0 else "no_models"
            }
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                "task": "health_check_models",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "unhealthy"
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_health_check())
    finally:
        loop.close()


@celery_app.task(name="cleanup_model_cache")
def cleanup_model_cache(force: bool = False) -> Dict[str, Any]:
    """
    Clean up model cache and free memory.
    
    Args:
        force: Whether to force unload all models
        
    Returns:
        Cleanup results
    """
    async def _cleanup():
        try:
            model_manager = ModelManager(
                models_dir=settings.models_dir,
                device=settings.device,
                cache_models=True
            )
            
            initial_models = len(model_manager.loaded_models)
            
            if force:
                # Unload all models
                for model_type in list(model_manager.loaded_models.keys()):
                    await model_manager.unload_model(model_type)
            
            await model_manager.cleanup()
            
            final_models = len(model_manager.loaded_models)
            
            logger.info(
                "Model cache cleanup completed",
                initial_models=initial_models,
                final_models=final_models,
                force=force
            )
            
            return {
                "task": "cleanup_model_cache",
                "completed_at": datetime.utcnow().isoformat(),
                "initial_models": initial_models,
                "final_models": final_models,
                "freed_models": initial_models - final_models,
                "force": force
            }
            
        except Exception as e:
            logger.error(f"Model cache cleanup failed: {e}")
            return {
                "task": "cleanup_model_cache",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_cleanup())
    finally:
        loop.close()


@celery_app.task(name="benchmark_model_performance")
def benchmark_model_performance(model_type: str, num_iterations: int = 10) -> Dict[str, Any]:
    """
    Benchmark model performance for optimization.
    
    Args:
        model_type: Type of model to benchmark
        num_iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results
    """
    async def _benchmark():
        try:
            import time
            import torch
            
            model_manager = ModelManager(
                models_dir=settings.models_dir,
                device=settings.device,
                cache_models=True
            )
            
            # Load model
            model_enum = ModelType(model_type)
            wrapper = await model_manager.load_model(model_enum)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 256, 256).to(wrapper.device)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = wrapper.model(dummy_input)
            
            # Benchmark
            times = []
            for i in range(num_iterations):
                start_time = time.time()
                
                with torch.no_grad():
                    output = wrapper.model(dummy_input)
                
                if wrapper.device.startswith("cuda"):
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            await model_manager.cleanup()
            
            results = {
                "model_type": model_type,
                "device": wrapper.device,
                "num_iterations": num_iterations,
                "avg_time_ms": round(avg_time, 2),
                "min_time_ms": round(min_time, 2),
                "max_time_ms": round(max_time, 2),
                "memory_usage_mb": wrapper.memory_usage_mb
            }
            
            logger.info("Model benchmark completed", results=results)
            
            return {
                "task": "benchmark_model_performance",
                "completed_at": datetime.utcnow().isoformat(),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {e}")
            return {
                "task": "benchmark_model_performance",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_benchmark())
    finally:
        loop.close()