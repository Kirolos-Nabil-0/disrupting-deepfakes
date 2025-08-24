"""
Model management for the Deepfake Protection API.

This module handles loading, caching, and managing deepfake models
from the existing codebase (StarGAN, GANimation, pix2pixHD, CycleGAN).
"""

import os
import sys
import gc
import asyncio
import psutil
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Add the parent directories to Python path to import existing models
sys.path.append(os.path.join(os.path.dirname(__file__), "../../stargan"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ganimation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pix2pixHD"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../cyclegan"))

from app.models.schemas import ModelType, ModelStatus, ModelInfo
from app.core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    ModelNotLoadedError,
    GPUMemoryError,
    GPUNotAvailableError
)
from app.core.logging import model_logger, performance_logger, get_logger

logger = get_logger(__name__)


class ModelWrapper:
    """Wrapper class for loaded models with metadata."""
    
    def __init__(
        self,
        model: nn.Module,
        model_type: ModelType,
        device: str,
        version: str = "1.0",
        capabilities: Optional[List[str]] = None,
        supported_attacks: Optional[List[str]] = None
    ):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.version = version
        self.capabilities = capabilities or []
        self.supported_attacks = supported_attacks or ["fgsm", "pgd", "ifgsm"]
        self.loaded_at = datetime.utcnow()
        self.memory_usage_mb: Optional[float] = None
        
        # Calculate memory usage
        self._calculate_memory_usage()
    
    def _calculate_memory_usage(self) -> None:
        """Calculate approximate memory usage of the model."""
        try:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            self.memory_usage_mb = (param_size + buffer_size) / (1024 ** 2)
        except Exception as e:
            logger.warning(f"Failed to calculate memory usage: {e}")
            self.memory_usage_mb = None
    
    def to_info(self) -> ModelInfo:
        """Convert to ModelInfo schema."""
        return ModelInfo(
            name=self.model_type.value,
            status=ModelStatus.LOADED,
            version=self.version,
            capabilities=self.capabilities,
            supported_attacks=self.supported_attacks,
            memory_usage_mb=int(self.memory_usage_mb) if self.memory_usage_mb else None,
            loaded_at=self.loaded_at
        )


class ModelManager:
    """Manages loading and caching of deepfake models."""
    
    def __init__(
        self,
        models_dir: str = "./models",
        device: str = "auto",
        cache_models: bool = True,
        max_cached_models: int = 4
    ):
        self.models_dir = Path(models_dir)
        self.device = self._resolve_device(device)
        self.cache_models = cache_models
        self.max_cached_models = max_cached_models
        
        # Model cache
        self.loaded_models: Dict[ModelType, ModelWrapper] = {}
        self.model_paths: Dict[ModelType, str] = {}
        
        # Initialize model paths
        self._initialize_model_paths()
        
        logger.info(
            "ModelManager initialized",
            device=self.device,
            cache_models=cache_models,
            max_cached_models=max_cached_models
        )
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_model_paths(self) -> None:
        """Initialize paths to model files."""
        # These paths should be configured based on where models are stored
        # For now, using placeholder paths - these should be updated based on actual model locations
        
        base_path = self.models_dir.parent
        
        self.model_paths = {
            ModelType.STARGAN: str(base_path / "stargan" / "models"),
            ModelType.GANIMATION: str(base_path / "ganimation" / "models"),
            ModelType.PIX2PIXHD: str(base_path / "pix2pixHD" / "models"),
            ModelType.CYCLEGAN: str(base_path / "cyclegan" / "models")
        }
        
        logger.info("Model paths initialized", model_paths=self.model_paths)
    
    async def load_model(self, model_type: ModelType) -> ModelWrapper:
        """
        Load a specific model.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            Loaded model wrapper
            
        Raises:
            ModelNotFoundError: If model files are not found
            ModelLoadError: If model loading fails
            GPUMemoryError: If insufficient GPU memory
        """
        if model_type in self.loaded_models:
            logger.info(f"Model {model_type.value} already loaded")
            return self.loaded_models[model_type]
        
        start_time = model_logger.log_model_load_start(model_type.value, self.device)
        
        try:
            # Check if we need to free memory for new model
            if len(self.loaded_models) >= self.max_cached_models:
                await self._free_oldest_model()
            
            # Load the specific model
            model = await self._load_model_by_type(model_type)
            
            # Create wrapper
            wrapper = ModelWrapper(
                model=model,
                model_type=model_type,
                device=self.device,
                capabilities=self._get_model_capabilities(model_type),
                supported_attacks=["fgsm", "pgd", "ifgsm"]
            )
            
            # Cache the model
            if self.cache_models:
                self.loaded_models[model_type] = wrapper
            
            model_logger.log_model_load_end(
                model_type.value,
                self.device,
                start_time,
                success=True,
                memory_usage_mb=wrapper.memory_usage_mb
            )
            
            logger.info(
                f"Model {model_type.value} loaded successfully",
                memory_usage_mb=wrapper.memory_usage_mb
            )
            
            return wrapper
            
        except Exception as e:
            model_logger.log_model_load_end(
                model_type.value,
                self.device,
                start_time,
                success=False,
                error=str(e)
            )
            
            if isinstance(e, (ModelNotFoundError, ModelLoadError)):
                raise
            elif "out of memory" in str(e).lower():
                raise GPUMemoryError()
            else:
                raise ModelLoadError(model_type.value, str(e))
    
    async def _load_model_by_type(self, model_type: ModelType) -> nn.Module:
        """Load a specific model implementation."""
        try:
            if model_type == ModelType.STARGAN:
                return await self._load_stargan_model()
            elif model_type == ModelType.GANIMATION:
                return await self._load_ganimation_model()
            elif model_type == ModelType.PIX2PIXHD:
                return await self._load_pix2pixhd_model()
            elif model_type == ModelType.CYCLEGAN:
                return await self._load_cyclegan_model()
            else:
                raise ModelNotFoundError(model_type.value)
                
        except ImportError as e:
            raise ModelLoadError(
                model_type.value,
                f"Failed to import model components: {e}"
            )
    
    async def _load_stargan_model(self) -> nn.Module:
        """Load StarGAN model."""
        try:
            # Import StarGAN components
            from model import Generator
            from solver import Solver
            
            # This is a simplified loader - actual implementation would need
            # to load from checkpoint files and configure properly
            model = Generator(64, 5, 6)  # Example parameters
            
            # Load checkpoint if available
            checkpoint_path = os.path.join(self.model_paths[ModelType.STARGAN], "200000-G.ckpt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise ModelLoadError(ModelType.STARGAN.value, f"StarGAN loading failed: {e}")
    
    async def _load_ganimation_model(self) -> nn.Module:
        """Load GANimation model."""
        try:
            # Import GANimation components
            from model import GANimationModel
            
            # Simplified loader
            model = GANimationModel()
            
            # Load checkpoint if available
            checkpoint_path = os.path.join(self.model_paths[ModelType.GANIMATION], "model.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise ModelLoadError(ModelType.GANIMATION.value, f"GANimation loading failed: {e}")
    
    async def _load_pix2pixhd_model(self) -> nn.Module:
        """Load pix2pixHD model."""
        try:
            # Import pix2pixHD components
            sys.path.append(os.path.join(os.path.dirname(__file__), "../../pix2pixHD"))
            from models.networks import GlobalGenerator
            
            # Simplified loader
            model = GlobalGenerator(3, 3, 64, 4, norm_layer=torch.nn.InstanceNorm2d)
            
            # Load checkpoint if available
            checkpoint_path = os.path.join(self.model_paths[ModelType.PIX2PIXHD], "latest_net_G.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise ModelLoadError(ModelType.PIX2PIXHD.value, f"pix2pixHD loading failed: {e}")
    
    async def _load_cyclegan_model(self) -> nn.Module:
        """Load CycleGAN model."""
        try:
            # Import CycleGAN components
            sys.path.append(os.path.join(os.path.dirname(__file__), "../../cyclegan"))
            from models.networks import ResnetGenerator
            
            # Simplified loader
            model = ResnetGenerator(3, 3, 64, norm_layer=torch.nn.InstanceNorm2d)
            
            # Load checkpoint if available
            checkpoint_path = os.path.join(self.model_paths[ModelType.CYCLEGAN], "latest_net_G_A.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise ModelLoadError(ModelType.CYCLEGAN.value, f"CycleGAN loading failed: {e}")
    
    def _get_model_capabilities(self, model_type: ModelType) -> List[str]:
        """Get capabilities for a specific model type."""
        capabilities_map = {
            ModelType.STARGAN: ["facial_attributes", "domain_transfer"],
            ModelType.GANIMATION: ["facial_animation", "expression_synthesis"],
            ModelType.PIX2PIXHD: ["image_translation", "high_resolution"],
            ModelType.CYCLEGAN: ["unpaired_translation", "style_transfer"]
        }
        return capabilities_map.get(model_type, [])
    
    async def unload_model(self, model_type: ModelType) -> None:
        """Unload a specific model from memory."""
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if using CUDA
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_type.value} unloaded")
    
    async def _free_oldest_model(self) -> None:
        """Free the oldest loaded model to make space."""
        if not self.loaded_models:
            return
        
        # Find oldest model
        oldest_model = min(
            self.loaded_models.items(),
            key=lambda x: x[1].loaded_at
        )
        
        await self.unload_model(oldest_model[0])
        logger.info(f"Freed oldest model: {oldest_model[0].value}")
    
    async def load_default_models(self) -> None:
        """Load default models for the API."""
        default_models = [ModelType.STARGAN, ModelType.GANIMATION]
        
        for model_type in default_models:
            try:
                await self.load_model(model_type)
            except Exception as e:
                logger.error(f"Failed to load default model {model_type.value}: {e}")
    
    def get_model(self, model_type: ModelType) -> ModelWrapper:
        """
        Get a loaded model.
        
        Args:
            model_type: Type of model to get
            
        Returns:
            Model wrapper
            
        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        if model_type not in self.loaded_models:
            raise ModelNotLoadedError(model_type.value)
        
        return self.loaded_models[model_type]
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """Get information about all loaded models."""
        return [wrapper.to_info() for wrapper in self.loaded_models.values()]
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return [model_type.value for model_type in ModelType]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on loaded models."""
        health_info = {
            "loaded_models": len(self.loaded_models),
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "memory_usage": {}
        }
        
        # Add memory usage information
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            health_info["memory_usage"]["gpu_total_gb"] = round(gpu_memory, 2)
            health_info["memory_usage"]["gpu_allocated_gb"] = round(gpu_allocated, 2)
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        health_info["memory_usage"]["cpu_total_gb"] = round(cpu_memory.total / (1024**3), 2)
        health_info["memory_usage"]["cpu_used_gb"] = round(cpu_memory.used / (1024**3), 2)
        
        return health_info
    
    async def cleanup(self) -> None:
        """Clean up all loaded models and resources."""
        logger.info("Cleaning up ModelManager...")
        
        for model_type in list(self.loaded_models.keys()):
            await self.unload_model(model_type)
        
        logger.info("ModelManager cleanup completed")