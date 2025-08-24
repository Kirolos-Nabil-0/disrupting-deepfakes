"""
Image processing module for the Deepfake Protection API.

This module handles image preprocessing, postprocessing, validation,
format conversion, and quality metrics calculation.
"""

import base64
import io
import math
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import numpy as np
from PIL import Image, ImageFile
import cv2
from skimage.metrics import structural_similarity as ssim

from app.models.schemas import ImageFormat
from app.core.exceptions import (
    InvalidImageError,
    ImageTooLargeError,
    UnsupportedImageFormatError
)
from app.core.logging import get_logger

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)


class ImageProcessor:
    """Handles all image processing operations for the API."""
    
    def __init__(
        self,
        max_size_mb: int = 10,
        max_dimension: int = 2048,
        supported_formats: list = None
    ):
        self.max_size_mb = max_size_mb
        self.max_dimension = max_dimension
        self.supported_formats = supported_formats or ["jpeg", "jpg", "png", "webp"]
        
        # Define preprocessing transforms
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Standard size for most models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        # Define postprocessing transforms
        self.postprocess_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Denormalize from [-1, 1] to [0, 1]
        ])
        
        logger.info(
            "ImageProcessor initialized",
            max_size_mb=max_size_mb,
            max_dimension=max_dimension,
            supported_formats=supported_formats
        )
    
    def decode_base64_image(self, base64_data: str) -> Image.Image:
        """
        Decode base64 image data to PIL Image.
        
        Args:
            base64_data: Base64 encoded image string
            
        Returns:
            PIL Image object
            
        Raises:
            InvalidImageError: If image data is invalid
        """
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            
            # Check size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                raise ImageTooLargeError(size_mb, self.max_size_mb)
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Validate image
            self._validate_image(image)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.debug(
                "Image decoded successfully",
                size_mb=round(size_mb, 2),
                dimensions=image.size,
                mode=image.mode
            )
            
            return image
            
        except (base64.binascii.Error, Exception) as e:
            if isinstance(e, (ImageTooLargeError, InvalidImageError)):
                raise
            raise InvalidImageError(f"Failed to decode image: {e}")
    
    def encode_image_to_base64(
        self, 
        image: Union[Image.Image, torch.Tensor], 
        format: ImageFormat = ImageFormat.JPEG,
        quality: int = 95
    ) -> str:
        """
        Encode PIL Image or tensor to base64 string.
        
        Args:
            image: PIL Image or tensor to encode
            format: Output image format
            quality: JPEG quality (1-100)
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert tensor to PIL if necessary
            if isinstance(image, torch.Tensor):
                image = self.tensor_to_pil(image)
            
            # Save to bytes
            buffer = io.BytesIO()
            save_format = format.value.upper()
            if save_format == 'JPEG':
                image.save(buffer, format=save_format, quality=quality, optimize=True)
            else:
                image.save(buffer, format=save_format, optimize=True)
            
            # Encode to base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            logger.debug(
                "Image encoded to base64",
                format=format.value,
                size_kb=round(len(image_bytes) / 1024, 2)
            )
            
            return base64_string
            
        except Exception as e:
            raise InvalidImageError(f"Failed to encode image: {e}")
    
    def _validate_image(self, image: Image.Image) -> None:
        """
        Validate image properties.
        
        Args:
            image: PIL Image to validate
            
        Raises:
            InvalidImageError: If image is invalid
            ImageTooLargeError: If image exceeds size limits
            UnsupportedImageFormatError: If format is not supported
        """
        # Check format
        if image.format and image.format.lower() not in self.supported_formats:
            raise UnsupportedImageFormatError(
                image.format.lower(),
                self.supported_formats
            )
        
        # Check dimensions
        width, height = image.size
        if width > self.max_dimension or height > self.max_dimension:
            raise ImageTooLargeError(
                0,  # Size in MB not applicable here
                self.max_size_mb,
                dimensions=(width, height)
            )
        
        # Check if image is valid
        try:
            image.verify()
        except Exception as e:
            raise InvalidImageError(f"Image verification failed: {e}")
        
        # Reload image after verify (verify() loads image data)
        image.load()
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Apply preprocessing transforms
            tensor = self.preprocess_transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            logger.debug(
                "Image preprocessed",
                input_size=image.size,
                output_shape=tensor.shape,
                tensor_range=(tensor.min().item(), tensor.max().item())
            )
            
            return tensor
            
        except Exception as e:
            raise InvalidImageError(f"Preprocessing failed: {e}")
    
    def postprocess_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Postprocess model output tensor.
        
        Args:
            tensor: Model output tensor to postprocess
            
        Returns:
            Postprocessed tensor in [0, 1] range
        """
        try:
            # Remove batch dimension if present
            if tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            
            # Denormalize from [-1, 1] to [0, 1]
            tensor = self.postprocess_transform(tensor)
            
            # Clamp to valid range
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
            logger.debug(
                "Image postprocessed",
                output_shape=tensor.shape,
                tensor_range=(tensor.min().item(), tensor.max().item())
            )
            
            return tensor
            
        except Exception as e:
            raise InvalidImageError(f"Postprocessing failed: {e}")
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image.
        
        Args:
            tensor: Input tensor in [0, 1] range
            
        Returns:
            PIL Image
        """
        try:
            # Ensure tensor is in correct format
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            # Convert to PIL
            image = to_pil_image(tensor)
            
            return image
            
        except Exception as e:
            raise InvalidImageError(f"Tensor to PIL conversion failed: {e}")
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to tensor.
        
        Args:
            image: PIL Image
            
        Returns:
            Tensor in [0, 1] range
        """
        try:
            tensor = to_tensor(image)
            return tensor
            
        except Exception as e:
            raise InvalidImageError(f"PIL to tensor conversion failed: {e}")
    
    def resize_image(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: PIL Image to resize
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        try:
            if maintain_aspect_ratio:
                # Calculate new size maintaining aspect ratio
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Pad to exact target size if needed
                new_image = Image.new('RGB', target_size, (0, 0, 0))
                paste_position = (
                    (target_size[0] - image.size[0]) // 2,
                    (target_size[1] - image.size[1]) // 2
                )
                new_image.paste(image, paste_position)
                return new_image
            else:
                return image.resize(target_size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            raise InvalidImageError(f"Image resizing failed: {e}")
    
    def calculate_protection_metrics(
        self,
        original: torch.Tensor,
        protected: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate protection quality metrics.
        
        Args:
            original: Original image tensor
            protected: Protected image tensor
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Ensure tensors are on CPU and have same shape
            original = original.detach().cpu()
            protected = protected.detach().cpu()
            
            if original.dim() == 4:
                original = original.squeeze(0)
            if protected.dim() == 4:
                protected = protected.squeeze(0)
            
            # Calculate perturbation
            perturbation = protected - original
            
            # L2 norm
            l2_norm = torch.norm(perturbation).item()
            
            # L-infinity norm
            linf_norm = torch.max(torch.abs(perturbation)).item()
            
            # Convert to numpy for SSIM calculation
            original_np = original.permute(1, 2, 0).numpy()
            protected_np = protected.permute(1, 2, 0).numpy()
            
            # Calculate SSIM
            ssim_value = ssim(
                original_np,
                protected_np,
                data_range=1.0,
                multichannel=True,
                channel_axis=2
            )
            
            # Calculate PSNR
            mse = torch.mean((original - protected) ** 2).item()
            if mse == 0:
                psnr_value = float('inf')
            else:
                psnr_value = 20 * math.log10(1.0 / math.sqrt(mse))
            
            metrics = {
                "l2_norm": round(l2_norm, 6),
                "linf_norm": round(linf_norm, 6),
                "ssim": round(ssim_value, 4),
                "psnr": round(psnr_value, 2) if psnr_value != float('inf') else 100.0
            }
            
            logger.debug("Protection metrics calculated", **metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            # Return default metrics on failure
            return {
                "l2_norm": 0.0,
                "linf_norm": 0.0,
                "ssim": 1.0,
                "psnr": 100.0
            }
    
    def create_perturbation_visualization(
        self,
        perturbation: torch.Tensor,
        amplification_factor: float = 10.0
    ) -> torch.Tensor:
        """
        Create visualization of perturbation.
        
        Args:
            perturbation: Perturbation tensor
            amplification_factor: Factor to amplify perturbations for visibility
            
        Returns:
            Visualization tensor
        """
        try:
            # Amplify perturbations for visualization
            amplified = perturbation * amplification_factor
            
            # Shift to [0, 1] range for visualization
            amplified = (amplified + 1.0) / 2.0
            
            # Clamp to valid range
            amplified = torch.clamp(amplified, 0.0, 1.0)
            
            return amplified
            
        except Exception as e:
            logger.error(f"Perturbation visualization failed: {e}")
            # Return zeros on failure
            return torch.zeros_like(perturbation)
    
    def validate_image_batch(self, images: list) -> list:
        """
        Validate a batch of base64 images.
        
        Args:
            images: List of base64 image strings
            
        Returns:
            List of validated PIL Images
            
        Raises:
            InvalidImageError: If any image is invalid
        """
        validated_images = []
        
        for i, image_data in enumerate(images):
            try:
                image = self.decode_base64_image(image_data)
                validated_images.append(image)
            except Exception as e:
                raise InvalidImageError(
                    f"Image {i} validation failed: {e}",
                    {"image_index": i, "error": str(e)}
                )
        
        logger.info(f"Validated batch of {len(validated_images)} images")
        return validated_images
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get detailed information about an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with image information
        """
        return {
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
            "format": image.format,
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        }