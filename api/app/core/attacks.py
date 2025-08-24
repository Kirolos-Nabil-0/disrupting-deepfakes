"""
Adversarial attack factory and implementations for the Deepfake Protection API.

This module integrates the existing attack implementations from the codebase
and provides a unified interface for generating adversarial perturbations.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Add attack modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../stargan"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ganimation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pix2pixHD"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../cyclegan"))

from app.models.schemas import AttackMethod, AttackParams, ModelType
from app.core.exceptions import AttackGenerationError, InvalidAttackParametersError
from app.core.logging import attack_logger, get_logger

logger = get_logger(__name__)


class BaseAttack(ABC):
    """Base class for adversarial attacks."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        epsilon: float = 0.05,
        **kwargs
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.model.eval()
    
    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adversarial examples.
        
        Args:
            images: Input images tensor
            targets: Target outputs for targeted attacks
            **kwargs: Additional attack-specific parameters
            
        Returns:
            Tuple of (adversarial_images, perturbations)
        """
        pass
    
    def _clamp_perturbation(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Clamp perturbation to epsilon bounds."""
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)
    
    def _clamp_images(self, images: torch.Tensor) -> torch.Tensor:
        """Clamp images to valid range [-1, 1]."""
        return torch.clamp(images, -1.0, 1.0)


class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method attack implementation."""
    
    def __init__(self, model: nn.Module, device: str, epsilon: float = 0.05, **kwargs):
        super().__init__(model, device, epsilon, **kwargs)
        self.attack_name = "FGSM"
    
    def generate(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_attributes: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate FGSM adversarial examples."""
        
        images = images.to(self.device)
        images.requires_grad_(True)
        
        try:
            # Forward pass
            if target_attributes is not None:
                # For models that use attribute vectors (e.g., StarGAN)
                outputs = self.model(images, target_attributes)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take first output if tuple
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            # Calculate loss
            if targets is not None:
                # Targeted attack - minimize distance to target
                loss = F.mse_loss(outputs, targets)
            else:
                # Untargeted attack - maximize output change
                loss = -F.mse_loss(outputs, images)
            
            # Backward pass
            loss.backward()
            
            # Generate perturbation
            grad_sign = images.grad.sign()
            perturbation = self.epsilon * grad_sign
            
            # Apply perturbation
            adversarial_images = images + perturbation
            adversarial_images = self._clamp_images(adversarial_images)
            
            # Calculate final perturbation
            final_perturbation = adversarial_images - images
            
            return adversarial_images.detach(), final_perturbation.detach()
            
        except Exception as e:
            raise AttackGenerationError("fgsm", f"FGSM generation failed: {e}")


class PGDAttack(BaseAttack):
    """Projected Gradient Descent attack implementation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        epsilon: float = 0.05,
        iterations: int = 10,
        step_size: float = 0.01,
        random_start: bool = True,
        **kwargs
    ):
        super().__init__(model, device, epsilon, **kwargs)
        self.iterations = iterations
        self.step_size = step_size
        self.random_start = random_start
        self.attack_name = "PGD"
    
    def generate(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_attributes: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate PGD adversarial examples."""
        
        original_images = images.to(self.device)
        
        # Initialize adversarial images
        if self.random_start:
            # Random initialization within epsilon ball
            noise = torch.empty_like(original_images).uniform_(-self.epsilon, self.epsilon)
            adversarial_images = original_images + noise
            adversarial_images = self._clamp_images(adversarial_images)
        else:
            adversarial_images = original_images.clone()
        
        try:
            for i in range(self.iterations):
                adversarial_images.requires_grad_(True)
                
                # Forward pass
                if target_attributes is not None:
                    outputs = self.model(adversarial_images, target_attributes)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = self.model(adversarial_images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                
                # Calculate loss
                if targets is not None:
                    # Targeted attack
                    loss = F.mse_loss(outputs, targets)
                else:
                    # Untargeted attack
                    loss = -F.mse_loss(outputs, original_images)
                
                # Backward pass
                loss.backward()
                
                # Update adversarial images
                grad = adversarial_images.grad.sign()
                adversarial_images = adversarial_images + self.step_size * grad
                
                # Project back to epsilon ball
                perturbation = adversarial_images - original_images
                perturbation = self._clamp_perturbation(perturbation)
                adversarial_images = original_images + perturbation
                adversarial_images = self._clamp_images(adversarial_images)
                adversarial_images = adversarial_images.detach()
            
            final_perturbation = adversarial_images - original_images
            
            return adversarial_images, final_perturbation
            
        except Exception as e:
            raise AttackGenerationError("pgd", f"PGD generation failed: {e}")


class IFGSMAttack(BaseAttack):
    """Iterative Fast Gradient Sign Method attack implementation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        epsilon: float = 0.05,
        iterations: int = 10,
        step_size: float = 0.01,
        **kwargs
    ):
        super().__init__(model, device, epsilon, **kwargs)
        self.iterations = iterations
        self.step_size = step_size
        self.attack_name = "I-FGSM"
    
    def generate(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_attributes: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate I-FGSM adversarial examples."""
        
        original_images = images.to(self.device)
        adversarial_images = original_images.clone()
        
        try:
            for i in range(self.iterations):
                adversarial_images.requires_grad_(True)
                
                # Forward pass
                if target_attributes is not None:
                    outputs = self.model(adversarial_images, target_attributes)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = self.model(adversarial_images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                
                # Calculate loss
                if targets is not None:
                    loss = F.mse_loss(outputs, targets)
                else:
                    loss = -F.mse_loss(outputs, original_images)
                
                # Backward pass
                loss.backward()
                
                # Update with sign of gradient
                grad_sign = adversarial_images.grad.sign()
                adversarial_images = adversarial_images + self.step_size * grad_sign
                
                # Clip to epsilon ball and valid image range
                perturbation = adversarial_images - original_images
                perturbation = self._clamp_perturbation(perturbation)
                adversarial_images = original_images + perturbation
                adversarial_images = self._clamp_images(adversarial_images)
                adversarial_images = adversarial_images.detach()
            
            final_perturbation = adversarial_images - original_images
            
            return adversarial_images, final_perturbation
            
        except Exception as e:
            raise AttackGenerationError("ifgsm", f"I-FGSM generation failed: {e}")


class AttackFactory:
    """Factory for creating adversarial attacks."""
    
    @staticmethod
    def create_attack(
        attack_method: AttackMethod,
        model: nn.Module,
        device: str,
        attack_params: AttackParams
    ) -> BaseAttack:
        """
        Create an attack instance based on method and parameters.
        
        Args:
            attack_method: Type of attack to create
            model: Target model
            device: Device to run attack on
            attack_params: Attack parameters
            
        Returns:
            Attack instance
            
        Raises:
            InvalidAttackParametersError: If parameters are invalid
        """
        
        # Validate parameters
        AttackFactory._validate_params(attack_params)
        
        try:
            if attack_method == AttackMethod.FGSM:
                return FGSMAttack(
                    model=model,
                    device=device,
                    epsilon=attack_params.epsilon
                )
            
            elif attack_method == AttackMethod.PGD:
                return PGDAttack(
                    model=model,
                    device=device,
                    epsilon=attack_params.epsilon,
                    iterations=attack_params.iterations,
                    step_size=attack_params.step_size,
                    random_start=attack_params.random_start
                )
            
            elif attack_method == AttackMethod.IFGSM:
                return IFGSMAttack(
                    model=model,
                    device=device,
                    epsilon=attack_params.epsilon,
                    iterations=attack_params.iterations,
                    step_size=attack_params.step_size
                )
            
            else:
                raise InvalidAttackParametersError(
                    "attack_method",
                    attack_method.value,
                    "Must be one of: fgsm, pgd, ifgsm"
                )
                
        except Exception as e:
            if isinstance(e, InvalidAttackParametersError):
                raise
            raise AttackGenerationError(
                attack_method.value,
                f"Failed to create attack: {e}"
            )
    
    @staticmethod
    def _validate_params(attack_params: AttackParams) -> None:
        """Validate attack parameters."""
        
        if attack_params.epsilon <= 0 or attack_params.epsilon > 1.0:
            raise InvalidAttackParametersError(
                "epsilon",
                attack_params.epsilon,
                "Must be in range (0, 1.0]"
            )
        
        if attack_params.iterations <= 0:
            raise InvalidAttackParametersError(
                "iterations",
                attack_params.iterations,
                "Must be positive integer"
            )
        
        if attack_params.step_size <= 0:
            raise InvalidAttackParametersError(
                "step_size",
                attack_params.step_size,
                "Must be positive"
            )
        
        if attack_params.step_size > attack_params.epsilon:
            raise InvalidAttackParametersError(
                "step_size",
                attack_params.step_size,
                f"Must be <= epsilon ({attack_params.epsilon})"
            )
    
    @staticmethod
    def get_available_methods() -> Dict[str, Dict[str, Any]]:
        """Get information about available attack methods."""
        return {
            "fgsm": {
                "name": "Fast Gradient Sign Method",
                "description": "Single-step gradient-based attack",
                "parameters": {
                    "epsilon": {
                        "type": "float",
                        "range": [0.01, 0.1],
                        "default": 0.05,
                        "description": "Maximum perturbation magnitude"
                    }
                }
            },
            "pgd": {
                "name": "Projected Gradient Descent",
                "description": "Multi-step gradient-based attack with projection",
                "parameters": {
                    "epsilon": {
                        "type": "float",
                        "range": [0.01, 0.1],
                        "default": 0.05,
                        "description": "Maximum perturbation magnitude"
                    },
                    "iterations": {
                        "type": "int",
                        "range": [1, 50],
                        "default": 10,
                        "description": "Number of attack iterations"
                    },
                    "step_size": {
                        "type": "float",
                        "range": [0.001, 0.05],
                        "default": 0.01,
                        "description": "Step size per iteration"
                    },
                    "random_start": {
                        "type": "bool",
                        "default": True,
                        "description": "Whether to use random initialization"
                    }
                }
            },
            "ifgsm": {
                "name": "Iterative Fast Gradient Sign Method",
                "description": "Multi-step FGSM attack",
                "parameters": {
                    "epsilon": {
                        "type": "float",
                        "range": [0.01, 0.1],
                        "default": 0.05,
                        "description": "Maximum perturbation magnitude"
                    },
                    "iterations": {
                        "type": "int",
                        "range": [1, 50],
                        "default": 10,
                        "description": "Number of attack iterations"
                    },
                    "step_size": {
                        "type": "float",
                        "range": [0.001, 0.05],
                        "default": 0.01,
                        "description": "Step size per iteration"
                    }
                }
            }
        }


class ModelSpecificAttackEngine:
    """Engine for generating model-specific adversarial attacks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def generate_protection(
        self,
        images: torch.Tensor,
        model_wrapper,
        attack_method: AttackMethod,
        attack_params: AttackParams,
        target_attributes: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adversarial protection for images.
        
        Args:
            images: Input images tensor
            model_wrapper: Loaded model wrapper
            attack_method: Attack method to use
            attack_params: Attack parameters
            target_attributes: Target attributes for facial manipulation models
            
        Returns:
            Tuple of (protected_images, perturbations)
        """
        
        start_time = attack_logger.log_attack_start(
            attack_method.value,
            model_wrapper.model_type.value,
            attack_params.dict()
        )
        
        try:
            # Create attack instance
            attack = AttackFactory.create_attack(
                attack_method,
                model_wrapper.model,
                model_wrapper.device,
                attack_params
            )
            
            # Prepare target attributes if needed
            target_tensor = None
            if target_attributes and model_wrapper.model_type == ModelType.STARGAN:
                target_tensor = self._prepare_stargan_attributes(
                    target_attributes,
                    images.size(0),
                    model_wrapper.device
                )
            
            # Generate adversarial examples
            protected_images, perturbations = attack.generate(
                images,
                target_attributes=target_tensor
            )
            
            # Calculate perturbation norm
            perturbation_norm = torch.norm(perturbations).item()
            
            attack_logger.log_attack_end(
                attack_method.value,
                model_wrapper.model_type.value,
                start_time,
                success=True,
                perturbation_norm=perturbation_norm
            )
            
            return protected_images, perturbations
            
        except Exception as e:
            attack_logger.log_attack_end(
                attack_method.value,
                model_wrapper.model_type.value,
                start_time,
                success=False,
                error=str(e)
            )
            raise
    
    def _prepare_stargan_attributes(
        self,
        target_attributes: list,
        batch_size: int,
        device: str
    ) -> torch.Tensor:
        """Prepare attribute vector for StarGAN."""
        # This is a simplified implementation
        # In practice, you'd need to map attribute names to indices
        attribute_names = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
        
        # Create binary vector
        attr_vector = torch.zeros(len(attribute_names))
        for attr in target_attributes:
            if attr in attribute_names:
                idx = attribute_names.index(attr)
                attr_vector[idx] = 1.0
        
        # Repeat for batch
        attr_batch = attr_vector.unsqueeze(0).repeat(batch_size, 1)
        return attr_batch.to(device)