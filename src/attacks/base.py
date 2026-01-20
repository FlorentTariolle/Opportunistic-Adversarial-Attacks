"""Abstract base class for adversarial attacks."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseAttack(ABC):
    """Abstract base class for all adversarial attack implementations.
    
    All attack classes should inherit from this base class and implement
    the `generate` method.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        max_iterations: int = 1000,
        device: Optional[torch.device] = None
    ):
        """Initialize the base attack.
        
        Args:
            model: The target model to attack (should be in eval mode).
            epsilon: Maximum perturbation magnitude (L∞ norm).
            max_iterations: Maximum number of iterations for the attack.
            device: Device to run the attack on. If None, uses model's device.
        
        Raises:
            ValueError: If epsilon is negative or max_iterations is non-positive.
        """
        # Input validation
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
            self.model = self.model.to(device)
    
    @abstractmethod
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate adversarial examples.
        
        Args:
            x: Input images tensor of shape (batch_size, channels, height, width).
            y: True labels tensor of shape (batch_size,).
            **kwargs: Additional attack-specific parameters.
        
        Returns:
            Adversarial examples tensor of the same shape as x.
        """
        pass
    
    def clip_perturbation(
        self,
        x: torch.Tensor,
        perturbation: torch.Tensor,
        pixel_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """Clip perturbation to ensure valid pixel values and epsilon constraint.
        
        Args:
            x: Original images tensor.
            perturbation: Perturbation tensor to clip.
            pixel_range: Optional tuple (min, max) for valid pixel range.
                        If None, assumes [0, 1] for unnormalized images.
        
        Returns:
            Clipped perturbation tensor.
        """
        # Clip perturbation by epsilon (L∞ norm)
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        # Clip to valid pixel range
        if pixel_range is None:
            # Default: assume [0, 1] range for unnormalized images
            adversarial = torch.clamp(x + perturbation, 0.0, 1.0)
        else:
            # Use provided range (e.g., for normalized images)
            min_val, max_val = pixel_range
            adversarial = torch.clamp(x + perturbation, min_val, max_val)
        
        return adversarial - x
    
    def check_adversarial(
        self,
        x_adv: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Check if adversarial examples are successful.
        
        Args:
            x_adv: Adversarial examples tensor.
            y_true: True labels tensor.
        
        Returns:
            Boolean tensor indicating success for each example.
        """
        with torch.no_grad():
            logits = self.model(x_adv)
            predictions = torch.argmax(logits, dim=1)
            return predictions != y_true
