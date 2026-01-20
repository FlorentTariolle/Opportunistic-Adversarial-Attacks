"""SimBA (Simple Black-box Adversarial) attack implementation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from .base import BaseAttack


class SimBA(BaseAttack):
    """Simple Black-box Adversarial (SimBA) attack.
    
    SimBA is a query-efficient black-box attack that uses random search
    in the DCT (Discrete Cosine Transform) space or pixel space.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        max_iterations: int = 1000,
        device: Optional[torch.device] = None,
        use_dct: bool = True,
        block_size: int = 8
    ):
        """Initialize SimBA attack.
        
        Args:
            model: The target model to attack.
            epsilon: Maximum perturbation magnitude (L∞ norm).
            max_iterations: Maximum number of iterations.
            device: Device to run the attack on.
            use_dct: If True, use DCT space for perturbations; else use pixel space.
            block_size: Block size for DCT transform (only used if use_dct=True).
        """
        super().__init__(model, epsilon, max_iterations, device)
        self.use_dct = use_dct
        self.block_size = block_size
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        track_confidence: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Generate adversarial examples using SimBA.
        
        Args:
            x: Input images tensor of shape (batch_size, channels, height, width).
            y: True labels tensor of shape (batch_size,).
            track_confidence: If True, track confidence values during attack.
            **kwargs: Additional parameters (not used currently).
        
        Returns:
            Adversarial examples tensor.
        
        Raises:
            ValueError: If input shapes are invalid or batch sizes don't match.
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got {x.dim()}D")
        if y.dim() != 1:
            raise ValueError(f"Expected 1D tensor for labels, got {y.dim()}D")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x has {x.shape[0]} samples, y has {y.shape[0]}")
        
        batch_size = x.shape[0]
        x_adv = x.clone().to(self.device)
        y = y.to(self.device)
        
        # Initialize confidence_history
        self.confidence_history = None
        
        # Process each image in the batch
        if track_confidence and batch_size == 1:
            result = self._attack_single_image(
                x[0], y[0], track_confidence=True
            )
            if isinstance(result, tuple) and len(result) == 2:
                x_adv[0], self.confidence_history = result
            else:
                # Fallback if result is not a tuple (shouldn't happen)
                x_adv[0] = result if not isinstance(result, tuple) else result[0]
                self.confidence_history = None
        else:
            if track_confidence:
                # If track_confidence is True but batch_size != 1, warn user
                import warnings
                warnings.warn(f"track_confidence=True is only supported for batch_size=1. Got batch_size={batch_size}. Confidence tracking disabled.")
            for i in range(batch_size):
                result = self._attack_single_image(x[i], y[i], track_confidence=False)
                if isinstance(result, tuple):
                    x_adv[i] = result[0]
                else:
                    x_adv[i] = result
        
        return x_adv
    
    def _attack_single_image(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        track_confidence: bool = False
    ) -> tuple:
        """Attack a single image.
        
        Args:
            x: Single image tensor of shape (channels, height, width).
            y_true: True label tensor (scalar).
            track_confidence: If True, track confidence values during attack.
        
        Returns:
            If track_confidence: (adversarial image tensor, confidence_history dict)
            Else: adversarial image tensor
        """
        x_adv = x.clone()
        confidence_history = {'iterations': [], 'original_class': [], 'max_other_class': []}
        
        # Get initial confidence
        if track_confidence:
            with torch.no_grad():
                logits = self.model(x_adv.unsqueeze(0))
                probs = torch.nn.functional.softmax(logits, dim=1)
                original_conf = probs[0][y_true].item()
                # Get max confidence excluding original class
                probs_excluding_original = probs[0].clone()
                probs_excluding_original[y_true] = -1.0  # Set to -1 to exclude
                max_other_conf = probs_excluding_original.max().item()
                confidence_history['iterations'].append(0)
                confidence_history['original_class'].append(original_conf)
                confidence_history['max_other_class'].append(max_other_conf)
        
        # Check if already misclassified
        if self._is_misclassified(x_adv.unsqueeze(0), y_true.unsqueeze(0)):
            if track_confidence:
                return x_adv, confidence_history
            return x_adv
        
        # Generate candidate indices (memory-efficient)
        if self.use_dct:
            candidate_indices = self._generate_dct_candidate_indices(x)
        else:
            candidate_indices = self._generate_pixel_candidate_indices(x)
        
        # Randomly shuffle candidate indices
        num_candidates = len(candidate_indices)
        shuffled_indices = torch.randperm(num_candidates, device=self.device)
        
        # Try each candidate perturbation
        for iteration in range(min(self.max_iterations, num_candidates)):
            idx = shuffled_indices[iteration]
            candidate_idx = candidate_indices[idx]
            
            # Generate perturbation on-the-fly
            perturbation = self._create_perturbation(x.shape, candidate_idx)
            
            # Track confidence at start of iteration (before trying perturbations)
            if track_confidence:
                should_track = (
                    iteration < 50 or  # Track every iteration for first 50
                    iteration % 10 == 0  # Then every 10 iterations
                )
                if should_track:
                    with torch.no_grad():
                        logits = self.model(x_adv.unsqueeze(0))
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        original_conf = probs[0][y_true].item()
                        probs_excluding_original = probs[0].clone()
                        probs_excluding_original[y_true] = -1.0
                        max_other_conf = probs_excluding_original.max().item()
                        confidence_history['iterations'].append(iteration + 1)
                        confidence_history['original_class'].append(original_conf)
                        confidence_history['max_other_class'].append(max_other_conf)
            
            # Get current confidence once (for efficiency)
            with torch.no_grad():
                logits_current = self.model(x_adv.unsqueeze(0))
                probs_current = torch.nn.functional.softmax(logits_current, dim=1)
                current_conf = probs_current[0][y_true].item()
                current_pred = torch.argmax(logits_current, dim=1).item()
            
            # Try positive perturbation
            x_candidate = x_adv + perturbation
            # Clip perturbation to respect epsilon constraint and valid pixel range
            # For normalized ImageNet images, use reasonable bounds
            perturbation_clipped = self.clip_perturbation(
                x_adv, perturbation, pixel_range=(-3.0, 3.0)
            )
            x_candidate = x_adv + perturbation_clipped
            x_candidate_batch = x_candidate.unsqueeze(0)
            
            # Check candidate in one forward pass
            with torch.no_grad():
                logits_candidate = self.model(x_candidate_batch)
                probs_candidate = torch.nn.functional.softmax(logits_candidate, dim=1)
                candidate_conf = probs_candidate[0][y_true].item()
                candidate_pred = torch.argmax(logits_candidate, dim=1).item()
            
            # Check if misclassified (success!)
            if candidate_pred != y_true.item():
                x_adv = x_candidate
                if track_confidence:
                    probs_excluding_original = probs_candidate[0].clone()
                    probs_excluding_original[y_true] = -1.0
                    max_other_conf = probs_excluding_original.max().item()
                    if confidence_history['iterations'] and confidence_history['iterations'][-1] == iteration + 1:
                        confidence_history['original_class'][-1] = candidate_conf
                        confidence_history['max_other_class'][-1] = max_other_conf
                    else:
                        confidence_history['iterations'].append(iteration + 1)
                        confidence_history['original_class'].append(candidate_conf)
                        confidence_history['max_other_class'].append(max_other_conf)
                return x_adv, confidence_history if track_confidence else x_adv
            
            # Check if it reduces confidence (SimBA criterion: accept if confidence reduced)
            if candidate_conf < current_conf:
                x_adv = x_candidate
                continue  # Accept this perturbation and move to next iteration
            
            # Try negative perturbation
            negative_perturbation = -perturbation
            # Clip perturbation to respect epsilon constraint and valid pixel range
            perturbation_clipped = self.clip_perturbation(
                x_adv, negative_perturbation, pixel_range=(-3.0, 3.0)
            )
            x_candidate = x_adv + perturbation_clipped
            x_candidate_batch = x_candidate.unsqueeze(0)
            
            # Check candidate in one forward pass
            with torch.no_grad():
                logits_candidate = self.model(x_candidate_batch)
                probs_candidate = torch.nn.functional.softmax(logits_candidate, dim=1)
                candidate_conf = probs_candidate[0][y_true].item()
                candidate_pred = torch.argmax(logits_candidate, dim=1).item()
            
            # Check if misclassified (success!)
            if candidate_pred != y_true.item():
                x_adv = x_candidate
                if track_confidence:
                    probs_excluding_original = probs_candidate[0].clone()
                    probs_excluding_original[y_true] = -1.0
                    max_other_conf = probs_excluding_original.max().item()
                    if confidence_history['iterations'] and confidence_history['iterations'][-1] == iteration + 1:
                        confidence_history['original_class'][-1] = candidate_conf
                        confidence_history['max_other_class'][-1] = max_other_conf
                    else:
                        confidence_history['iterations'].append(iteration + 1)
                        confidence_history['original_class'].append(candidate_conf)
                        confidence_history['max_other_class'].append(max_other_conf)
                return x_adv, confidence_history if track_confidence else x_adv
            
            # Check if it reduces confidence (SimBA criterion: accept if confidence reduced)
            if candidate_conf < current_conf:
                x_adv = x_candidate
                continue  # Accept this perturbation and move to next iteration
        
        if track_confidence:
            return x_adv, confidence_history
        return x_adv
    
    def _generate_dct_candidate_indices(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Generate candidate indices in DCT space.
        
        Args:
            x: Single image tensor of shape (channels, height, width).
        
        Returns:
            Tensor of candidate indices (channel, row, col).
        
        Note:
            Currently falls back to pixel-space candidates. A full DCT implementation
            would transform the image to DCT space and generate candidates there.
        """
        # TODO: Implement proper DCT-based candidate generation
        # This would involve:
        # 1. Apply DCT transform to image blocks
        # 2. Generate candidate indices in DCT coefficient space
        # 3. Return indices that can be used to modify DCT coefficients
        # For now, we'll use pixel-space candidates as a fallback.
        return self._generate_pixel_candidate_indices(x)
    
    def _generate_pixel_candidate_indices(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Generate candidate indices in pixel space (memory-efficient).
        
        Args:
            x: Single image tensor of shape (channels, height, width).
        
        Returns:
            Tensor of candidate indices of shape (num_candidates, 3) where
            each row is (channel, row, col).
        """
        c, h, w = x.shape
        
        # Generate all (channel, row, col) combinations
        channels = torch.arange(c, device=self.device)
        rows = torch.arange(h, device=self.device)
        cols = torch.arange(w, device=self.device)
        
        # Create meshgrid and flatten
        ch_grid, row_grid, col_grid = torch.meshgrid(
            channels, rows, cols, indexing='ij'
        )
        
        # Stack and reshape to (num_candidates, 3)
        candidate_indices = torch.stack([
            ch_grid.flatten(),
            row_grid.flatten(),
            col_grid.flatten()
        ], dim=1)
        
        return candidate_indices
    
    def _create_perturbation(
        self,
        shape: Tuple[int, int, int],
        candidate_idx: torch.Tensor
    ) -> torch.Tensor:
        """Create a single perturbation tensor from candidate index.
        
        Args:
            shape: Image shape (channels, height, width).
            candidate_idx: Index tensor of shape (3,) with (channel, row, col).
        
        Returns:
            Perturbation tensor of the given shape.
        
        Raises:
            ValueError: If candidate_idx has invalid shape or indices out of bounds.
        """
        if candidate_idx.shape != (3,):
            raise ValueError(f"Expected candidate_idx of shape (3,), got {candidate_idx.shape}")
        
        c, h, w = shape
        perturbation = torch.zeros(c, h, w, device=self.device)
        
        ch, row, col = candidate_idx[0].item(), candidate_idx[1].item(), candidate_idx[2].item()
        
        # Validate indices are within bounds
        if not (0 <= ch < c and 0 <= row < h and 0 <= col < w):
            raise ValueError(
                f"Index out of bounds: ({ch}, {row}, {col}) for shape ({c}, {h}, {w})"
            )
        
        perturbation[ch, row, col] = self.epsilon
        
        return perturbation
    
    def _is_misclassified(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor
    ) -> bool:
        """Check if image is misclassified.
        
        Args:
            x: Image tensor of shape (1, channels, height, width).
            y_true: True label tensor of shape (1,).
        
        Returns:
            True if misclassified, False otherwise.
        """
        with torch.no_grad():
            logits = self.model(x)
            prediction = torch.argmax(logits, dim=1)
            return (prediction != y_true).item()
    