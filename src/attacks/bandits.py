"""Bandits attack implementation (Ilyas et al. 2019, Prior Convictions)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseAttack


class BanditsAttack(BaseAttack):
    """Bandits attack using gradient estimation with data-dependent priors.

    A black-box attack that estimates gradients via finite differences with
    a bandit prior, then applies sign-based updates (like PGD).
    Uses 2 queries per iteration.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        max_iterations: int = 5000,
        device: Optional[torch.device] = None,
        fd_eta: float = 0.01,
        image_lr: Optional[float] = None,
        exploration: float = 0.01,
        prior_lr: float = 0.1,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        seed: int = 0,
    ):
        super().__init__(model, epsilon, max_iterations, device)
        self.fd_eta = fd_eta
        self.image_lr = image_lr if image_lr is not None else epsilon
        self.exploration = exploration
        self.prior_lr = prior_lr
        self.pixel_range = pixel_range
        self.seed = seed

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        track_confidence: bool = False,
        targeted: bool = False,
        target_class: Optional[torch.Tensor] = None,
        early_stop: bool = True,
        opportunistic: bool = False,
        stability_threshold: int = 30,
        reference_direction: Optional[torch.Tensor] = None,
        naive_switch_iteration: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        if y.dim() != 1:
            raise ValueError(f"Expected 1D tensor for labels, got {y.dim()}D")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x={x.shape[0]}, y={y.shape[0]}")
        if targeted and target_class is None:
            raise ValueError("target_class is required when targeted=True")
        if opportunistic and targeted:
            raise ValueError("opportunistic=True cannot be combined with targeted=True")

        batch_size = x.shape[0]
        x_adv = x.clone().to(self.device)
        y = y.to(self.device)
        if target_class is not None:
            target_class = target_class.to(self.device)

        self.confidence_history = None

        if track_confidence and batch_size == 1:
            t_class = target_class[0] if targeted else None
            result = self._attack_single_image(
                x[0], y[0], track_confidence=True, targeted=targeted, target_class=t_class,
                early_stop=early_stop, opportunistic=opportunistic, stability_threshold=stability_threshold,
                reference_direction=reference_direction, naive_switch_iteration=naive_switch_iteration,
            )
            if isinstance(result, tuple) and len(result) == 2:
                x_adv[0], self.confidence_history = result
            else:
                x_adv[0] = result if not isinstance(result, tuple) else result[0]
                self.confidence_history = None
        else:
            if track_confidence and batch_size != 1:
                import warnings
                warnings.warn(f"track_confidence=True only supported for batch_size=1. Got {batch_size}.")
            for i in range(batch_size):
                t_class = target_class[i] if targeted else None
                result = self._attack_single_image(
                    x[i], y[i], track_confidence=False, targeted=targeted, target_class=t_class,
                    early_stop=early_stop, opportunistic=opportunistic, stability_threshold=stability_threshold,
                    reference_direction=reference_direction, naive_switch_iteration=naive_switch_iteration,
                )
                if isinstance(result, tuple):
                    x_adv[i] = result[0]
                else:
                    x_adv[i] = result

        return x_adv

    def _attack_single_image(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        track_confidence: bool = False,
        targeted: bool = False,
        target_class: Optional[torch.Tensor] = None,
        early_stop: bool = True,
        opportunistic: bool = False,
        stability_threshold: int = 30,
        reference_direction: Optional[torch.Tensor] = None,
        naive_switch_iteration: Optional[int] = None,
    ) -> tuple:
        x_adv = x.clone()
        prior = torch.zeros_like(x)
        confidence_history = {
            'iterations': [],
            'original_class': [],
            'max_other_class': [],
            'max_other_class_id': [],
            'target_class': [],
            'cos_sim_to_ref': [],
            'cos_sim_iterations': [],
            'switch_iteration': None,
            'top_classes': [],
            'locked_class': None,
        }

        ref_flat = None
        if reference_direction is not None:
            ref_flat = reference_direction.flatten().to(self.device)

        # Opportunistic targeting state
        if opportunistic:
            stability_counter = 0
            prev_max_class = None
            switched_to_targeted = False
            switch_iteration = None

        # Seed RNG
        rng = torch.Generator(device=self.device if self.device.type == 'cuda' else 'cpu')
        rng.manual_seed(self.seed)

        # Record initial confidence
        if track_confidence:
            self._record_confidence(
                x_adv, y_true, target_class, targeted, confidence_history,
                iteration=0, ref_flat=ref_flat, x_orig=x,
                opportunistic=opportunistic,
                switched_to_targeted=opportunistic and False,
            )

        # Check if already misclassified
        if early_stop:
            if targeted:
                with torch.no_grad():
                    pred = self.model(x_adv.unsqueeze(0)).argmax(dim=1).item()
                if pred == target_class.item():
                    return (x_adv, confidence_history) if track_confidence else x_adv
            else:
                with torch.no_grad():
                    pred = self.model(x_adv.unsqueeze(0)).argmax(dim=1).item()
                if pred != y_true.item():
                    return (x_adv, confidence_history) if track_confidence else x_adv

        for iteration in range(self.max_iterations):
            # Generate random noise
            noise = torch.randn(x.shape, device=self.device, generator=rng if self.device.type != 'cuda' else None)
            noise = noise / (noise.norm() + 1e-12)

            # Combine exploration noise with prior
            v = self.exploration * noise + (1 - self.exploration) * prior
            v = v / (v.norm() + 1e-12)

            # Finite-difference gradient estimation (2 queries)
            with torch.no_grad():
                x_plus = torch.clamp(x_adv + self.fd_eta * v, self.pixel_range[0], self.pixel_range[1])
                x_minus = torch.clamp(x_adv - self.fd_eta * v, self.pixel_range[0], self.pixel_range[1])

                if targeted:
                    loss_label = target_class
                else:
                    loss_label = y_true

                logits_plus = self.model(x_plus.unsqueeze(0))
                loss_plus = F.cross_entropy(logits_plus, loss_label.unsqueeze(0))

                logits_minus = self.model(x_minus.unsqueeze(0))
                loss_minus = F.cross_entropy(logits_minus, loss_label.unsqueeze(0))

            grad_est = (loss_plus - loss_minus) / (2 * self.fd_eta) * v

            # Update prior (exponential moving average)
            prior = self.prior_lr * grad_est + (1 - self.prior_lr) * prior

            # Update adversarial image
            # Untargeted: ascend loss (maximize CE w.r.t. true class)
            # Targeted: descend loss (minimize CE w.r.t. target class)
            if targeted:
                x_adv = x_adv - self.image_lr * grad_est.sign()
            else:
                x_adv = x_adv + self.image_lr * grad_est.sign()

            # Project to L∞ ball and pixel range
            perturbation = x_adv - x
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + perturbation, self.pixel_range[0], self.pixel_range[1])

            # Track confidence
            if track_confidence:
                should_track = (iteration + 1 < 50) or ((iteration + 1) % 10 == 0)
                if should_track:
                    self._record_confidence(
                        x_adv, y_true, target_class, targeted, confidence_history,
                        iteration=iteration + 1, ref_flat=ref_flat, x_orig=x,
                        opportunistic=opportunistic,
                        switched_to_targeted=opportunistic and switched_to_targeted if opportunistic else False,
                    )

            # Check current prediction for early stop and OTS
            with torch.no_grad():
                logits = self.model(x_adv.unsqueeze(0))
                current_pred = logits.argmax(dim=1).item()

            # Opportunistic stability check (every iteration, since Bandits always updates)
            if opportunistic and not switched_to_targeted:
                probs = F.softmax(logits, dim=1)
                probs_excluding_true = probs[0].clone()
                probs_excluding_true[y_true] = -1.0
                current_max_class = torch.argmax(probs_excluding_true).item()

                if naive_switch_iteration is not None and iteration + 1 >= naive_switch_iteration:
                    targeted = True
                    target_class = torch.tensor(current_max_class, device=self.device)
                    switched_to_targeted = True
                    switch_iteration = iteration + 1
                    confidence_history['switch_iteration'] = switch_iteration
                    confidence_history['locked_class'] = current_max_class
                elif naive_switch_iteration is None and prev_max_class is not None and current_max_class == prev_max_class:
                    stability_counter += 1
                    if stability_counter >= stability_threshold:
                        targeted = True
                        target_class = torch.tensor(current_max_class, device=self.device)
                        switched_to_targeted = True
                        switch_iteration = iteration + 1
                        confidence_history['switch_iteration'] = switch_iteration
                        confidence_history['locked_class'] = current_max_class
                else:
                    stability_counter = 0
                prev_max_class = current_max_class

            # Early stop check
            if early_stop:
                if targeted and current_pred == target_class.item():
                    if track_confidence:
                        return x_adv, confidence_history
                    return x_adv
                elif not targeted and current_pred != y_true.item():
                    if track_confidence:
                        return x_adv, confidence_history
                    return x_adv

        # Exhausted loop: record final state
        if track_confidence and (not confidence_history['iterations'] or confidence_history['iterations'][-1] != self.max_iterations):
            self._record_confidence(
                x_adv, y_true, target_class, targeted, confidence_history,
                iteration=self.max_iterations, ref_flat=ref_flat, x_orig=x,
                opportunistic=opportunistic,
                switched_to_targeted=True,  # doesn't matter at end
            )

        if track_confidence:
            return x_adv, confidence_history
        return x_adv

    def _record_confidence(
        self, x_adv, y_true, target_class, targeted, confidence_history,
        iteration, ref_flat, x_orig, opportunistic, switched_to_targeted,
    ):
        """Record confidence metrics at the current iteration."""
        with torch.no_grad():
            logits = self.model(x_adv.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            original_conf = probs[0][y_true].item()
            probs_excluding_original = probs[0].clone()
            probs_excluding_original[y_true] = -1.0
            max_other_conf = probs_excluding_original.max().item()
            max_other_class_id = probs_excluding_original.argmax().item()

        confidence_history['iterations'].append(iteration)
        confidence_history['original_class'].append(original_conf)
        confidence_history['max_other_class'].append(max_other_conf)
        confidence_history['max_other_class_id'].append(max_other_class_id)

        if targeted and target_class is not None:
            confidence_history['target_class'].append(probs[0][target_class].item())

        if ref_flat is not None:
            delta = (x_adv - x_orig).flatten()
            cos = F.cosine_similarity(delta.unsqueeze(0), ref_flat.unsqueeze(0)).item()
            confidence_history['cos_sim_to_ref'].append(cos)
            confidence_history['cos_sim_iterations'].append(iteration)

        if opportunistic and not switched_to_targeted:
            top10_indices = torch.topk(probs_excluding_original, k=10).indices.tolist()
            top10_conf = {idx: probs[0][idx].item() for idx in top10_indices}
            confidence_history['top_classes'].append(top10_conf)
