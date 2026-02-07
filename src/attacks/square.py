"""Square Attack wrapper with opportunistic targeting support.

Subclasses torchattacks.Square to inject per-iteration confidence tracking
and rank-stability monitoring, while reusing all of torchattacks' core
infrastructure (p_selection, initialization, perturbation strategy).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks

from .base import BaseAttack
from src.utils.imaging import IMAGENET_MEAN, IMAGENET_STD


class _OpportunisticSquare(torchattacks.Square):
    """Internal torchattacks.Square subclass with per-iteration hooks.

    Overrides ``attack_single_run`` (Linf branch only) to record softmax
    confidences and to implement opportunistic target-locking.  All other
    behaviour (p_selection, initialization, L2 branch) is inherited.
    """

    def __init__(self, model, eps, n_queries, device,
                 track_confidence=False, y_true=None,
                 opportunistic=False, stability_threshold=30,
                 targeted_mode=False, target_class=None,
                 early_stop=True, loss='margin', normalize=True):
        super().__init__(
            model,
            norm='Linf',
            eps=eps,
            n_queries=n_queries,
            n_restarts=1,
            p_init=0.8,
            loss=loss,
            resc_schedule=True,
            seed=0,
            verbose=False,
        )
        if normalize:
            self.set_normalization_used(IMAGENET_MEAN, IMAGENET_STD)

        # Hook configuration
        self._track_confidence = track_confidence
        self._y_true = y_true          # true label (scalar tensor)
        self._opportunistic = opportunistic
        self._stability_threshold = stability_threshold
        self._targeted_mode = targeted_mode
        self._target_class = target_class
        self._early_stop = early_stop

        # Will be populated during attack_single_run
        self.confidence_history = None

    # ------------------------------------------------------------------
    # Linf branch of attack_single_run, with hooks injected
    # ------------------------------------------------------------------
    def attack_single_run(self, x, y):
        """Override to inject confidence tracking & opportunistic switching.

        Only the Linf branch is implemented here; L2 falls through to the
        parent class.
        """
        if self.norm != 'Linf':
            return super().attack_single_run(x, y)

        # Save targeted state — _check_stability may flip self.targeted
        # during the run, but perturb() must see the original value.
        _orig_targeted = self.targeted

        with torch.no_grad():
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]

            # ---- confidence tracking state ----
            confidence_history = {
                'iterations': [],
                'original_class': [],
                'max_other_class': [],
                'target_class': [],
                'switch_iteration': None,
                'top_classes': [],
                'locked_class': None,
            }
            y_true = self._y_true
            opportunistic = self._opportunistic
            stability_threshold = self._stability_threshold
            switched_to_targeted = False
            stability_counter = 0
            prev_max_class = None
            is_targeted = self._targeted_mode
            locked_target = self._target_class  # may be None

            # Helper: record confidence at current x_best
            def _record_confidence(x_img, iteration):
                """Record softmax confidences for a single image."""
                logits = self.get_logits(x_img)
                probs = F.softmax(logits, dim=1)
                original_conf = probs[0][y_true].item()
                probs_excl = probs[0].clone()
                probs_excl[y_true] = -1.0
                max_other_conf = probs_excl.max().item()
                confidence_history['iterations'].append(iteration)
                confidence_history['original_class'].append(original_conf)
                confidence_history['max_other_class'].append(max_other_conf)
                if is_targeted and locked_target is not None:
                    confidence_history['target_class'].append(
                        probs[0][locked_target].item()
                    )
                if opportunistic and not switched_to_targeted:
                    top10_idx = torch.topk(probs_excl, k=10).indices.tolist()
                    top10_conf = {idx: probs[0][idx].item() for idx in top10_idx}
                    confidence_history['top_classes'].append(top10_conf)

            # Helper: opportunistic stability check after an accepted perturbation
            def _check_stability(x_img, iteration):
                nonlocal switched_to_targeted, stability_counter, prev_max_class
                nonlocal is_targeted, locked_target
                if not opportunistic or switched_to_targeted:
                    return
                logits = self.get_logits(x_img)
                probs = F.softmax(logits, dim=1)
                probs_excl = probs[0].clone()
                probs_excl[y_true] = -1.0
                current_max_class = torch.argmax(probs_excl).item()
                if prev_max_class is not None and current_max_class == prev_max_class:
                    stability_counter += 1
                    if stability_counter >= stability_threshold:
                        is_targeted = True
                        locked_target = current_max_class
                        switched_to_targeted = True
                        confidence_history['switch_iteration'] = iteration
                        confidence_history['locked_class'] = current_max_class
                        # Flip torchattacks' targeted flag so margin_and_loss flips
                        self.targeted = True
                else:
                    stability_counter = 0
                prev_max_class = current_max_class

            # ---- standard Square Linf initialisation ----
            x_best = torch.clamp(
                x + self.eps * self.random_choice([x.shape[0], c, 1, w]),
                0.0, 1.0,
            )
            margin_min, loss_min = self.margin_and_loss(x_best, y)
            n_queries = torch.ones(x.shape[0]).to(self.device)

            # Record initial confidence (iteration 0)
            if self._track_confidence and n_ex_total == 1:
                _record_confidence(x_best, 0)

            for i_iter in range(self.n_queries):
                idx_to_fool = (margin_min > 0.0).nonzero().flatten()

                if len(idx_to_fool) == 0 and self._early_stop:
                    break

                if len(idx_to_fool) == 0:
                    # Attack succeeded but early_stop=False; keep iterating
                    # for confidence tracking (just record, no perturbation)
                    if self._track_confidence and n_ex_total == 1:
                        should_track = (i_iter < 50) or (i_iter % 10 == 0)
                        if should_track:
                            _record_confidence(x_best, i_iter + 1)
                    continue

                x_curr = self.check_shape(x[idx_to_fool])
                x_best_curr = self.check_shape(x_best[idx_to_fool])
                y_curr = y[idx_to_fool]
                if len(y_curr.shape) == 0:
                    y_curr = y_curr.unsqueeze(0)
                margin_min_curr = margin_min[idx_to_fool]
                loss_min_curr = loss_min[idx_to_fool]

                # If we switched to targeted via opportunistic, update y_curr
                if switched_to_targeted and locked_target is not None:
                    y_curr = torch.tensor(
                        [locked_target], device=self.device, dtype=y.dtype
                    )

                p = self.p_selection(i_iter)
                s = max(int(round(math.sqrt(p * n_features / c))), 1)
                vh = self.random_int(0, h - s)
                vw = self.random_int(0, w - s)
                new_deltas = torch.zeros([c, h, w]).to(self.device)
                new_deltas[:, vh:vh + s, vw:vw + s] = (
                    2.0 * self.eps * self.random_choice([c, 1, 1])
                )

                x_new = x_best_curr + new_deltas
                x_new = torch.min(
                    torch.max(x_new, x_curr - self.eps),
                    x_curr + self.eps,
                )
                x_new = torch.clamp(x_new, 0.0, 1.0)
                x_new = self.check_shape(x_new)

                margin, loss = self.margin_and_loss(x_new, y_curr)

                # Update loss if new loss is better
                idx_improved = (loss < loss_min_curr).float()
                loss_min[idx_to_fool] = (
                    idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                )

                # Update margin and x_best if new loss is better or misclassified
                idx_miscl = (margin <= 0.0).float()
                idx_improved = torch.max(idx_improved, idx_miscl)
                margin_min[idx_to_fool] = (
                    idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                )
                idx_improved_r = idx_improved.reshape(
                    [-1, *[1] * len(x.shape[:-1])]
                )
                x_best[idx_to_fool] = (
                    idx_improved_r * x_new + (1.0 - idx_improved_r) * x_best_curr
                )
                n_queries[idx_to_fool] += 1.0

                # ---- hooks (only for single-image tracking) ----
                accepted = idx_improved.sum().item() > 0
                was_switched = switched_to_targeted
                if self._track_confidence and n_ex_total == 1:
                    should_track = (i_iter < 50) or (i_iter % 10 == 0)
                    if should_track or accepted:
                        _record_confidence(x_best, i_iter + 1)

                # Opportunistic stability check (independent of confidence tracking)
                if n_ex_total == 1 and accepted:
                    _check_stability(x_best, i_iter + 1)

                # After opportunistic switch, recompute margins with
                # targeted semantics so the loop keeps optimising.
                if switched_to_targeted and not was_switched:
                    y_target = torch.tensor(
                        [locked_target], device=self.device, dtype=y.dtype
                    )
                    margin_min, loss_min = self.margin_and_loss(
                        x_best, y_target
                    )

                # Early exit if all samples fooled
                ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                if ind_succ.numel() == n_ex_total and self._early_stop:
                    break

            self.confidence_history = confidence_history
            self.targeted = _orig_targeted
            return n_queries, x_best


class SquareAttack(BaseAttack):
    """Square Attack wrapper conforming to the BaseAttack interface.

    Uses ``torchattacks.Square`` internally (Linf, margin loss, 1 restart)
    with an overridden ``attack_single_run`` that adds confidence tracking
    and opportunistic rank-stability monitoring.

    Epsilon is in [0, 1] pixel space (same scale as torchattacks).
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8 / 255,
        max_iterations: int = 1000,
        device: Optional[torch.device] = None,
        loss: str = 'margin',
        normalize: bool = True,
    ):
        super().__init__(model, epsilon, max_iterations, device)
        self.loss = loss
        self.normalize = normalize

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
        **kwargs,
    ) -> torch.Tensor:
        """Generate adversarial examples using Square Attack.

        The interface mirrors ``SimBA.generate`` exactly so that the demo
        can treat all attacks interchangeably.

        Args:
            x: Input images tensor (batch, C, H, W) — ImageNet-normalised.
            y: True labels tensor (batch,).
            track_confidence: Record per-iteration softmax confidences.
            targeted: Perform a targeted attack toward *target_class*.
            target_class: Target class tensor (batch,).  Required when
                ``targeted=True``.
            early_stop: Stop as soon as attack succeeds.
            opportunistic: Start untargeted, lock onto a stable class.
            stability_threshold: Consecutive improved iterations with
                the same top non-true class before locking.

        Returns:
            Adversarial examples tensor (same shape as *x*).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")
        if y.dim() != 1:
            raise ValueError(f"Expected 1D labels, got {y.dim()}D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Batch size mismatch between x and y")
        if targeted and target_class is None:
            raise ValueError("target_class required when targeted=True")
        if opportunistic and targeted:
            raise ValueError(
                "opportunistic=True cannot be combined with targeted=True"
            )

        batch_size = x.shape[0]
        x = x.to(self.device)
        y = y.to(self.device)
        if target_class is not None:
            target_class = target_class.to(self.device)

        self.confidence_history = None

        if track_confidence and batch_size != 1:
            import warnings
            warnings.warn(
                "track_confidence only supported for batch_size=1; disabling."
            )
            track_confidence = False

        x_adv = x.clone()

        for i in range(batch_size):
            xi = x[i].unsqueeze(0)
            yi = y[i].unsqueeze(0)
            t_cls = target_class[i].item() if targeted else None

            sq = _OpportunisticSquare(
                model=self.model,
                eps=self.epsilon,
                n_queries=self.max_iterations,
                device=self.device,
                track_confidence=track_confidence,
                y_true=y[i],
                opportunistic=opportunistic,
                stability_threshold=stability_threshold,
                targeted_mode=targeted,
                target_class=t_cls,
                early_stop=early_stop,
                loss=self.loss,
                normalize=self.normalize,
            )

            if targeted:
                sq.set_mode_targeted_by_label(quiet=True)

            # torchattacks' __call__ handles denorm -> attack -> renorm
            x_adv_i = sq(xi, yi if not targeted else target_class[i].unsqueeze(0))
            x_adv[i] = x_adv_i.squeeze(0)

            if track_confidence and i == 0:
                self.confidence_history = sq.confidence_history

        return x_adv
