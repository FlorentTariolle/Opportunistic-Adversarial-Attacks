"""
Mini-benchmark: Targeted direct vs. Hybrid (100 untargeted + targeted)

Compare two approaches to reach target class 285:
1. Direct: Targeted attack from the original image
2. Hybrid: 100 untargeted iterations then targeted from the perturbed image

Parameters:
- Epsilon: 0.5, 1.25, 2.0
- Seeds: 3 per configuration
- Max budget: 2000 iterations
- Success criterion: class 285 becomes top-1
"""

import torch
import numpy as np
import random
from src.models.loader import get_model
from src.utils.imaging import load_image, preprocess_image, get_imagenet_label
from src.attacks import SimBA


def set_seed(seed: int):
    """Fix all seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_targeted_attack(model, image_tensor, true_class, target_class, epsilon, max_iter, device):
    """
    Run a targeted attack and return the number of iterations.
    Returns (iterations, success).
    """
    attack = SimBA(
        model=model,
        epsilon=epsilon,
        max_iterations=max_iter,
        device=device,
        use_dct=True
    )

    y_true = torch.tensor([true_class], device=device)
    target_tensor = torch.tensor([target_class], device=device)

    x_adv = attack.generate(
        image_tensor.clone(),
        y_true,
        track_confidence=True,
        targeted=True,
        target_class=target_tensor
    )

    # Check success
    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_class = torch.argmax(adv_logits, dim=1).item()

    success = (adv_class == target_class)

    # Get iteration count (on failure = max budget)
    if attack.confidence_history and attack.confidence_history['iterations']:
        iterations = attack.confidence_history['iterations'][-1]
    else:
        iterations = 0
    if not success:
        iterations = max_iter

    return iterations, success


def run_untargeted_attack(model, image_tensor, true_class, epsilon, max_iter, device, fixed_iterations=False):
    """
    Run an untargeted attack and return the perturbed image.
    If fixed_iterations=True, run exactly max_iter iterations (no early stopping).
    Returns (x_adv, iterations).
    """
    attack = SimBA(
        model=model,
        epsilon=epsilon,
        max_iterations=max_iter,
        device=device,
        use_dct=True
    )

    y_true = torch.tensor([true_class], device=device)

    x_adv = attack.generate(
        image_tensor.clone(),
        y_true,
        track_confidence=True,
        targeted=False,
        early_stop=not fixed_iterations
    )

    # Get iteration count
    if attack.confidence_history and attack.confidence_history['iterations']:
        iterations = attack.confidence_history['iterations'][-1]
    else:
        iterations = max_iter

    return x_adv, iterations


def run_hybrid_attack(model, image_tensor, true_class, target_class, epsilon, untargeted_iter, max_targeted_iter, device):
    """
    Run untargeted_iter untargeted iterations then targeted from the perturbed image.
    Returns (total_iterations, success).
    """
    # Phase 1: Untargeted (exactly untargeted_iter iterations, no early stop)
    x_perturbed, untargeted_done = run_untargeted_attack(
        model, image_tensor, true_class, epsilon, untargeted_iter, device,
        fixed_iterations=True
    )

    # Phase 2: Targeted from the perturbed image
    # Get current class of perturbed image for y_true
    with torch.no_grad():
        perturbed_logits = model(x_perturbed)
        current_class = torch.argmax(perturbed_logits, dim=1).item()

    # Check if target was already reached by chance
    if current_class == target_class:
        return untargeted_iter, True

    targeted_iter, success = run_targeted_attack(
        model, x_perturbed, current_class, target_class, epsilon, max_targeted_iter, device
    )

    # Total = untargeted iterations + targeted iterations
    total_iterations = untargeted_iter + targeted_iter

    return total_iterations, success


def main():
    # Config
    TARGET_CLASS = 285
    EPSILONS = [0.5, 1.25, 2.0]
    SEEDS = [42, 123, 456]
    MAX_ITERATIONS = 2000
    UNTARGETED_ITERATIONS = 100
    IMAGE_PATH = "data/cat.jpg"

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target class: {TARGET_CLASS} ({get_imagenet_label(TARGET_CLASS)})")
    print(f"Epsilons: {EPSILONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Untargeted warmup: {UNTARGETED_ITERATIONS} iterations")
    print("=" * 70)

    # Load model
    model = get_model('resnet18', device=device)

    # Load image
    image = load_image(IMAGE_PATH)
    image_tensor = preprocess_image(image, normalize=True, device=device)

    # Get original class
    with torch.no_grad():
        logits = model(image_tensor)
        original_class = torch.argmax(logits, dim=1).item()

    print(f"Original class: {original_class} ({get_imagenet_label(original_class)})")
    print("=" * 70)

    # Results
    results = {eps: {'direct': [], 'hybrid': []} for eps in EPSILONS}

    for epsilon in EPSILONS:
        print(f"\n{'='*70}")
        print(f"EPSILON = {epsilon}")
        print(f"{'='*70}")

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            # Direct approach
            set_seed(seed)
            direct_iter, direct_success = run_targeted_attack(
                model, image_tensor, original_class, TARGET_CLASS,
                epsilon, MAX_ITERATIONS, device
            )
            results[epsilon]['direct'].append((direct_iter, direct_success))
            status_direct = "OK" if direct_success else "FAIL"
            print(f"Direct targeted:  {direct_iter:4d} iterations [{status_direct}]")

            # Hybrid approach
            set_seed(seed)
            hybrid_iter, hybrid_success = run_hybrid_attack(
                model, image_tensor, original_class, TARGET_CLASS,
                epsilon, UNTARGETED_ITERATIONS, MAX_ITERATIONS - UNTARGETED_ITERATIONS, device
            )
            results[epsilon]['hybrid'].append((hybrid_iter, hybrid_success))
            status_hybrid = "OK" if hybrid_success else "FAIL"
            print(f"Hybrid (100+targeted): {hybrid_iter:4d} iterations [{status_hybrid}]")

            # Comparison
            diff = hybrid_iter - direct_iter
            if diff < 0:
                print(f"  -> Hybrid wins by {-diff} iterations")
            elif diff > 0:
                print(f"  -> Direct wins by {diff} iterations")
            else:
                print(f"  -> Tie")

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Epsilon':<10} {'Method':<20} {'Mean':<12} {'Success':<10}")
    print("-" * 70)

    for epsilon in EPSILONS:
        direct_iters = [r[0] for r in results[epsilon]['direct']]
        direct_success = sum(1 for r in results[epsilon]['direct'] if r[1])
        hybrid_iters = [r[0] for r in results[epsilon]['hybrid']]
        hybrid_success = sum(1 for r in results[epsilon]['hybrid'] if r[1])

        avg_direct = np.mean(direct_iters)
        avg_hybrid = np.mean(hybrid_iters)

        print(f"{epsilon:<10} {'Direct':<20} {avg_direct:<12.1f} {direct_success}/{len(SEEDS)}")
        print(f"{'':<10} {'Hybrid (100+target)':<20} {avg_hybrid:<12.1f} {hybrid_success}/{len(SEEDS)}")

        diff = avg_hybrid - avg_direct
        if diff < 0:
            print(f"{'':<10} -> Hybrid wins by {abs(diff):.1f} iterations on average")
        else:
            print(f"{'':<10} -> Direct wins by {diff:.1f} iterations on average")
        print()


if __name__ == "__main__":
    main()
