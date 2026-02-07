"""Benchmark script for opportunistic adversarial attacks.

Runs SimBA and SquareAttack (CE loss) across multiple models, images, epsilons,
and seeds in three modes: untargeted, targeted-oracle, and opportunistic.

To swap to robust models, change SOURCE and MODELS below (2 lines).
"""

import argparse
import os
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.loader import load_pretrained_model, load_robustbench_model, NormalizedModel
from src.attacks.simba import SimBA
from src.attacks.square import SquareAttack
from src.utils.imaging import IMAGENET_MEAN, IMAGENET_STD

# ===========================================================================
# Configuration — change these 2 lines to swap to robust models
# ===========================================================================
SOURCE = 'standard'                                          # → 'robust'
MODELS = ['resnet18', 'resnet50', 'vgg16', 'alexnet']        # → robust names

# ---------------------------------------------------------------------------
EPSILONS = [8 / 255]
SEEDS = [0, 1, 2]
MAX_ITERATIONS = 10_000
STABILITY_THRESHOLD = {'standard': 5, 'robust': 10}
IMAGE_FILES = ['corgi.jpg', 'porsche.jpg', 'dumbbell.jpg', 'hammer.jpg']
IMAGE_DIR = Path('data')
RESULTS_DIR = Path('results')

CSV_COLUMNS = [
    'model', 'method', 'epsilon', 'seed', 'image', 'mode',
    'iterations', 'success', 'adversarial_class', 'oracle_target',
    'switch_iteration', 'locked_class', 'true_conf_final', 'adv_conf_final', 'timestamp',
]


# ===========================================================================
# Model loading
# ===========================================================================
def load_benchmark_model(name: str, source: str, device: torch.device):
    """Load a model that accepts [0,1] input regardless of source."""
    if source == 'standard':
        raw = load_pretrained_model(name, device=device)
        model = NormalizedModel(raw, IMAGENET_MEAN, IMAGENET_STD).to(device)
    else:
        model = load_robustbench_model(name, device=device)
    model.eval()
    return model


# ===========================================================================
# Image loading
# ===========================================================================
def load_benchmark_image(path: Path, device: torch.device):
    """Load an image as a [0,1] tensor and return (x, true_label).

    x has shape (1, 3, 224, 224) in [0,1].
    true_label is an int.
    """
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # -> [0,1]
    ])
    x = transform(img).unsqueeze(0).to(device)
    return x


def get_true_label(model, x: torch.Tensor) -> int:
    """Get model's prediction on clean image (serves as true label)."""
    with torch.no_grad():
        logits = model(x)
        return logits.argmax(dim=1).item()


# ===========================================================================
# Attack factory
# ===========================================================================
def create_attack(method: str, model, epsilon: float, seed: int, device):
    """Create an attack instance. Both sources use [0,1] models."""
    if method == 'SimBA':
        return SimBA(
            model=model,
            epsilon=epsilon,
            max_iterations=MAX_ITERATIONS,
            device=device,
            use_dct=True,
            pixel_range=(0.0, 1.0),
        )
    elif method == 'SquareAttack':
        return SquareAttack(
            model=model,
            epsilon=epsilon,
            max_iterations=MAX_ITERATIONS,
            device=device,
            loss='ce',
            normalize=False,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ===========================================================================
# Single attack run
# ===========================================================================
def run_single_attack(model, attack, x, y_true_tensor, mode, target_class, seed):
    """Execute a single attack and extract metrics.

    Args:
        model: The model.
        attack: Attack instance (SimBA or SquareAttack).
        x: Input tensor (1, 3, 224, 224) in [0,1].
        y_true_tensor: True label tensor (1,).
        mode: 'untargeted', 'targeted', or 'opportunistic'.
        target_class: int or None — required for targeted mode.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: iterations, success, adversarial_class,
        switch_iteration, true_conf_final, adv_conf_final.
    """
    # Seed for SimBA (SquareAttack seeds via constructor)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    is_targeted = (mode == 'targeted')
    is_opportunistic = (mode == 'opportunistic')
    target_tensor = None
    if is_targeted and target_class is not None:
        target_tensor = torch.tensor([target_class], device=x.device)

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=is_targeted,
        target_class=target_tensor,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=STABILITY_THRESHOLD[SOURCE],
    )

    # Extract iteration count
    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = attack.max_iterations

    # Check success
    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()
        probs = F.softmax(logits, dim=1)
        y_true_int = y_true_tensor.item()
        true_conf_final = probs[0][y_true_int].item()

    if is_targeted:
        success = (pred == target_class)
    else:
        success = (pred != y_true_int)

    adv_conf_final = probs[0][pred].item()

    # Switch iteration and locked class (opportunistic only)
    switch_iter = None
    locked_class = None
    if conf_hist:
        switch_iter = conf_hist.get('switch_iteration')
        locked_class = conf_hist.get('locked_class')

    # Verify L-inf constraint
    linf = (x_adv - x).abs().max().item()
    eps = attack.epsilon
    if linf > eps + 1e-6:
        print(f"  WARNING: L-inf violation! {linf:.6f} > {eps:.6f}")

    return {
        'iterations': iterations,
        'success': success,
        'adversarial_class': pred,
        'switch_iteration': switch_iter,
        'locked_class': locked_class,
        'true_conf_final': round(true_conf_final, 6),
        'adv_conf_final': round(adv_conf_final, 6),
    }


# ===========================================================================
# 3-mode pipeline
# ===========================================================================
def run_targeted_oracle_pipeline(model, method, eps, seed, x, y_true, device,
                                 completed_count, success_count, total_runs,
                                 model_name, image_name, csv_path, existing_keys):
    """Run untargeted → targeted-oracle → opportunistic for one config.

    Returns updated (completed_count, success_count).
    """
    y_true_tensor = torch.tensor([y_true], device=device)

    for mode in ['untargeted', 'targeted', 'opportunistic']:
        # Check if already done (crash recovery)
        key = (model_name, method, f"{eps:.6f}", str(seed), image_name, mode)
        if key in existing_keys:
            completed_count += 1
            # Count as success if it was successful before (we don't know, skip)
            continue

        attack = create_attack(method, model, eps, seed, device)

        # For targeted mode, determine oracle target
        oracle_target = None
        if mode == 'targeted':
            # Run a quick untargeted to find which class emerges
            # Use the same seed to get the same adversarial class as the
            # untargeted run that was already recorded
            probe_attack = create_attack(method, model, eps, seed, device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            probe_adv = probe_attack.generate(
                x, y_true_tensor,
                track_confidence=False,
                targeted=False,
                early_stop=True,
            )
            with torch.no_grad():
                probe_logits = model(probe_adv)
                probe_pred = probe_logits.argmax(dim=1).item()

            if probe_pred != y_true:
                oracle_target = probe_pred
            else:
                # Untargeted failed — pick most-predicted non-true class
                probs = F.softmax(probe_logits, dim=1)
                probs[0][y_true] = -1.0
                oracle_target = probs.argmax(dim=1).item()

        result = run_single_attack(
            model, attack, x, y_true_tensor, mode, oracle_target, seed,
        )

        row = {
            'model': model_name,
            'method': method,
            'epsilon': f"{eps:.6f}",
            'seed': seed,
            'image': image_name,
            'mode': mode,
            'iterations': result['iterations'],
            'success': result['success'],
            'adversarial_class': result['adversarial_class'],
            'oracle_target': oracle_target if oracle_target is not None else '',
            'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
            'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
            'true_conf_final': result['true_conf_final'],
            'adv_conf_final': result['adv_conf_final'],
            'timestamp': datetime.now().isoformat(),
        }

        append_result_to_csv(row, csv_path)

        completed_count += 1
        if result['success']:
            success_count += 1

        status = 'SUCCESS' if result['success'] else 'FAIL'
        iter_str = f"{result['iterations']} iters"
        extra = ''
        if mode == 'opportunistic' and result['switch_iteration'] is not None:
            extra = f" (switch@{result['switch_iteration']}, locked={result['locked_class']})"
        if mode == 'targeted':
            extra = f" (target={oracle_target})"

        print(
            f"[{completed_count}/{total_runs}] {model_name} | {method} | "
            f"eps={eps:.4f} | seed={seed} | {image_name} | {mode} | "
            f"{iter_str} | {status}{extra}"
        )
        print(f"Successes: {success_count}/{completed_count} ({100*success_count/completed_count:.1f}%)")

    return completed_count, success_count


# ===========================================================================
# CSV I/O
# ===========================================================================
def append_result_to_csv(result: dict, path: Path):
    """Append a single result row to the CSV file."""
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def load_existing_results(path: Path) -> set:
    """Load existing result keys for crash recovery.

    Returns a set of (model, method, epsilon, seed, image, mode) tuples.
    """
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['model'], row['method'], row['epsilon'],
                   row['seed'], row['image'], row['mode'])
            keys.add(key)
    return keys


# ===========================================================================
# Summary statistics
# ===========================================================================
def compute_summary_statistics(csv_path: Path):
    """Compute and save summary statistics from results CSV."""
    df = pd.read_csv(csv_path)
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['success'] = df['success'].astype(bool)

    summary = df.groupby(['model', 'method', 'epsilon', 'mode']).agg(
        mean_iterations=('iterations', 'mean'),
        median_iterations=('iterations', 'median'),
        std_iterations=('iterations', 'std'),
        success_rate=('success', 'mean'),
        n_runs=('success', 'count'),
    ).reset_index()

    # Add switch rate for opportunistic mode
    opp = df[df['mode'] == 'opportunistic'].copy()
    if not opp.empty:
        opp['switched'] = opp['switch_iteration'].notna() & (opp['switch_iteration'] != '')
        switch_rate = opp.groupby(['model', 'method', 'epsilon']).agg(
            switch_rate=('switched', 'mean'),
        ).reset_index()
        switch_rate['mode'] = 'opportunistic'
        summary = summary.merge(switch_rate, on=['model', 'method', 'epsilon', 'mode'], how='left')
    else:
        summary['switch_rate'] = float('nan')

    summary_path = csv_path.with_name(csv_path.stem + '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary.to_string(index=False))
    return summary


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run adversarial attack benchmark")
    parser.add_argument('--clear', action='store_true', help="Delete previous CSV results before running")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Source: {SOURCE}")
    print(f"Models: {MODELS}")
    print(f"Images: {IMAGE_FILES}")
    print(f"Epsilons: {[f'{e:.4f}' for e in EPSILONS]}")
    print(f"Seeds: {SEEDS}")
    print(f"Stability threshold: {STABILITY_THRESHOLD[SOURCE]} ({SOURCE})")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f'benchmark_{SOURCE}.csv'

    if args.clear and csv_path.exists():
        csv_path.unlink()
        summary_path = csv_path.with_name(csv_path.stem + '_summary.csv')
        if summary_path.exists():
            summary_path.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_results(csv_path)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    methods = ['SimBA', 'SquareAttack']
    total_runs = len(MODELS) * len(IMAGE_FILES) * len(methods) * len(EPSILONS) * len(SEEDS) * 3
    print(f"Total runs: {total_runs}")
    print("=" * 80)

    completed_count = len(existing_keys)
    # Count successes from existing results
    success_count = 0
    if existing_keys and csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'].lower() == 'true':
                    success_count += 1

    start_time = time.time()

    for model_name in MODELS:
        print(f"\nLoading model: {model_name} ({SOURCE})...")
        model = load_benchmark_model(model_name, SOURCE, device)

        for image_name in IMAGE_FILES:
            image_path = IMAGE_DIR / image_name
            if not image_path.exists():
                print(f"  WARNING: {image_path} not found, skipping")
                # Still count the skipped runs
                skipped = len(methods) * len(EPSILONS) * len(SEEDS) * 3
                completed_count += skipped
                continue

            x = load_benchmark_image(image_path, device)
            y_true = get_true_label(model, x)
            print(f"\n  Image: {image_name} (true class: {y_true})")

            for method in methods:
                for eps in EPSILONS:
                    for seed in SEEDS:
                        completed_count, success_count = run_targeted_oracle_pipeline(
                            model, method, eps, seed, x, y_true, device,
                            completed_count, success_count, total_runs,
                            model_name, image_name, csv_path, existing_keys,
                        )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Benchmark complete in {elapsed:.0f}s")
    print(f"Results: {csv_path}")

    # Generate summary
    if csv_path.exists():
        compute_summary_statistics(csv_path)


if __name__ == '__main__':
    main()
