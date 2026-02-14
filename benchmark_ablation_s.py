"""Stability-threshold ablation: sweep S on opportunistic SimBA.

Runs opportunistic SimBA for S in {2, 3, 5, 8, 10} on N images.
Reuses S=5 results from benchmark_winrate.csv when available, and looks
up oracle targets from the same file to avoid redundant untargeted probes.

Usage:
    python benchmark_ablation_s.py                     # Full run (100 images)
    python benchmark_ablation_s.py --n-images 2        # Smoke test
    python benchmark_ablation_s.py --clear             # Clear previous CSV
"""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import torch

from benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from benchmark_winrate import (
    select_images, lookup_oracle_targets, run_attack, make_row,
)
from src.attacks.simba import SimBA

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_NAME = 'resnet50'
SOURCE = 'standard'
EPSILON = 8 / 255
S_VALUES = [2, 3, 5, 8, 10]
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
CSV_PATH = RESULTS_DIR / 'benchmark_ablation_s.csv'
WINRATE_CSV = RESULTS_DIR / 'benchmark_winrate.csv'

CSV_COLUMNS = [
    's_value', 'image', 'true_label', 'iterations', 'success',
    'adversarial_class', 'switch_iteration', 'locked_class', 'timestamp',
]


# ===========================================================================
# CSV I/O
# ===========================================================================
def append_row(row: dict, path: Path):
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_keys(path: Path) -> set:
    """Load existing (s_value, image) keys for resume."""
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row['s_value'], row['image']))
    return keys


def import_s5_from_winrate(winrate_csv: Path, image_names: set,
                           ablation_csv: Path, existing_keys: set) -> int:
    """Import S=5 opportunistic SimBA rows from benchmark_winrate.csv.

    Returns number of rows imported.
    """
    if not winrate_csv.exists():
        return 0
    imported = 0
    with open(winrate_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['method'] != 'SimBA'
                    or row['mode'] != 'opportunistic'
                    or row['image'] not in image_names):
                continue
            key = ('5', row['image'])
            if key in existing_keys:
                continue
            ablation_row = {
                's_value': 5,
                'image': row['image'],
                'true_label': row['true_label'],
                'iterations': row['iterations'],
                'success': row['success'],
                'adversarial_class': row['adversarial_class'],
                'switch_iteration': row.get('switch_iteration', ''),
                'locked_class': row.get('locked_class', ''),
                'timestamp': row.get('timestamp', datetime.now().isoformat()),
            }
            append_row(ablation_row, ablation_csv)
            existing_keys.add(key)
            imported += 1
    return imported


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stability-threshold ablation for opportunistic SimBA"
    )
    parser.add_argument('--clear', action='store_true',
                        help="Delete previous CSV results before running")
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images (default: 100)")
    parser.add_argument('--budget', type=int, default=15_000,
                        help="Query budget per run (default: 15000)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"S values: {S_VALUES}")
    print(f"Images: {args.n_images} (seed={args.image_seed})")
    print(f"Budget: {args.budget}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.clear and CSV_PATH.exists():
        CSV_PATH.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(CSV_PATH)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    # Load model and images
    print(f"\nLoading model: {MODEL_NAME} ({SOURCE})...")
    model = load_benchmark_model(MODEL_NAME, SOURCE, device)

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)

    images = []
    image_name_set = set()
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        name = path.name
        images.append((name, x, y_true))
        image_name_set.add(name)
        print(f"  {name}: true_label={y_true}")

    # Import S=5 from winrate benchmark
    imported = import_s5_from_winrate(
        WINRATE_CSV, image_name_set, CSV_PATH, existing_keys)
    if imported:
        print(f"\nImported {imported} S=5 rows from {WINRATE_CSV}")

    total_runs = len(S_VALUES) * len(images)
    completed = len(existing_keys)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Running ablation: {len(S_VALUES)} S values x {len(images)} images")
    print(f"Total runs: {total_runs} (skipping {completed} existing)")
    print(f"{'='*70}")

    for s_val in S_VALUES:
        for img_idx, (image_name, x, y_true) in enumerate(images):
            key = (str(s_val), image_name)
            if key in existing_keys:
                continue

            y_true_tensor = torch.tensor([y_true], device=device)

            # Run opportunistic SimBA with this S value
            attack = SimBA(
                model=model, epsilon=EPSILON, max_iterations=args.budget,
                device=device, use_dct=True, pixel_range=(0.0, 1.0),
            )
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(0)

            x_adv = attack.generate(
                x, y_true_tensor,
                track_confidence=True,
                targeted=False,
                early_stop=True,
                opportunistic=True,
                stability_threshold=s_val,
            )

            # Extract results
            conf_hist = attack.confidence_history
            if conf_hist and conf_hist.get('iterations'):
                iterations = conf_hist['iterations'][-1]
            else:
                iterations = args.budget

            with torch.no_grad():
                logits = model(x_adv)
                pred = logits.argmax(dim=1).item()
            success = (pred != y_true)

            switch_iter = None
            locked_cls = None
            if conf_hist:
                switch_iter = conf_hist.get('switch_iteration')
                locked_cls = conf_hist.get('locked_class')

            row = {
                's_value': s_val,
                'image': image_name,
                'true_label': y_true,
                'iterations': iterations,
                'success': success,
                'adversarial_class': pred,
                'switch_iteration': switch_iter if switch_iter is not None else '',
                'locked_class': locked_cls if locked_cls is not None else '',
                'timestamp': datetime.now().isoformat(),
            }
            append_row(row, CSV_PATH)
            existing_keys.add(key)
            completed += 1

            status = 'OK' if success else 'FAIL'
            extra = ''
            if switch_iter is not None:
                extra = f" (switch@{switch_iter}, locked={locked_cls})"
            print(f"[{completed}/{total_runs}] S={s_val} | {image_name} | "
                  f"{iterations} iters | {status}{extra}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Ablation complete in {elapsed:.0f}s")
    print(f"Results: {CSV_PATH}")
    print(f"Completed: {completed} runs")


if __name__ == '__main__':
    main()
