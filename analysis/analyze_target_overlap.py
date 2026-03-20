"""Target-class overlap analysis (Issue #25, lightweight variant).

For each image, computes the clean-image top-K non-true class ranking and
checks whether OTS's locked class falls within that ranking.  Also checks
overlap with the trajectory oracle class.

This answers: is OTS's target selection correlated with initial confidence
rankings, or does exploration reveal qualitatively different targets?

Usage:
    python analysis/analyze_target_overlap.py
    python analysis/analyze_target_overlap.py --show
    python analysis/analyze_target_overlap.py --models resnet50 vit_b_16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from benchmarks.benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from benchmarks.winrate import select_images
from pathlib import Path


# ===========================================================================
# Style
# ===========================================================================
def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })
    try:
        from matplotlib.texmanager import TexManager
        TexManager._run_checked_subprocess(["latex", "--version"], "latex")
        plt.rcParams["text.usetex"] = True
    except Exception:
        plt.rcParams["text.usetex"] = False


def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


# ===========================================================================
# Compute clean-image top-K for each image
# ===========================================================================
def compute_topk(model, image_paths: list[Path], device, k: int = 10):
    """Return dict: image_name -> list of top-K non-true class IDs (ordered)."""
    topk_map = {}
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].clone()
            probs[y_true] = -1.0
            _, topk_indices = probs.topk(k)
            topk_map[path.name] = {
                "true_label": y_true,
                "topk": topk_indices.cpu().tolist(),
            }
    return topk_map


# ===========================================================================
# Analysis
# ===========================================================================
def analyze_overlap(df: pd.DataFrame, topk_map: dict, model_name: str,
                    k_values: list[int] = [1, 3, 5, 10]):
    """Compute overlap rates between locked_class and clean-image top-K."""
    opp = df[(df["mode"] == "opportunistic") & df["locked_class"].notna()].copy()
    if "model" in opp.columns:
        opp = opp[opp["model"] == model_name]
    opp["locked_class"] = opp["locked_class"].astype(int)

    # Also get oracle target (untargeted adversarial_class)
    unt = df[df["mode"] == "untargeted"].copy()
    if "model" in unt.columns:
        unt = unt[unt["model"] == model_name]

    results = []
    for method in sorted(opp["method"].unique()):
        opp_m = opp[opp["method"] == method]
        unt_m = unt[unt["method"] == method]

        for k in k_values:
            n_in_topk = 0
            n_oracle_in_topk = 0
            n_total = 0
            n_oracle_total = 0

            for _, row in opp_m.iterrows():
                img = row["image"]
                if img not in topk_map:
                    continue
                topk = topk_map[img]["topk"][:k]
                locked = int(row["locked_class"])
                n_total += 1
                if locked in topk:
                    n_in_topk += 1

            # Oracle overlap
            for _, row in unt_m.iterrows():
                img = row["image"]
                if img not in topk_map:
                    continue
                if not row.get("success", False):
                    continue
                topk = topk_map[img]["topk"][:k]
                oracle_cls = int(row["adversarial_class"])
                n_oracle_total += 1
                if oracle_cls in topk:
                    n_oracle_in_topk += 1

            results.append({
                "method": method,
                "k": k,
                "ots_overlap": n_in_topk / n_total if n_total > 0 else 0,
                "ots_n": n_total,
                "oracle_overlap": n_oracle_in_topk / n_oracle_total if n_oracle_total > 0 else 0,
                "oracle_n": n_oracle_total,
            })

    return pd.DataFrame(results)


# ===========================================================================
# Figure
# ===========================================================================
METHOD_COLORS = {
    "SimBA": "#2176AE",
    "SquareAttack": "#E07A30",
    "Bandits": "#6BA353",
}


def fig_overlap(results: pd.DataFrame, outdir: str, model_name: str):
    """Grouped bar chart: OTS and oracle overlap with clean-image top-K."""
    methods = sorted(results["method"].unique())
    k_values = sorted(results["k"].unique())

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = results[results["method"] == method].sort_values("k")
        x = np.arange(len(k_values))
        width = 0.35

        bars_ots = ax.bar(x - width / 2, sub["ots_overlap"] * 100, width,
                          color=METHOD_COLORS.get(method, "#888"),
                          label="OTS locked class", edgecolor="white")
        bars_oracle = ax.bar(x + width / 2, sub["oracle_overlap"] * 100, width,
                             color="#E8873A", alpha=0.7,
                             label="Oracle class", edgecolor="white")

        for bar, v in zip(bars_ots, sub["ots_overlap"] * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                    f"{v:.0f}%", ha="center", fontsize=8)
        for bar, v in zip(bars_oracle, sub["oracle_overlap"] * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                    f"{v:.0f}%", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-{k}" for k in k_values])
        ax.set_title(method, fontweight="bold")
        ax.set_ylim(0, 115)
        if ax is axes[0]:
            ax.set_ylabel("Overlap Rate (%)")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Target Class Overlap with Clean-Image Ranking -- {model_name}",
                 fontsize=13)
    _savefig(fig, outdir, "fig_target_overlap")
    return fig


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Target-class overlap analysis (Issue #25)")
    parser.add_argument("--csv", default="results/benchmark_standard.csv")
    parser.add_argument("--models", nargs="+", default=["resnet50"])
    parser.add_argument("--source", default="standard")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--image-seed", type=int, default=42)
    parser.add_argument("--outdir", default="results/figures_target_overlap")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = Path("data/imagenet/val")

    print(f"Loading data from {args.csv}")
    df = pd.read_csv(args.csv)
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )

    print(f"Selecting {args.n_images} images from {val_dir}...")
    image_paths = select_images(val_dir, args.n_images, args.image_seed)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        print(f"  Loading model...")
        model = load_benchmark_model(model_name, args.source, device)

        print(f"  Computing clean-image top-10 rankings...")
        topk_map = compute_topk(model, image_paths, device, k=10)

        print(f"  Analyzing overlap...")
        results = analyze_overlap(df, topk_map, model_name)

        # Console output
        print(f"\n  {'Method':<14} {'K':>3} {'OTS overlap':>12} {'Oracle overlap':>15}")
        print(f"  {'-'*48}")
        for _, r in results.iterrows():
            print(f"  {r['method']:<14} {r['k']:>3} "
                  f"{r['ots_overlap']:>11.1%} "
                  f"{r['oracle_overlap']:>14.1%}")

        # Figure
        print(f"\n  Generating figure...")
        fig_overlap(results, args.outdir, model_name)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
