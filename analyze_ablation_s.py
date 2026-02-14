"""Analyze stability-threshold ablation results.

Reads benchmark_ablation_s.csv and produces a 2-subplot figure:
  - Success rate vs S
  - Median iterations to success vs S

Usage:
    python analyze_ablation_s.py                       # Default CSV + output
    python analyze_ablation_s.py --show                # Interactive display
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style (matches analyze_winrate.py)
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
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze stability-threshold ablation results"
    )
    parser.add_argument("--csv", default="results/benchmark_ablation_s.csv",
                        help="Path to ablation CSV")
    parser.add_argument("--outdir", default="results/figures_ablation_s",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plots")
    args = parser.parse_args()

    _setup_style()

    print(f"Loading data from {args.csv}")
    df = pd.read_csv(args.csv)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["s_value"] = pd.to_numeric(df["s_value"])

    os.makedirs(args.outdir, exist_ok=True)

    s_values = sorted(df["s_value"].unique())
    print(f"S values found: {s_values}")

    success_rates = []
    median_iters = []
    for s in s_values:
        subset = df[df["s_value"] == s]
        sr = subset["success"].mean()
        success_rates.append(sr)
        succ_subset = subset[subset["success"]]
        med = succ_subset["iterations"].median() if len(succ_subset) > 0 else np.nan
        median_iters.append(med)
        n = len(subset)
        n_succ = len(succ_subset)
        print(f"  S={s:>2d}: {sr:.1%} success ({n_succ}/{n}), "
              f"median iters={med:.0f}" if not np.isnan(med) else
              f"  S={s:>2d}: {sr:.1%} success ({n_succ}/{n}), "
              f"median iters=N/A")

    # Find optimal S (highest success rate, break ties by lowest median iters)
    best_idx = max(range(len(s_values)),
                   key=lambda i: (success_rates[i],
                                  -median_iters[i] if not np.isnan(median_iters[i]) else float('inf')))
    best_s = s_values[best_idx]
    print(f"\nOptimal S: {best_s} (success={success_rates[best_idx]:.1%}, "
          f"median iters={median_iters[best_idx]:.0f})")

    # ---- Figure ----
    color = "#6BA353"  # green (opportunistic)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Success rate
    ax1.plot(s_values, success_rates, 'o-', color=color, linewidth=1.5,
             markersize=6)
    ax1.axvline(best_s, color='gray', linestyle=':', alpha=0.6)
    ax1.set_xlabel("Stability threshold $S$")
    ax1.set_ylabel("Success rate")
    ax1.set_xticks(s_values)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Success Rate vs $S$")

    # Median iterations
    ax2.plot(s_values, median_iters, 's-', color=color, linewidth=1.5,
             markersize=6)
    ax2.axvline(best_s, color='gray', linestyle=':', alpha=0.6)
    ax2.set_xlabel("Stability threshold $S$")
    ax2.set_ylabel("Median iterations (successful)")
    ax2.set_xticks(s_values)
    ax2.set_title("Median Iterations vs $S$")

    _savefig(fig, args.outdir, "fig_ablation_s")
    if args.show:
        plt.show()
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
