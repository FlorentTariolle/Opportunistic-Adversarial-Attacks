"""CDF analysis: success rate vs query budget figures.

Reads benchmark_winrate.csv and produces:
  - fig_winrate: combined 6-curve plot (2 methods x 3 modes)
  - fig_winrate_by_mode: 3 subplots (one per mode, 2 curves each)

Both methods use true CDF: CDF(alpha) = fraction of attacks with t_a <= alpha,
derived from a single run per (method, image, mode) at a fixed budget.
Legacy SimBA rows with budget > 15K are capped at 15K.

Usage:
    python analyze_winrate.py                       # Default CSV + output
    python analyze_winrate.py --show                # Interactive display
    python analyze_winrate.py --csv results/custom.csv --outdir results/figs
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style configuration (matches analyze_benchmark.py)
# ===========================================================================
def _setup_style():
    """Configure matplotlib for publication-quality output."""
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
        TexManager._run_checked_subprocess(
            ["latex", "--version"], "latex")
        plt.rcParams["text.usetex"] = True
    except Exception:
        plt.rcParams["text.usetex"] = False


def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


# ===========================================================================
# Color / style constants (matches analyze_benchmark.py)
# ===========================================================================
MODE_COLORS = {
    "untargeted": "#4878CF",     # blue
    "targeted": "#E8873A",       # orange
    "opportunistic": "#6BA353",  # green
}
MODE_ORDER = ["untargeted", "targeted", "opportunistic"]
MODE_LABELS = {
    "untargeted": "Untargeted",
    "targeted": "Targeted (oracle)",
    "opportunistic": "Opportunistic",
}
METHOD_LINESTYLES = {
    "SimBA": "-",        # solid
    "SquareAttack": "--",  # dashed
}
METHOD_LABELS = {
    "SimBA": "SimBA",
    "SquareAttack": "Square Attack",
}


# ===========================================================================
# Data loading
# ===========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["budget"] = pd.to_numeric(df["budget"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["switch_iteration"] = pd.to_numeric(
        df["switch_iteration"], errors="coerce"
    )
    df["locked_class"] = pd.to_numeric(df["locked_class"], errors="coerce")
    df["oracle_target"] = pd.to_numeric(df["oracle_target"], errors="coerce")
    df["adversarial_class"] = pd.to_numeric(
        df["adversarial_class"], errors="coerce"
    )
    return df


# ===========================================================================
# CDF computation
# ===========================================================================
BUDGET_CAP = 15_000


def bootstrap_cdf(df: pd.DataFrame, budgets: np.ndarray,
                   n_bootstrap: int = 1000, seed: int = 0):
    """Bootstrap CDF with 90% confidence intervals.

    Args:
        df: DataFrame filtered to a single method (excluding oracle_probe).
        budgets: Array of budget thresholds.
        n_bootstrap: Number of bootstrap samples.
        seed: RNG seed for reproducibility.

    Returns:
        dict mapping mode -> (cdf_mean, ci_lo, ci_hi) arrays.
    """
    rng = np.random.RandomState(seed)
    result = {}
    for mode in MODE_ORDER:
        subset = df[df["mode"] == mode].copy()
        image_names = subset["image"].unique()
        n_images = len(image_names)
        if n_images == 0:
            z = np.zeros(len(budgets))
            result[mode] = (z, z.copy(), z.copy())
            continue

        # Pre-compute per-image success iteration (or NaN if failed/capped)
        img_iter = {}
        for name in image_names:
            row = subset[subset["image"] == name].iloc[0]
            if row["success"] and row["iterations"] <= BUDGET_CAP:
                img_iter[name] = row["iterations"]
            else:
                img_iter[name] = np.nan

        all_cdfs = np.empty((n_bootstrap, len(budgets)))
        for b in range(n_bootstrap):
            sample_names = rng.choice(image_names, size=n_images, replace=True)
            iters = np.array([img_iter[n] for n in sample_names])
            success_iters = np.sort(iters[~np.isnan(iters)])
            counts = np.searchsorted(success_iters, budgets, side="right")
            all_cdfs[b] = counts / n_images

        cdf_mean = all_cdfs.mean(axis=0)
        ci_lo = np.percentile(all_cdfs, 5, axis=0)
        ci_hi = np.percentile(all_cdfs, 95, axis=0)
        result[mode] = (cdf_mean, ci_lo, ci_hi)
    return result


def compute_cdf(df: pd.DataFrame, budgets: np.ndarray) -> dict:
    """Compute CDF of attack success at each budget threshold.

    For each mode, CDF(B) = count(success AND iterations <= B) / n_images.
    One row per (image, mode) is expected (single run per combo).

    Legacy SimBA rows recorded at budget > BUDGET_CAP are handled:
      - If success=True and iterations <= BUDGET_CAP: kept as success
      - If success=True and iterations > BUDGET_CAP: treated as failure
      - If success=False: stays failure (iterations capped to BUDGET_CAP)

    Args:
        df: DataFrame filtered to a single method (excluding oracle_probe).
        budgets: Array of budget thresholds to evaluate.

    Returns:
        dict mapping mode -> array of CDF values (same length as budgets).
    """
    result = {}
    for mode in MODE_ORDER:
        subset = df[df["mode"] == mode].copy()
        n_images = len(subset)
        if n_images == 0:
            result[mode] = np.zeros(len(budgets))
            continue

        # Cap legacy rows: success only counts if iterations <= BUDGET_CAP
        capped_success = subset["success"] & (subset["iterations"] <= BUDGET_CAP)

        # Get iteration counts for attacks that succeed within the cap
        success_iters = np.sort(
            subset.loc[capped_success, "iterations"].values
        )

        # CDF via searchsorted: count of success_iters <= B
        counts = np.searchsorted(success_iters, budgets, side="right")
        result[mode] = counts / n_images

    return result


# ===========================================================================
# Figures
# ===========================================================================
def fig_winrate(simba_cdf, sq_cdf, budgets, outdir, show):
    """Combined plot: methods x modes with 90% CI bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in MODE_ORDER:
        color = MODE_COLORS[mode]
        label_mode = MODE_LABELS[mode]

        if simba_cdf is not None:
            mean, lo, hi = simba_cdf[mode]
            ax.plot(budgets, mean,
                    color=color, linestyle=METHOD_LINESTYLES["SimBA"],
                    linewidth=1.5,
                    label=f"{METHOD_LABELS['SimBA']} — {label_mode}")
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.12)

        if sq_cdf is not None:
            mean, lo, hi = sq_cdf[mode]
            ax.plot(budgets, mean,
                    color=color, linestyle=METHOD_LINESTYLES["SquareAttack"],
                    linewidth=1.5,
                    label=f"{METHOD_LABELS['SquareAttack']} — {label_mode}")
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel("Query budget")
    ax.set_ylabel("Success rate (CDF)")
    ax.set_xlim(0, budgets[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Success Rate vs Query Budget (ResNet-50)")

    _savefig(fig, outdir, "fig_winrate")
    if show:
        plt.show()
    plt.close(fig)


def fig_winrate_by_mode(simba_cdf, sq_cdf, budgets, outdir, show):
    """3 subplots (one per mode), with 90% CI bands."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, mode in zip(axes, MODE_ORDER):
        color = MODE_COLORS[mode]
        label_mode = MODE_LABELS[mode]

        if simba_cdf is not None:
            mean, lo, hi = simba_cdf[mode]
            ax.plot(budgets, mean,
                    color=color, linestyle=METHOD_LINESTYLES["SimBA"],
                    linewidth=1.5,
                    label=METHOD_LABELS["SimBA"])
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.12)

        if sq_cdf is not None:
            mean, lo, hi = sq_cdf[mode]
            ax.plot(budgets, mean,
                    color=color, linestyle=METHOD_LINESTYLES["SquareAttack"],
                    linewidth=1.5,
                    label=METHOD_LABELS["SquareAttack"])
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.12)

        ax.set_xlabel("Query budget")
        ax.set_title(label_mode)
        ax.set_xlim(0, budgets[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", framealpha=0.9)

    axes[0].set_ylabel("Success rate (CDF)")

    _savefig(fig, outdir, "fig_winrate_by_mode")
    if show:
        plt.show()
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze winrate benchmark results"
    )
    parser.add_argument("--csv", default="results/benchmark_winrate.csv",
                        help="Path to benchmark CSV")
    parser.add_argument("--outdir", default="results/figures_winrate",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plots")
    parser.add_argument("--method", choices=["SimBA", "SquareAttack"],
                        default=None,
                        help="Plot only one method (default: both)")
    args = parser.parse_args()

    _setup_style()

    print(f"Loading data from {args.csv}")
    df = load_data(args.csv)

    os.makedirs(args.outdir, exist_ok=True)

    # Split by method (exclude oracle_probe rows)
    run_simba = args.method in (None, "SimBA")
    run_square = args.method in (None, "SquareAttack")

    df_simba = df[(df["method"] == "SimBA") & (df["mode"] != "oracle_probe")] if run_simba else pd.DataFrame()
    df_square = df[(df["method"] == "SquareAttack") & (df["mode"] != "oracle_probe")] if run_square else pd.DataFrame()

    if run_simba:
        print(f"SimBA: {len(df_simba)} rows, {df_simba['image'].nunique()} images")
    if run_square:
        print(f"SquareAttack: {len(df_square)} rows, {df_square['image'].nunique()} images")

    # Unified budget axis: 50-step from 50 to BUDGET_CAP
    step = 50
    budgets = np.arange(step, BUDGET_CAP + 1, step)

    print(f"Budget axis: {budgets[0]}..{budgets[-1]} (step={step}, "
          f"{len(budgets)} points)")

    # Compute bootstrapped CDF (with 90% CI)
    simba_cdf = bootstrap_cdf(df_simba, budgets) if run_simba else None
    sq_cdf = bootstrap_cdf(df_square, budgets) if run_square else None

    # Print summary (use bootstrap mean)
    for mode in MODE_ORDER:
        parts = []
        if simba_cdf:
            parts.append(f"SimBA={simba_cdf[mode][0][-1]:.1%}")
        if sq_cdf:
            parts.append(f"SquareAttack={sq_cdf[mode][0][-1]:.1%}")
        print(f"  {mode:>14s}: {', '.join(parts)} (at {budgets[-1]})")

    # Generate figures
    print(f"\nGenerating figures in {args.outdir}/")
    fig_winrate(simba_cdf, sq_cdf, budgets, args.outdir, args.show)
    fig_winrate_by_mode(simba_cdf, sq_cdf, budgets, args.outdir, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
