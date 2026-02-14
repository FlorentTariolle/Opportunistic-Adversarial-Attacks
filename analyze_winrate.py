"""Winrate analysis: success rate vs query budget figures.

Reads benchmark_winrate.csv and produces:
  - fig_winrate: combined 6-curve plot (2 methods x 3 modes)
  - fig_winrate_by_mode: 3 subplots (one per mode, 2 curves each)

SimBA winrate is derived from a single run via CDF (iterations <= budget).
SquareAttack winrate is one point per budget (separate runs).

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
        plt.rcParams["text.usetex"] = True
        fig_test, ax_test = plt.subplots(figsize=(1, 1))
        ax_test.set_title("$x$")
        fig_test.canvas.draw()
        plt.close(fig_test)
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
# Winrate computation
# ===========================================================================
def compute_simba_winrate(df: pd.DataFrame, budgets: np.ndarray) -> dict:
    """Compute SimBA winrate at each budget via CDF of iteration counts.

    For each mode, winrate(B) = count(success AND iterations <= B) / n_images.

    Args:
        df: DataFrame filtered to method='SimBA'.
        budgets: Array of budget values to evaluate.

    Returns:
        dict mapping mode -> array of winrates (same length as budgets).
    """
    result = {}
    for mode in MODE_ORDER:
        subset = df[df["mode"] == mode]
        n_images = len(subset)
        if n_images == 0:
            result[mode] = np.zeros(len(budgets))
            continue

        # Get iteration counts for successful attacks
        success_iters = np.sort(
            subset.loc[subset["success"], "iterations"].values
        )

        # CDF via searchsorted: count of success_iters <= B
        counts = np.searchsorted(success_iters, budgets, side="right")
        result[mode] = counts / n_images

    return result


def compute_square_winrate(df: pd.DataFrame) -> dict:
    """Compute SquareAttack winrate per budget.

    For each mode, winrate(B) = count(success at budget=B) / n_images_at_B.

    Args:
        df: DataFrame filtered to method='SquareAttack' (excluding oracle_probe).

    Returns:
        dict mapping mode -> (budgets_array, winrates_array).
    """
    result = {}
    for mode in MODE_ORDER:
        subset = df[df["mode"] == mode]
        if subset.empty:
            result[mode] = (np.array([]), np.array([]))
            continue

        grouped = subset.groupby("budget")["success"].agg(["sum", "count"])
        grouped = grouped.sort_index()
        budgets = grouped.index.values
        winrates = grouped["sum"].values / grouped["count"].values
        result[mode] = (budgets, winrates)

    return result


# ===========================================================================
# Figures
# ===========================================================================
def fig_winrate(simba_wr, sq_wr, budgets, outdir, show):
    """Combined 6-curve plot: 2 methods x 3 modes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in MODE_ORDER:
        color = MODE_COLORS[mode]
        label_mode = MODE_LABELS[mode]

        # SimBA (solid)
        ax.plot(budgets, simba_wr[mode],
                color=color, linestyle=METHOD_LINESTYLES["SimBA"],
                linewidth=1.5,
                label=f"{METHOD_LABELS['SimBA']} — {label_mode}")

        # SquareAttack (dashed)
        sq_b, sq_w = sq_wr[mode]
        if len(sq_b) > 0:
            ax.plot(sq_b, sq_w,
                    color=color, linestyle=METHOD_LINESTYLES["SquareAttack"],
                    linewidth=1.5,
                    label=f"{METHOD_LABELS['SquareAttack']} — {label_mode}")

    ax.set_xlabel("Query budget")
    ax.set_ylabel("Success rate (winrate)")
    ax.set_xlim(0, budgets[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Success Rate vs Query Budget (ResNet-50)")

    _savefig(fig, outdir, "fig_winrate")
    if show:
        plt.show()
    plt.close(fig)


def fig_winrate_by_mode(simba_wr, sq_wr, budgets, outdir, show):
    """3 subplots (one per mode), 2 curves each."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, mode in zip(axes, MODE_ORDER):
        color = MODE_COLORS[mode]
        label_mode = MODE_LABELS[mode]

        # SimBA
        ax.plot(budgets, simba_wr[mode],
                color=color, linestyle=METHOD_LINESTYLES["SimBA"],
                linewidth=1.5,
                label=METHOD_LABELS["SimBA"])

        # SquareAttack
        sq_b, sq_w = sq_wr[mode]
        if len(sq_b) > 0:
            ax.plot(sq_b, sq_w,
                    color=color, linestyle=METHOD_LINESTYLES["SquareAttack"],
                    linewidth=1.5,
                    label=METHOD_LABELS["SquareAttack"])

        ax.set_xlabel("Query budget")
        ax.set_title(label_mode)
        ax.set_xlim(0, budgets[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", framealpha=0.9)

    axes[0].set_ylabel("Success rate (winrate)")

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
    args = parser.parse_args()

    _setup_style()

    print(f"Loading data from {args.csv}")
    df = load_data(args.csv)

    os.makedirs(args.outdir, exist_ok=True)

    # Split by method
    df_simba = df[df["method"] == "SimBA"]
    df_square = df[
        (df["method"] == "SquareAttack") & (df["mode"] != "oracle_probe")
    ]

    n_simba = df_simba["image"].nunique()
    n_square = df_square["image"].nunique()
    print(f"SimBA: {len(df_simba)} rows, {n_simba} images")
    print(f"SquareAttack: {len(df_square)} rows, {n_square} images")

    # Determine budget range for SimBA CDF evaluation
    # Use SquareAttack budgets as x-axis, plus extend to SimBA max
    simba_max = int(df_simba["budget"].max()) if len(df_simba) > 0 else 20_000
    sq_budgets_raw = sorted(df_square["budget"].unique()) if len(df_square) > 0 else []
    sq_step = (sq_budgets_raw[1] - sq_budgets_raw[0]) if len(sq_budgets_raw) > 1 else 50
    # Unified budget axis: from sq_step to simba_max in steps
    budgets = np.arange(sq_step, simba_max + 1, sq_step)

    print(f"Budget axis: {budgets[0]}..{budgets[-1]} (step={sq_step}, "
          f"{len(budgets)} points)")

    # Compute winrates
    simba_wr = compute_simba_winrate(df_simba, budgets)
    sq_wr = compute_square_winrate(df_square)

    # Print summary
    for mode in MODE_ORDER:
        simba_final = simba_wr[mode][-1] if len(simba_wr[mode]) > 0 else 0
        sq_b, sq_w = sq_wr[mode]
        sq_final = sq_w[-1] if len(sq_w) > 0 else 0
        print(f"  {mode:>14s}: SimBA={simba_final:.1%} (at {budgets[-1]}), "
              f"SquareAttack={sq_final:.1%} (at {sq_b[-1] if len(sq_b) > 0 else 'N/A'})")

    # Generate figures
    print(f"\nGenerating figures in {args.outdir}/")
    fig_winrate(simba_wr, sq_wr, budgets, args.outdir, args.show)
    fig_winrate_by_mode(simba_wr, sq_wr, budgets, args.outdir, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
