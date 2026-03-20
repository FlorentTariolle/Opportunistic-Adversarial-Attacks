"""
Multi-Seed Validation Analysis (#24)

Analyzes cross-seed consistency of OTS results on ResNet-50.
Generates per-seed statistics, Wilcoxon tests, and publication figures.

Usage:
    python analyze_multiseed.py
    python analyze_multiseed.py --csv results/benchmark_multiseed.csv
    python analyze_multiseed.py --show
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# Style (matches analyze_benchmark.py)
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


# ===========================================================================
# Colors & constants
# ===========================================================================
MODE_COLORS = {
    "untargeted": "#4878CF",
    "targeted": "#E8873A",
    "opportunistic": "#6BA353",
}
MODE_ORDER = ["untargeted", "targeted", "opportunistic"]
MODE_LABELS = {
    "untargeted": "Untargeted",
    "targeted": "Targeted (oracle)",
    "opportunistic": "Opportunistic",
}

COLOR_UNTARGETED = "#4878CF"
COLOR_OPPORTUNISTIC = "#D64541"

SEED_COLORS = ["#4878CF", "#E8873A", "#6BA353", "#D64541", "#8B5CF6"]

METHODS = ["SimBA", "SquareAttack"]
MAX_ITER = 15_000


# ===========================================================================
# Data loading
# ===========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["switch_iteration"] = pd.to_numeric(df["switch_iteration"], errors="coerce")
    df["locked_class"] = pd.to_numeric(df["locked_class"], errors="coerce")
    df["oracle_target"] = pd.to_numeric(df["oracle_target"], errors="coerce")
    df["adversarial_class"] = pd.to_numeric(df["adversarial_class"], errors="coerce")
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df


def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


def _sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ===========================================================================
# Per-seed Wilcoxon tests
# ===========================================================================
def compute_perseed_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test per (method, seed) comparing untargeted vs opportunistic."""
    key_cols = ["method", "seed", "image"]
    rows = []

    for method in df["method"].unique():
        for seed in sorted(df["seed"].unique()):
            unt = df[(df["method"] == method) & (df["seed"] == seed) &
                     (df["mode"] == "untargeted")][["image", "iterations", "success"]]
            opp = df[(df["method"] == method) & (df["seed"] == seed) &
                     (df["mode"] == "opportunistic")][["image", "iterations", "success"]]

            merged = unt.merge(opp, on="image", suffixes=("_unt", "_opp"))
            # Only paired successes for iteration comparison
            both_ok = merged[merged["success_unt"] & merged["success_opp"]]

            sr_unt = merged["success_unt"].mean()
            sr_opp = merged["success_opp"].mean()

            if len(both_ok) >= 2:
                va = both_ok["iterations_unt"].values
                vb = both_ok["iterations_opp"].values
                try:
                    w_stat, w_p = stats.wilcoxon(va, vb, alternative="two-sided")
                except ValueError:
                    w_stat, w_p = 0.0, 1.0
                med_unt = np.median(va)
                med_opp = np.median(vb)
                savings = (med_unt - med_opp) / med_unt * 100 if med_unt > 0 else 0.0
            else:
                w_stat, w_p, med_unt, med_opp, savings = 0, 1.0, 0, 0, 0

            rows.append({
                "method": method, "seed": seed,
                "n_images": len(merged), "n_paired": len(both_ok),
                "sr_unt": sr_unt, "sr_opp": sr_opp,
                "sr_delta": sr_opp - sr_unt,
                "med_unt": med_unt, "med_opp": med_opp,
                "savings_pct": savings,
                "w_stat": w_stat, "w_p": w_p,
            })

    result = pd.DataFrame(rows)
    # Bonferroni correction (per method: 5 seeds)
    for method in result["method"].unique():
        mask = result["method"] == method
        n_tests = mask.sum()
        result.loc[mask, "w_p_bonf"] = (result.loc[mask, "w_p"] * n_tests).clip(upper=1.0)
    result["sig"] = result["w_p_bonf"].apply(_sig_stars)
    return result


def compute_pooled_test(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon test pooling across all seeds per method."""
    rows = []
    for method in df["method"].unique():
        unt = df[(df["method"] == method) & (df["mode"] == "untargeted")][
            ["image", "seed", "iterations", "success"]]
        opp = df[(df["method"] == method) & (df["mode"] == "opportunistic")][
            ["image", "seed", "iterations", "success"]]

        merged = unt.merge(opp, on=["image", "seed"], suffixes=("_unt", "_opp"))
        both_ok = merged[merged["success_unt"] & merged["success_opp"]]

        sr_unt = merged["success_unt"].mean()
        sr_opp = merged["success_opp"].mean()

        if len(both_ok) >= 2:
            va = both_ok["iterations_unt"].values
            vb = both_ok["iterations_opp"].values
            w_stat, w_p = stats.wilcoxon(va, vb, alternative="two-sided")
            med_unt, med_opp = np.median(va), np.median(vb)
            savings = (med_unt - med_opp) / med_unt * 100
        else:
            w_stat, w_p, med_unt, med_opp, savings = 0, 1.0, 0, 0, 0

        rows.append({
            "method": method,
            "n_paired": len(both_ok),
            "sr_unt": sr_unt, "sr_opp": sr_opp,
            "sr_delta": sr_opp - sr_unt,
            "med_unt": med_unt, "med_opp": med_opp,
            "savings_pct": savings,
            "w_stat": w_stat, "w_p": w_p,
            "sig": _sig_stars(w_p),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================

def fig_success_rate_seeds(df: pd.DataFrame, outdir: str):
    """Per-seed success rate: grouped bars showing consistency across seeds."""
    seeds = sorted(df["seed"].unique())
    methods = [m for m in METHODS if m in df["method"].unique()]

    sr = df.groupby(["method", "mode", "seed"], observed=True)["success"].mean().reset_index()

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = sr[sr["method"] == method]
        x = np.arange(len(seeds))
        width = 0.25

        for i, mode in enumerate(MODE_ORDER):
            vals = sub[sub["mode"] == mode].set_index("seed").reindex(seeds)["success"] * 100
            ax.bar(
                x + (i - 1) * width, vals, width,
                color=MODE_COLORS[mode], edgecolor="white", linewidth=0.5,
                label=MODE_LABELS[mode] if ax is axes[0] else None,
            )

        # Mean ± std annotation for each mode
        for i, mode in enumerate(MODE_ORDER):
            vals = sub[sub["mode"] == mode]["success"] * 100
            mean, std = vals.mean(), vals.std()
            ax.axhline(mean, color=MODE_COLORS[mode], linestyle="--", alpha=0.4, linewidth=1)
            ax.text(
                len(seeds) - 0.5, mean + 1,
                f"{mean:.1f} $\\pm$ {std:.1f}",
                fontsize=8, color=MODE_COLORS[mode], ha="right", va="bottom",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.set_title(method, fontweight="bold")
        ax.set_ylim(0, 115)
        if ax is axes[0]:
            ax.set_ylabel(r"Success Rate (\%)")

    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m]) for m in MODE_ORDER],
        [MODE_LABELS[m] for m in MODE_ORDER],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Per-Seed Success Rate — ResNet-50", fontsize=14)
    _savefig(fig, outdir, "fig_multiseed_success_rate")
    return fig


def fig_iterations_seeds(df: pd.DataFrame, outdir: str):
    """Boxplot of iteration counts per seed, untargeted vs opportunistic (successful runs)."""
    methods = [m for m in METHODS if m in df["method"].unique()]
    ok = df[df["success"] & df["mode"].isin(["untargeted", "opportunistic"])].copy()

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), sharey=False)
    if len(methods) == 1:
        axes = [axes]

    seeds = sorted(df["seed"].unique())

    for ax, method in zip(axes, methods):
        sub = ok[ok["method"] == method]
        positions = []
        data_unt, data_opp = [], []

        for i, seed in enumerate(seeds):
            unt_vals = sub[(sub["seed"] == seed) & (sub["mode"] == "untargeted")]["iterations"]
            opp_vals = sub[(sub["seed"] == seed) & (sub["mode"] == "opportunistic")]["iterations"]
            data_unt.append(unt_vals.values)
            data_opp.append(opp_vals.values)

        x = np.arange(len(seeds))
        w = 0.35

        bp_unt = ax.boxplot(
            data_unt, positions=x - w/2, widths=w * 0.8,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=COLOR_UNTARGETED, alpha=0.6),
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp_opp = ax.boxplot(
            data_opp, positions=x + w/2, widths=w * 0.8,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=COLOR_OPPORTUNISTIC, alpha=0.6),
            medianprops=dict(color="black", linewidth=1.5),
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.set_title(method, fontweight="bold")
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            ax.set_ylabel("Iterations (successful runs)")
            ax.legend(
                [bp_unt["boxes"][0], bp_opp["boxes"][0]],
                ["Untargeted", "Opportunistic"],
                loc="upper right",
            )

    fig.suptitle("Iteration Distribution per Seed — ResNet-50", fontsize=14)
    _savefig(fig, outdir, "fig_multiseed_iterations")
    return fig


def fig_cdf_seeds(df: pd.DataFrame, outdir: str):
    """Per-seed CDF curves: one subplot per method, colored by seed, solid=unt dashed=opp."""
    methods = [m for m in METHODS if m in df["method"].unique()]
    seeds = sorted(df["seed"].unique())

    budgets = np.arange(50, MAX_ITER + 1, 50)

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]

        for si, seed in enumerate(seeds):
            color = SEED_COLORS[si % len(SEED_COLORS)]

            for mode, ls in [("untargeted", "-"), ("opportunistic", "--")]:
                group = sub[(sub["seed"] == seed) & (sub["mode"] == mode)]
                n_total = len(group)
                if n_total == 0:
                    continue

                asr = []
                for q in budgets:
                    n_ok = ((group["success"]) & (group["iterations"] <= q)).sum()
                    asr.append(n_ok / n_total)

                label = None
                if mode == "untargeted":
                    label = f"Seed {seed}"
                ax.plot(budgets, asr, color=color, linestyle=ls, linewidth=1.2,
                        alpha=0.8, label=label)

        # Mean CDF across seeds (thick lines)
        for mode, ls, color in [("untargeted", "-", COLOR_UNTARGETED),
                                 ("opportunistic", "--", COLOR_OPPORTUNISTIC)]:
            all_asr = []
            for seed in seeds:
                group = sub[(sub["seed"] == seed) & (sub["mode"] == mode)]
                n_total = len(group)
                if n_total == 0:
                    continue
                asr = []
                for q in budgets:
                    n_ok = ((group["success"]) & (group["iterations"] <= q)).sum()
                    asr.append(n_ok / n_total)
                all_asr.append(asr)

            if all_asr:
                mean_asr = np.mean(all_asr, axis=0)
                tag = "Untargeted" if mode == "untargeted" else "Opportunistic"
                ax.plot(budgets, mean_asr, color=color, linestyle=ls,
                        linewidth=2.5, label=f"Mean — {tag}")

        ax.set_xlabel("Query Budget")
        ax.set_title(method, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(budgets[0], budgets[-1])
        if ax is axes[0]:
            ax.set_ylabel("Success Rate")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("CDF per Seed — ResNet-50 (solid = untargeted, dashed = opportunistic)",
                 fontsize=13)
    _savefig(fig, outdir, "fig_multiseed_cdf")
    return fig


def fig_effect_size(df: pd.DataFrame, outdir: str, test_results: pd.DataFrame):
    """Per-seed effect size: success rate delta and iteration savings with error bars."""
    methods = [m for m in METHODS if m in df["method"].unique()]
    seeds = sorted(df["seed"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for mi, method in enumerate(methods):
        sub = test_results[test_results["method"] == method].sort_values("seed")
        x = sub["seed"].values
        offset = (mi - 0.5 * (len(methods) - 1)) * 0.15

        # Success rate delta
        ax1.bar(
            x + offset, sub["sr_delta"].values * 100, 0.3,
            color=MODE_COLORS["opportunistic"] if mi == 0 else MODE_COLORS["targeted"],
            edgecolor="white", linewidth=0.5, label=method,
        )
        for _, row in sub.iterrows():
            marker = row["sig"]
            if marker != "ns":
                ax1.text(row["seed"] + offset, row["sr_delta"] * 100 + 1,
                         marker, ha="center", fontsize=8, color="0.3")

        # Iteration savings %
        ax2.bar(
            x + offset, sub["savings_pct"].values, 0.3,
            color=MODE_COLORS["opportunistic"] if mi == 0 else MODE_COLORS["targeted"],
            edgecolor="white", linewidth=0.5, label=method,
        )
        for _, row in sub.iterrows():
            marker = row["sig"]
            if marker != "ns":
                ax2.text(row["seed"] + offset, row["savings_pct"] + 1,
                         marker, ha="center", fontsize=8, color="0.3")

    # Mean lines
    for mi, method in enumerate(methods):
        sub = test_results[test_results["method"] == method]
        color = MODE_COLORS["opportunistic"] if mi == 0 else MODE_COLORS["targeted"]
        ax1.axhline(sub["sr_delta"].mean() * 100, color=color, linestyle="--",
                     alpha=0.5, linewidth=1)
        ax2.axhline(sub["savings_pct"].mean(), color=color, linestyle="--",
                     alpha=0.5, linewidth=1)

    ax1.set_xticks(seeds)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylabel(r"$\Delta$ Success Rate (pp)")
    ax1.set_title("Success Rate Gain (OTS vs Untargeted)")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.legend()

    ax2.set_xticks(seeds)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylabel(r"Iteration Savings (\%)")
    ax2.set_title("Median Iteration Savings (OTS vs Untargeted)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend()

    fig.suptitle("Cross-Seed Effect Size — ResNet-50", fontsize=14)
    _savefig(fig, outdir, "fig_multiseed_effect_size")
    return fig


# ===========================================================================
# Summary printing
# ===========================================================================

def print_summary(df: pd.DataFrame, perseed: pd.DataFrame, pooled: pd.DataFrame):
    print("\n" + "=" * 80)
    print("MULTI-SEED VALIDATION SUMMARY — ResNet-50")
    print("=" * 80)

    seeds = sorted(df["seed"].unique())
    print(f"\nSeeds: {seeds}")
    print(f"Total rows: {len(df)}")

    # Per-seed success rates
    print("\n--- Per-Seed Success Rates ---")
    sr = df.groupby(["method", "mode", "seed"], observed=True)["success"].mean() * 100
    sr_pivot = sr.reset_index().pivot_table(
        index=["method", "seed"], columns="mode", values="success", observed=True,
    )
    sr_pivot = sr_pivot[[m for m in MODE_ORDER if m in sr_pivot.columns]]
    print(sr_pivot.round(1).to_string())

    # Cross-seed mean ± std
    print("\n--- Cross-Seed Mean ± Std (Success Rate %) ---")
    sr2 = df.groupby(["method", "mode", "seed"], observed=True)["success"].mean() * 100
    cross = sr2.reset_index().groupby(["method", "mode"], observed=True)["success"].agg(
        ["mean", "std"]).round(1)
    print(cross.to_string())

    # Per-seed Wilcoxon tests
    print("\n--- Per-Seed Wilcoxon Tests (Untargeted vs Opportunistic) ---")
    header = (f"{'Method':<14} {'Seed':>4} {'N':>4} {'SR_unt':>7} {'SR_opp':>7} "
              f"{'ΔSR':>6} {'Med_unt':>8} {'Med_opp':>8} {'Sav%':>6} "
              f"{'W':>8} {'p':>9} {'p_adj':>8} {'Sig':>4}")
    print(header)
    print("-" * len(header))
    for _, r in perseed.iterrows():
        print(f"{r['method']:<14} {r['seed']:>4} {r['n_paired']:>4} "
              f"{r['sr_unt']*100:>6.1f}% {r['sr_opp']*100:>6.1f}% "
              f"{r['sr_delta']*100:>+5.1f} "
              f"{r['med_unt']:>8.0f} {r['med_opp']:>8.0f} {r['savings_pct']:>+5.1f}% "
              f"{r['w_stat']:>8.0f} {r['w_p']:>9.4g} {r['w_p_bonf']:>8.4g} {r['sig']:>4}")

    # Pooled test
    print("\n--- Pooled Wilcoxon Tests (all seeds) ---")
    for _, r in pooled.iterrows():
        print(f"{r['method']:<14} N={r['n_paired']:>4}  "
              f"SR: {r['sr_unt']*100:.1f}% → {r['sr_opp']*100:.1f}% (Δ={r['sr_delta']*100:+.1f}pp)  "
              f"Med: {r['med_unt']:.0f} → {r['med_opp']:.0f} ({r['savings_pct']:+.1f}%)  "
              f"p={r['w_p']:.4g} {r['sig']}")

    # Cross-seed consistency of significance
    print("\n--- Significance Consistency ---")
    for method in perseed["method"].unique():
        sub = perseed[perseed["method"] == method]
        n_sig = (sub["sig"] != "ns").sum()
        n_total = len(sub)
        print(f"  {method}: {n_sig}/{n_total} seeds significant (Bonferroni-corrected)")

    print("\n" + "=" * 80)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-seed validation analysis.")
    parser.add_argument("--csv", default="results/benchmark_multiseed.csv")
    parser.add_argument("--outdir", default="results/figures/multiseed")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data from {args.csv} ...")
    df = load_data(args.csv)
    print(f"  {len(df)} rows, {df['method'].nunique()} methods, "
          f"{len(df['seed'].unique())} seeds, {df['mode'].nunique()} modes")

    # --- Statistical tests ---
    print("\n=== Per-Seed Wilcoxon Tests ===")
    perseed = compute_perseed_tests(df)
    pooled = compute_pooled_test(df)

    # --- Figures ---
    print("\n=== Figures ===")
    print("\n--- Per-Seed Success Rate ---")
    fig_success_rate_seeds(df, args.outdir)

    print("\n--- Per-Seed Iteration Boxplots ---")
    fig_iterations_seeds(df, args.outdir)

    print("\n--- Per-Seed CDF ---")
    fig_cdf_seeds(df, args.outdir)

    print("\n--- Effect Size ---")
    fig_effect_size(df, args.outdir, perseed)

    # --- Summary ---
    print_summary(df, perseed, pooled)

    if args.show:
        plt.show()

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
