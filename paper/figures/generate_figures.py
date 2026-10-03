"""
generate_figures.py
===================
Generate all figures for the DGE paper from raw experimental JSON data.

Usage:
    cd paper/figures
    python generate_figures.py

Outputs (PDF, 300 DPI):
    convergence_mnist.pdf     — Fig 1: Convergence curves on full MNIST
    non_diff_barplot.pdf      — Fig 2: Non-differentiable architectures
    synthetic_benchmarks.pdf  — Fig 3: Synthetic function optimization
    ablation_components.pdf   — Fig 4: Component ablation study
    convergence_6seeds.pdf    — Fig 5: 6-seed convergence (3K subset)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Configuration ----------
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "raw"
OUT_DIR = Path(__file__).resolve().parent
DPI = 300

# Consistent style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "SPSA": "#e74c3c",
    "MeZO": "#e67e22",
    "PureDGE": "#3498db",
    "ConsistencyDGE": "#2ecc71",
    "SGD": "#9b59b6",
    "Adam": "#2c3e50",
    "SMA (T=20)": "#e74c3c",
    "DS-EMA": "#2ecc71",
}


def load_json(name):
    path = RAW_DIR / name
    with open(path) as f:
        return json.load(f)


def avg_curves(results, method_name):
    """Average convergence curves across seeds for a given method."""
    curves = [r["curve_acc"] for r in results if r["method"] == method_name]
    evals = [r["curve_evals"] for r in results if r["method"] == method_name]
    if not curves:
        return None, None
    # Use the shortest evals grid for averaging
    min_len = min(len(e) for e in evals)
    evals_grid = evals[0][:min_len]
    arr = np.array([c[:min_len] for c in curves])
    return np.array(evals_grid), arr.mean(axis=0)


def avg_curves_std(results, method_name):
    """Average convergence curves across seeds, return mean and std."""
    curves = [r["curve_acc"] for r in results if r["method"] == method_name]
    evals = [r["curve_evals"] for r in results if r["method"] == method_name]
    if not curves:
        return None, None, None
    min_len = min(len(e) for e in evals)
    evals_grid = evals[0][:min_len]
    arr = np.array([c[:min_len] for c in curves])
    return np.array(evals_grid), arr.mean(axis=0), arr.std(axis=0)


# ================================================================
# Figure 1: Convergence on Full MNIST (v30e)
# ================================================================
def fig1_convergence_mnist():
    data = load_json("v30e_fullmnist_comparison.json")
    results = data["results"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    methods = ["SPSA", "PureDGE", "ConsistencyDGE", "Adam"]
    for method in methods:
        evals, mean = avg_curves(results, method)
        if evals is None:
            continue
        evals_k = np.array(evals) / 1000
        label = method if method != "ConsistencyDGE" else "DGE (DS-EMA)"
        ax.plot(evals_k, mean, label=label, color=COLORS.get(method, "#333"),
                linestyle="--" if method in ("SPSA",) else "-")

    ax.set_xlabel("Function Evaluations (×10³)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Convergence on MNIST (60K samples)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.98, color="gray", linestyle=":", alpha=0.5, label="Adam ceiling")
    fig.savefig(OUT_DIR / "convergence_mnist.pdf")
    plt.close(fig)
    print("  [OK] convergence_mnist.pdf")


# ================================================================
# Figure 2: Non-Differentiable Architectures
# ================================================================
def fig2_non_diff_barplot():
    # Data from findings v11, v31, v32
    architectures = ["Sign\nActivations", "INT8\nQuantized", "INT4\nQuantized"]
    adam_acc = [61.20, 8.40, 9.30]
    dge_acc = [73.20, 82.20, 77.80]

    x = np.arange(len(architectures))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars1 = ax.bar(x - width / 2, adam_acc, width, label="Adam (backprop)",
                   color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, dge_acc, width, label="DGE (zeroth-order)",
                   color="#2ecc71", alpha=0.85)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Non-Differentiable Architectures (no STE)")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.set_ylim(0, 100)
    fig.savefig(OUT_DIR / "non_diff_barplot.pdf")
    plt.close(fig)
    print("  [OK] non_diff_barplot.pdf")


# ================================================================
# Figure 3: Synthetic Benchmarks (v67)
# ================================================================
def fig3_synthetic_benchmarks():
    data = load_json("v67_dual_ema.json")

    benchmarks = ["rosenbrock", "rotated_quadratic", "ellipsoid", "sphere"]
    titles = ["Rosenbrock", "Rotated Quadratic", "Ellipsoid (κ=10⁶)", "Sphere (8192D)"]
    methods = ["PureDGE", "ConsistencyDGE_T20", "DualEMADGE_v67"]
    labels = ["PureDGE", "Consistency (T=20)", "DS-EMA"]
    colors = [COLORS["PureDGE"], COLORS["SMA (T=20)"], COLORS["DS-EMA"]]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes = axes.flatten()

    for i, (bench, title) in enumerate(zip(benchmarks, titles)):
        ax = axes[i]
        for method, label, color in zip(methods, labels, colors):
            losses = [d["loss"] for d in data if d["benchmark"] == bench and d["method"] == method]
            if not losses:
                continue
            mean = np.mean(losses)
            std = np.std(losses)
            # Use log scale for all except Rosenbrock
            if bench == "sphere":
                # Sphere already near zero, use scientific notation
                ax.bar(label, mean, yerr=std, color=color, alpha=0.8, capsize=3)
                ax.set_yscale("log")
            elif bench == "rotated_quadratic":
                ax.bar(label, mean, yerr=std, color=color, alpha=0.8, capsize=3)
                ax.set_yscale("log")
            else:
                ax.bar(label, mean, yerr=std, color=color, alpha=0.8, capsize=3)

        ax.set_title(title, fontsize=10)
        if i >= 2:
            ax.set_xlabel("")
        if i % 2 == 0:
            ax.set_ylabel("Final Loss")
        # Rotate x labels for readability
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Synthetic Benchmarks (200K evaluations, 5 seeds)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "synthetic_benchmarks.pdf")
    plt.close(fig)
    print("  [OK] synthetic_benchmarks.pdf")


# ================================================================
# Figure 4: Component Ablation Study
# ================================================================
def fig4_ablation_components():
    v70_path = RAW_DIR / "v70_ablation_study.json"
    v30e_path = RAW_DIR / "v30e_fullmnist_comparison.json"

    components = []
    accuracy = []
    stds = []
    colors_ablation = []

    color_map = {
        "SPSA": "#e74c3c",
        "MeZO": "#e67e22",
        "Block-SGD": "#f39c12",
        "Block-Adam": "#3498db",
        "DGE Full": "#2ecc71",
    }

    if v70_path.exists():
        with open(v70_path) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        mapping = [
            ("SPSA", "SPSA\n(Global, Adam)"),
            ("MeZO", "MeZO\n(Global, SGD)"),
            ("PureDGE_SGD", "Block-SGD\n(Blocks, no EMA)"),
            ("PureDGE_Adam", "Block-Adam\n(Blocks + EMA)"),
            ("DGE_Full", "DGE Full\n(+ Consistency)"),
        ]
        for key, label in mapping:
            if key in summary:
                components.append(label)
                accuracy.append(summary[key]["best_acc_mean"] * 100.0)
                stds.append(summary[key].get("best_acc_std", 0.0) * 100.0)
                short_name = key.replace("PureDGE_", "Block-").replace("_", " ")
                colors_ablation.append(color_map.get(short_name, "#34495e"))

    if not components and v30e_path.exists():
        # Fallback to verified v30e data (without unverified/fabricated intermediate steps)
        with open(v30e_path) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        mapping = [
            ("SPSA", "SPSA\n(Global, Adam)"),
            ("PureDGE", "Block-Adam\n(PureDGE)"),
            ("ConsistencyDGE", "DGE Full\n(+ Consistency)"),
        ]
        for key, label in mapping:
            if key in summary:
                components.append(label)
                accuracy.append(summary[key]["mean"] * 100.0)
                stds.append(summary[key].get("std", 0.0) * 100.0)
                short_name = "SPSA" if "SPSA" in key else ("Block-Adam" if "Pure" in key else "DGE Full")
                colors_ablation.append(color_map.get(short_name, "#34495e"))

    if not components:
        print("  [SKIP] fig4_ablation_components: No JSON data available in results/raw/")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    xerr = stds if any(s > 0 for s in stds) else None
    bars = ax.barh(components, accuracy, xerr=xerr, color=colors_ablation, alpha=0.85, height=0.6, capsize=4)

    # Add value labels
    for bar, acc, std in zip(bars, accuracy, stds):
        label_text = f"{acc:.2f}%" if std == 0 else f"{acc:.2f}±{std:.2f}%"
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                label_text, va="center", fontsize=8.5)

    ax.set_xlabel("Best Test Accuracy (%)")
    ax.set_title("Ablation: Contribution of Each Component (Full MNIST)")
    ax.set_xlim(0, 108)
    ax.axvline(x=98.0, color="gray", linestyle=":", alpha=0.5)
    ax.text(98.5, len(components) - 0.7, "Adam (98.0%)", fontsize=8, color="gray", va="center")
    fig.savefig(OUT_DIR / "ablation_components.pdf")
    plt.close(fig)
    print("  [OK] ablation_components.pdf")


# ================================================================
# Figure 5: 6-Seed Convergence (v29, 3K subset)
# ================================================================
def fig5_convergence_6seeds():
    data = load_json("v29_paper_stats.json")
    results = data["results"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for method, label, color in [
        ("PureDGE", "PureDGE", COLORS["PureDGE"]),
        ("ConsistencyDGE", "DGE + Consistency", COLORS["ConsistencyDGE"]),
    ]:
        evals, mean, std = avg_curves_std(results, method)
        evals_k = np.array(evals) / 1000
        ax.plot(evals_k, mean, label=label, color=color)
        ax.fill_between(evals_k, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Function Evaluations (×10³)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Convergence on MNIST-3K (6 seeds, ±1 std)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.0)
    fig.savefig(OUT_DIR / "convergence_6seeds.pdf")
    plt.close(fig)
    print("  [OK] convergence_6seeds.pdf")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("Generating figures for DGE paper...")
    fig1_convergence_mnist()
    fig2_non_diff_barplot()
    fig3_synthetic_benchmarks()
    fig4_ablation_components()
    fig5_convergence_6seeds()
    print("Done. All figures saved to:", OUT_DIR)
