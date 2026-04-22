"""
dge_figures_v30d.py
====================
Genera figuras para el paper desde los datos de v30d (comparacion completa).

Figuras:
  1. comparison_bar.png       — Bar chart de accuracy final por metodo
  2. convergence_dge.png      — Curvas de convergencia DGE vs PureDGE (full budget)
  3. collapse_spsa.png        — Curva de colapso SPSA/MeZO (muestra el fracaso)
  4. summary_table.png        — Tabla resumen visual para el paper
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = {
    "SPSA":          "#e07070",
    "MeZO":          "#e09050",
    "PureDGE":       "#6c8ebf",
    "ConsistencyDGE":"#d6813a",
    "SGD":           "#7ab87a",
    "Adam":          "#5aaa62",
}
HATCHES = {
    "SPSA": "///", "MeZO": "///",
    "PureDGE": "", "ConsistencyDGE": "",
    "SGD": "...", "Adam": "...",
}

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data_path = ROOT / "results" / "raw" / "v30d_fullmnist_comparison.json"
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

results  = data["results"]
summary  = data["summary"]
seeds    = data["seeds"]
ORDER    = ["SPSA", "MeZO", "PureDGE", "ConsistencyDGE", "SGD", "Adam"]

# ---------------------------------------------------------------------------
# Figure 1 — Grouped bar chart: método × accuracy
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.0))

x    = np.arange(len(ORDER))
means = [summary[m]["mean"] * 100 for m in ORDER]
stds  = [summary[m]["std"]  * 100 for m in ORDER]

bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
              color=[COLORS[m] for m in ORDER],
              hatch=[HATCHES[m] for m in ORDER],
              edgecolor="white", linewidth=1.2, error_kw=dict(elinewidth=1.5))

# Value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.8,
            f"{mean:.1f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=bar.get_facecolor())

ax.set_xticks(x)
ax.set_xticklabels(ORDER, rotation=15, ha="right")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("MNIST Benchmark — Full Dataset (60K train / 10K test)\n"
             "MLP [784→128→64→10] ~109K params  |  3 seeds  |  mean ± std")
ax.set_ylim(0, 105)

# Annotations
ax.axhline(summary["Adam"]["mean"]*100, color=COLORS["Adam"],
           linestyle="--", lw=1.2, alpha=0.6, label="Adam ceiling")
ax.axhline(10, color="#888", linestyle=":", lw=1, alpha=0.5, label="Random chance (10%)")

# Legend for hatch groups
patches = [
    mpatches.Patch(facecolor="#ddd", hatch="///", label="Zero-order (global pert.)"),
    mpatches.Patch(facecolor="#ddd", hatch="",    label="Zero-order (block pert.)"),
    mpatches.Patch(facecolor="#ddd", hatch="...", label="Gradient (backprop)"),
]
ax.legend(handles=patches, loc="upper left", fontsize=9)
fig.tight_layout()
out = OUT_DIR / "comparison_bar.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 2 — Convergence curves: PureDGE vs ConsistencyDGE (full 3M budget)
# ---------------------------------------------------------------------------

def get_curves(method):
    evs_list, acc_list = [], []
    for r in results:
        if r["method"] != method: continue
        evs_list.append(r["curve_evals"])
        acc_list.append(r["curve_acc"])
    # Align to shortest
    min_len = min(len(e) for e in evs_list)
    evals_k = np.array(evs_list[0][:min_len]) / 1_000
    accs    = np.array([a[:min_len] for a in acc_list]) * 100
    return evals_k, accs

fig, ax = plt.subplots(figsize=(7.5, 4.8))

for method in ["PureDGE", "ConsistencyDGE"]:
    evals_k, accs = get_curves(method)
    if len(evals_k) == 0:
        continue
    mean = accs.mean(axis=0)
    std  = accs.std(axis=0)
    color = COLORS[method]
    label = f"{method} ({mean[-1]:.1f}% ± {std[-1]:.1f}%, n={len(seeds)})"
    ax.plot(evals_k, mean, color=color, lw=2.2, label=label)
    ax.fill_between(evals_k, mean-std, mean+std, color=color, alpha=0.15)
    # Annotate final value
    ax.annotate(f"{mean[-1]:.1f}%",
                xy=(evals_k[-1], mean[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=color, fontweight="bold", va="center", fontsize=10)

# Adam reference line
adam_mean = summary["Adam"]["mean"] * 100
ax.axhline(adam_mean, color=COLORS["Adam"], linestyle="--", lw=1.4,
           alpha=0.7, label=f"Adam (backprop) {adam_mean:.1f}%")

ax.set_xlabel("Function Evaluations (×1,000)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Convergence on Full MNIST — Block Zeroth-Order Methods\n"
             "Adam reference shown (backprop, 30 epochs)")
ax.legend(loc="lower right")
ax.set_ylim(55, 100)
ax.set_xlim(0, evals_k[-1] + 150 if len(evals_k) > 0 else 3100)
fig.tight_layout()
out = OUT_DIR / "convergence_dge_fullmnist.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 3 — SPSA/MeZO collapse curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 4.2))

for method in ["SPSA", "MeZO"]:
    evs_list, acc_list = [], []
    for r in results:
        if r["method"] != method: continue
        evs_list.append(r["curve_evals"])
        acc_list.append(r["curve_acc"])
    if not evs_list: continue
    min_len = min(len(e) for e in evs_list)
    evals_k = np.array(evs_list[0][:min_len]) / 1_000
    accs    = np.array([a[:min_len] for a in acc_list]) * 100
    mean = accs.mean(axis=0)
    std  = accs.std(axis=0)
    color = COLORS[method]
    ax.plot(evals_k, mean, color=color, lw=2.0, label=f"{method} (mean, n={len(seeds)})")
    for row in accs:
        ax.plot(evals_k, row, color=color, lw=0.8, alpha=0.45)

# Random chance and PureDGE plateau reference
ax.axhline(10, color="#888", linestyle=":", lw=1.2, alpha=0.7, label="Random chance (10%)")
ax.axhline(summary["PureDGE"]["mean"]*100, color=COLORS["PureDGE"],
           linestyle="--", lw=1.2, alpha=0.6,
           label=f"PureDGE plateau ({summary['PureDGE']['mean']*100:.1f}%)")

ax.set_xlabel("Function Evaluations (×1,000)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Global Perturbation Methods Collapse on 109K-Param Network\n"
             "SPSA & MeZO — gradient SNR ~ 1/√D (D≈109K)")
ax.legend(loc="upper right")
ax.set_ylim(0, 80)
fig.tight_layout()
out = OUT_DIR / "collapse_spsa_mezo.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 4 — Summary panel (2x2): all key comparisons for paper
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: final bar chart (simplified, 4 key methods)
ax = axes[0]
key_methods = ["SPSA", "PureDGE", "ConsistencyDGE", "Adam"]
x2 = np.arange(len(key_methods))
means2 = [summary[m]["mean"]*100 for m in key_methods]
stds2  = [summary[m]["std"]*100  for m in key_methods]
bars2 = ax.bar(x2, means2, yerr=stds2, capsize=5, width=0.55,
               color=[COLORS[m] for m in key_methods],
               edgecolor="white", lw=1.2,
               error_kw=dict(elinewidth=1.5))
for bar, mean in zip(bars2, means2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{mean:.1f}%", ha="center", fontsize=10, fontweight="bold",
            color=bar.get_facecolor())
ax.set_xticks(x2)
ax.set_xticklabels(key_methods, fontsize=11)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Final Accuracy Comparison\nMNIST (60K/10K)", fontsize=12)
ax.set_ylim(0, 107)

# Gap annotation between ConsistencyDGE and Adam
yann = 101
ax.annotate("", xy=(x2[3], yann), xytext=(x2[2], yann),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
delta = means2[3] - means2[2]
ax.text((x2[2]+x2[3])/2, yann+1, f"−{delta:.1f}pp\n(w/ backprop)",
        ha="center", va="bottom", fontsize=9, color="#555")

# Right: convergence PureDGE vs ConsistencyDGE
ax = axes[1]
for method in ["PureDGE", "ConsistencyDGE"]:
    evals_k, accs = get_curves(method)
    if len(evals_k) == 0:
        continue
    mean = accs.mean(axis=0)
    std  = accs.std(axis=0)
    color = COLORS[method]
    ax.plot(evals_k, mean, color=color, lw=2.2,
            label=f"{method}\n{mean[-1]:.1f}% ± {std[-1]:.1f}%")
    ax.fill_between(evals_k, mean-std, mean+std, color=color, alpha=0.15)

ax.axhline(summary["Adam"]["mean"]*100, color=COLORS["Adam"],
           linestyle="--", lw=1.2, alpha=0.7,
           label=f"Adam {summary['Adam']['mean']*100:.1f}%")
ax.set_xlabel("Function Evaluations (×1,000)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Convergence Curves (mean ± std, n=3)", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(55, 100)

fig.suptitle("ConsistencyDGE: Zero-Order Training on Full MNIST",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out = OUT_DIR / "paper_summary_v30d.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll figures in: {OUT_DIR}")
