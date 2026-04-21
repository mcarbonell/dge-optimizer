"""
dge_figures_v29.py
==================
Genera las figuras para el paper desde los datos de v29.

Figuras producidas:
  1. convergence_curves.png  — Accuracy de test vs evaluaciones (media ± std band)
  2. final_accuracy_dist.png — Boxplot + puntos por seed para comparar distribuciones
  3. seed_improvement.png    — Mejora por seed (bar chart apilado)

Requiere: pip install matplotlib numpy
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuracion de estilo
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linewidth":   0.6,
    "figure.dpi":       150,
})

COLORS = {
    "PureDGE":       "#6c8ebf",   # blue-grey
    "ConsistencyDGE":"#d6813a",   # warm orange
    "Adam":          "#5aaa72",   # green
    "SPSA":          "#b05a7a",   # rose
}

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cargar datos v29
# ---------------------------------------------------------------------------

data_path = Path(__file__).parent.parent / "results" / "raw" / "v29_paper_stats.json"
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
seeds   = data["seeds"]
methods = ["PureDGE", "ConsistencyDGE"]

# ---------------------------------------------------------------------------
# Helper: construir matriz de curvas (seeds x checkpoints) por método
# ---------------------------------------------------------------------------

def build_curves(method):
    """Returns arrays: evals (E,), accs (n_seeds, E)"""
    all_evals = None
    all_accs  = []
    for r in results:
        if r["method"] != method:
            continue
        e = np.array(r["curve_evals"])
        a = np.array(r["curve_acc"])
        if all_evals is None:
            all_evals = e
        min_len = min(len(all_evals), len(e))
        all_accs.append(a[:min_len])
        all_evals = all_evals[:min_len]
    return all_evals / 1_000, np.array(all_accs) * 100   # K evals, %

# ---------------------------------------------------------------------------
# Figure 1: Convergence curves with ± std shading
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.5))

for method in methods:
    color = COLORS[method]
    evals_k, accs = build_curves(method)
    mean = accs.mean(axis=0)
    std  = accs.std(axis=0)
    label = f"{method} (mean ± std, n={len(seeds)})"

    ax.plot(evals_k, mean, color=color, lw=2.2, label=label)
    ax.fill_between(evals_k, mean - std, mean + std,
                    color=color, alpha=0.18)

ax.set_xlabel("Function Evaluations (×1,000)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Convergence on MNIST — ConsistencyDGE vs Pure DGE\n"
             "MLP [784→128→64→10], 3K train, 6 seeds, 800K eval budget")
ax.legend(loc="lower right")
ax.set_ylim(60, 95)
ax.set_xlim(0, evals_k[-1] + 20)

# Annotate final values
for method in methods:
    color = COLORS[method]
    _, accs = build_curves(method)
    final = accs[:, -1].mean()
    ax.annotate(f"{final:.1f}%",
                xy=(evals_k[-1], final),
                xytext=(8, 0), textcoords="offset points",
                color=color, fontweight="bold", va="center", fontsize=10)

fig.tight_layout()
out = OUT_DIR / "convergence_curves.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 2: Final accuracy distribution — boxplot + strip (beeswarm-style)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.5, 4.5))

pos = [1, 2]
box_data = []
for method in methods:
    accs = [r["best_test_acc"] * 100 for r in results if r["method"] == method]
    box_data.append(accs)

bp = ax.boxplot(box_data, positions=pos, widths=0.35,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", lw=2.5),
                whiskerprops=dict(lw=1.4),
                capprops=dict(lw=1.4))

for patch, method in zip(bp["boxes"], methods):
    patch.set_facecolor(COLORS[method])
    patch.set_alpha(0.7)

# Strip plot (individual seeds)
rng = np.random.default_rng(0)
for i, (method, p) in enumerate(zip(methods, pos)):
    accs = [r["best_test_acc"] * 100 for r in results if r["method"] == method]
    jitter = rng.uniform(-0.08, 0.08, len(accs))
    ax.scatter([p + j for j in jitter], accs,
               color=COLORS[method], s=55, zorder=5,
               edgecolors="white", linewidths=0.8, alpha=0.9)
    mean = np.mean(accs)
    ax.hlines(mean, p - 0.22, p + 0.22, colors=COLORS[method], lw=2,
              linestyles="-", zorder=6)
    ax.text(p, mean + 0.6, f"{mean:.2f}%", ha="center", fontsize=10,
            color=COLORS[method], fontweight="bold")

ax.set_xticks(pos)
ax.set_xticklabels(methods)
ax.set_ylabel("Best Test Accuracy (%)")
ax.set_title("Test Accuracy Distribution\n"
             "MNIST, 6 seeds, 800K eval budget")
ax.set_ylim(77, 93)

# Delta annotation
y_ann = 91.5
ax.annotate("", xy=(2, y_ann), xytext=(1, y_ann),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.5))
delta = np.mean([r["best_test_acc"] for r in results if r["method"] == "ConsistencyDGE"]) - \
        np.mean([r["best_test_acc"] for r in results if r["method"] == "PureDGE"])
ax.text(1.5, y_ann + 0.3, f"+{delta*100:.2f}pp", ha="center", fontsize=10,
        color="#333", fontweight="bold")

fig.tight_layout()
out = OUT_DIR / "final_accuracy_dist.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 3: Per-seed improvement bar chart
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 4.0))

pure_accs  = [r["best_test_acc"] * 100 for r in results if r["method"] == "PureDGE"]
cons_accs  = [r["best_test_acc"] * 100 for r in results if r["method"] == "ConsistencyDGE"]
deltas     = [c - p for c, p in zip(cons_accs, pure_accs)]
x          = np.arange(len(seeds))
bar_w      = 0.38

bars_p = ax.bar(x - bar_w / 2, pure_accs, bar_w,
                color=COLORS["PureDGE"], alpha=0.85, label="PureDGE")
bars_c = ax.bar(x + bar_w / 2, cons_accs, bar_w,
                color=COLORS["ConsistencyDGE"], alpha=0.85, label="ConsistencyDGE")

# Delta labels on top of ConsistencyDGE bars
for i, (bar, d) in enumerate(zip(bars_c, deltas)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"+{d:.1f}", ha="center", va="bottom",
            fontsize=9, color=COLORS["ConsistencyDGE"], fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("Best Test Accuracy (%)")
ax.set_title("Per-Seed Accuracy — ConsistencyDGE always wins\n"
             "MNIST [784→128→64→10], 800K eval budget")
ax.legend(loc="lower right")
ax.set_ylim(75, 93)

fig.tight_layout()
out = OUT_DIR / "per_seed_improvement.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nAll figures saved to: {OUT_DIR}")
print("\nFigure list:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
