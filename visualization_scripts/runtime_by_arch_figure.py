"""
runtime_by_arch_figure.py  —  Section 4.1 runtime overhead figure

Strip-chart: RunTime / Gold Model (log scale) for every unlearning method
across all available GNN architectures, using the 5% forget set.
One subplot per dataset (Cora, Citeseer, Coauthor, Pubmed).
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import NullFormatter

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer", "Coauthor", "Pubmed"]
ARCHS    = ["GCN", "GIN", "GAT", "GraphSAGE", "SGC", "SGCCGU"]
PCT      = 5
SKIP     = {"Identity", "Gold Model"}

# Focus methods (alphabetical) — consistent colours with intro figure
FOCUS = ["CEU", "IDEA", "SCRUB", "SSD", "SalUn"]
OTHER = sorted([
    "AdvancedNegGrad", "BadTeaching", "CGU", "FT", "NegGrad",
])
ALL_METHODS = FOCUS + OTHER          # display / offset order

# Colours: keep focus colours from intro figure; tab20 for the rest
_extra_c = plt.cm.tab20(np.linspace(0.0, 0.95, len(OTHER)))
UNL_COLOR = {
    "CEU"   : "#76b7b2",
    "IDEA"  : "#b07aa1",
    "SCRUB" : "#4e79a7",
    "SSD"   : "#f28e2b",
    "SalUn" : "#59a14f",
    **{m: c for m, c in zip(OTHER, _extra_c)},
}
UNL_MARKER = {
    "CEU"                  : "X",
    "IDEA"                 : "P",
    "SCRUB"                : "o",
    "SSD"                  : "^",
    "SalUn"                : "v",
    "AdvancedNegGrad"      : "s",
    "BadTeaching"          : "D",
    "Cascade"              : "8",
    "CGU"                  : "h",
    "FT"                   : ">",
    "FisherForgetting"     : "p",
    "NegGrad"              : "<",
    "SuccessiveRandomLabels": "*",
    "cfk"                  : "H",
    "eu_k"                 : "d",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    return json.loads(content if content.startswith("[")
                      else "[" + content.rstrip(",") + "]")


def label_unlearner(r):
    u = r["unlearner"]
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    classes = [s.get("class", "") for s in (sub or [])]
    if u == "Cascade":
        return "SalUn" if any("Saliency" in c for c in classes) else "Cascade"
    ltl = r.get("parameters", {}).get("last_trainable_layers", -1)
    return {
        "Identity"                   : "Identity",
        "GoldModelGraph"             : "Gold Model",
        "Finetuning"                 : "FT" if ltl == -1 else "cfk",
        "Scrub"                      : "SCRUB",
        "SelectiveSynapticDampening" : "SSD",
        "IDEA"                       : "IDEA",
        "CGU_edge"                   : "CGU",
        "CEU"                        : "CEU",
    }.get(u, u)


def load_ratios(dataset, arch, pct):
    path = os.path.join(INPUT_DIR, f"{dataset}_{arch}_{pct}.json")
    if not os.path.exists(path):
        return None
    records  = load_json(path)
    labelled = {label_unlearner(r): r.get("RunTime") for r in records}
    gold_rt  = labelled.get("Gold Model")
    if not gold_rt:
        return None
    return {
        m: labelled[m] / gold_rt
        for m in labelled
        if m not in SKIP and labelled.get(m) and labelled[m] > 0
    }


# ── load data ─────────────────────────────────────────────────────────────────
data = {}
for ds in DATASETS:
    data[ds] = {}
    for arch in ARCHS:
        ratios = load_ratios(ds, arch, PCT)
        if ratios is not None:
            data[ds][arch] = ratios

# ── plot ──────────────────────────────────────────────────────────────────────
FS   = 13
FS_S = 11

YTICKS  = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
YLABELS = ["0.1×", "0.5×", "1×", "2×", "5×", "10×", "20×", "50×", "100×"]
YLIM    = (0.05, 130)

rc = {
    "font.family"      : "serif",
    "font.serif"       : ["Times New Roman", "DejaVu Serif"],
    "font.size"        : FS,
    "axes.labelsize"   : FS,
    "axes.titlesize"   : FS,
    "xtick.labelsize"  : FS_S,
    "ytick.labelsize"  : FS,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "figure.dpi"       : 600,
}

# Width ratios proportional to arch count per dataset
# Even offsets for each method within one arch position (span = 0.75)
SPAN    = 0.75
offsets = np.linspace(-SPAN / 2, SPAN / 2, len(ALL_METHODS))
M_OFF   = {m: offsets[i] for i, m in enumerate(ALL_METHODS)}

with plt.rc_context(rc):
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharey=True,
    )
    axes_flat = axes.flatten()

    for di, (ds, ax) in enumerate(zip(DATASETS, axes_flat)):
        archs_avail = [a for a in ARCHS if a in data[ds]]
        arch_x      = {a: i for i, a in enumerate(archs_avail)}

        for method in ALL_METHODS:
            xs, ys = [], []
            for arch in archs_avail:
                val = data[ds][arch].get(method)
                if val is not None:
                    xs.append(arch_x[arch] + M_OFF[method])
                    ys.append(val)
            if not xs:
                continue
            is_focus = method in set(FOCUS)
            ax.scatter(
                xs, ys,
                color=UNL_COLOR.get(method, "gray"),
                marker=UNL_MARKER.get(method, "o"),
                s=50, zorder=4,
                linewidths=0.6,
                edgecolors="black" if is_focus else "none",
            )

        # Vertical separators between architecture groups
        for xi in range(len(archs_avail) - 1):
            ax.axvline(xi + 0.5, color="#cccccc", linewidth=0.8,
                       linestyle="-", zorder=1)

        # Gold Model reference line
        ax.axhline(1.0, color="#c0392b", linewidth=1.5, linestyle="--", zorder=5)
        # Shaded "slower than retraining" region
        ax.axhspan(1.0, YLIM[1], color="#c0392b", alpha=0.04, zorder=0)

        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.set_xticks(range(len(archs_avail)))
        ax.set_xticklabels(archs_avail, rotation=35, ha="right")
        ax.set_xlim(-0.65, len(archs_avail) - 0.35)
        ax.set_title(ds, pad=6)
        ax.grid(axis="y", linestyle=":", alpha=0.3, zorder=0)

        ax.set_yticks(YTICKS)
        if di % 2 == 0:   # left column
            ax.set_ylabel("RunTime / Gold Model (log scale)")
            ax.set_yticklabels(YLABELS)
            if di == 0:
                ax.text(0.02, 0.82, "slower than\nretraining",
                        transform=ax.transAxes, fontsize=FS_S - 1,
                        color="#c0392b", va="center", style="italic")
        else:
            ax.set_yticklabels([])
            ax.tick_params(labelleft=False)

    # ── legend ────────────────────────────────────────────────────────────────
    gold_h = mlines.Line2D([], [], color="#c0392b", linestyle="--",
                            linewidth=1.5, label="Gold Model")
    focus_handles = [
        mlines.Line2D([], [], color=UNL_COLOR[m], marker=UNL_MARKER[m],
                      markersize=7, linestyle="None",
                      markeredgecolor="black", markeredgewidth=0.6, label=m)
        for m in FOCUS
    ]
    other_handles = [
        mlines.Line2D([], [], color=UNL_COLOR[m], marker=UNL_MARKER[m],
                      markersize=7, linestyle="None", label=m)
        for m in OTHER
    ]

    n_handles = 1 + len(FOCUS) + len(OTHER)
    fig.legend(
        handles=[gold_h] + focus_handles + other_handles,
        loc="lower center",
        ncol=n_handles,          # single row
        fontsize=FS_S,
        frameon=True, framealpha=0.9, edgecolor="#ccc",
        handlelength=1.4,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.06, hspace=0.45, bottom=0.14)

    for fmt in ("png", "pdf"):
        kw = {"bbox_inches": "tight"}
        if fmt == "png":
            kw["dpi"] = 600
        out = os.path.join(OUTPUT_DIR, f"runtime_by_arch.{fmt}")
        plt.savefig(out, **kw)
        print(f"Saved {out}")

    plt.close()

print("Done.")
