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
from matplotlib.transforms import blended_transform_factory as btf

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer", "Pubmed"]
ARCHS    = ["GCN", "GIN", "GAT", "GraphSAGE", "SGCCGU"]
PCT      = 5
SKIP     = {"Identity"}

# Methods — consistent colours with intro figure
FOCUS = ["CEU", "SalUn", "GNNDelete", "IDEA", "SSD", "CGU", "ScaleGUN"]
OTHER = []
ALL_METHODS = FOCUS

UNL_COLOR = {
    "Gold Model": "#c0392b",
    "CEU"       : "#76b7b2",
    "IDEA"      : "#b07aa1",
    "SSD"       : "#f28e2b",
    "SalUn"     : "#59a14f",
    "GNNDelete" : "#e15759",
    "CGU"       : "#edc948",
    "ScaleGUN"  : "#9c755f",
}
UNL_MARKER = {
    "Gold Model": "D",
    "CEU"       : "X",
    "IDEA"      : "P",
    "SSD"       : "^",
    "SalUn"     : "v",
    "GNNDelete" : "s",
    "CGU"       : "p",
    "ScaleGUN"  : "*",
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
        "GNNDelete"                  : "GNNDelete",
        "ScaleGUN"                   : "ScaleGUN",
    }.get(u, u)


def load_runtimes(dataset, arch, pct):
    """Returns dict of method -> absolute RunTime in seconds, or None."""
    path = os.path.join(INPUT_DIR, f"{dataset}_{arch}_{pct}.json")
    if not os.path.exists(path):
        return None
    records  = load_json(path)
    labelled = {label_unlearner(r): r.get("RunTime") for r in records}
    if not labelled.get("Gold Model"):
        return None
    return {
        m: labelled[m]
        for m in labelled
        if m not in SKIP and labelled.get(m) and labelled[m] > 0
    }


# ── load data ─────────────────────────────────────────────────────────────────
data = {}
for ds in DATASETS:
    data[ds] = {}
    for arch in ARCHS:
        runtimes = load_runtimes(ds, arch, PCT)
        if runtimes is not None:
            data[ds][arch] = runtimes

# ── plot ──────────────────────────────────────────────────────────────────────
FS   = 13
FS_S = 11

YTICKS  = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
YLIM    = (0.05, 300)

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
        1, 3, figsize=(14, 4.5), sharey=True,
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

        # Gold Model: red hue + dashed line per architecture
        for arch in archs_avail:
            gm_rt = data[ds][arch].get("Gold Model")
            if gm_rt is not None:
                xi = arch_x[arch]
                ax.fill_between([xi - 0.45, xi + 0.45], gm_rt, YLIM[1],
                                color="#c0392b", alpha=0.07, zorder=0)
                ax.hlines(gm_rt, xi - 0.45, xi + 0.45,
                          colors="#c0392b", linewidth=1.4,
                          linestyle="--", zorder=5)

        # Vertical separators between architecture groups
        for xi in range(len(archs_avail) - 1):
            ax.axvline(xi + 0.5, color="#cccccc", linewidth=0.8,
                       linestyle="-", zorder=1)


        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.set_xticks(range(len(archs_avail)))
        ax.set_xticklabels(archs_avail, rotation=35, ha="right")
        ax.set_xlim(-0.65, len(archs_avail) - 0.35)
        ax.set_title(ds, pad=6)

        # Highlight linear-model architecture
        if "SGCCGU" in arch_x:
            xi = arch_x["SGCCGU"]
            ax.axvspan(xi - 0.5, xi + 0.5, color="#e8e8e8", zorder=0, alpha=0.6)
            ax.text(xi, 1.0, "linear", transform=btf(ax.transData, ax.transAxes),
                    fontsize=FS_S - 2, color="#666", style="italic",
                    ha="center", va="bottom")
        ax.grid(axis="y", linestyle=":", alpha=0.3, zorder=0)

        if di == 0:
            ax.set_ylabel("RunTime (seconds, log scale)")
        else:
            ax.tick_params(labelleft=False)

    # Set y-ticks once on the shared axis after the loop
    tick_labels = [f"{s:.2f}s" if s < 1 else f"{s:.1f}s" if s < 10 else f"{s:.0f}s"
                   for s in YTICKS]
    axes_flat[0].set_yticks(YTICKS)
    axes_flat[0].set_yticklabels(tick_labels)

    # ── legend ────────────────────────────────────────────────────────────────
    gold_h = mlines.Line2D([], [], color="#c0392b", linestyle="--",
                            linewidth=2.0, label="Gold Model")
    focus_handles = [
        mlines.Line2D([], [], color=UNL_COLOR[m], marker=UNL_MARKER[m],
                      markersize=7, linestyle="None",
                      markeredgecolor="black", markeredgewidth=0.6, label=m)
        for m in FOCUS
    ]

    n_handles = 1 + len(FOCUS)
    fig.legend(
        handles=[gold_h] + focus_handles,
        loc="lower center",
        ncol=n_handles,          # single row
        fontsize=FS_S,
        frameon=True, framealpha=0.9, edgecolor="#ccc",
        handlelength=1.4,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.06, bottom=0.22)

    for fmt in ("png", "pdf"):
        kw = {"bbox_inches": "tight"}
        if fmt == "png":
            kw["dpi"] = 600
        out = os.path.join(OUTPUT_DIR, f"runtime_by_arch.{fmt}")
        plt.savefig(out, **kw)
        print(f"Saved {out}")

    plt.close()

print("Done.")
