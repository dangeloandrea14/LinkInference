"""
runtime_vs_goldmodel_gcn.py

Produces one NeurIPS-ready figure:
  output/viz/LinkAttack/edge/runtime_vs_goldmodel_gcn.{png,pdf}

GCN only — shows RunTime of six focused unlearners (SCRUB, SSD, SalUn,
IDEA, CGU, CEU) relative to the Gold Model (retrain from scratch) cost,
for both Cora and Citeseer in a single grouped-bar plot.

Y-axis: RunTime(unlearner) / RunTime(Gold Model), log scale.
Two bars per unlearner group: Cora (left) and Citeseer (right).
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, NullFormatter

# ── paths ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer"]
FOCUS    = ["SCRUB", "SSD", "SalUn", "IDEA", "CGU", "CEU"]

DATASET_COLORS = {
    "Cora"    : "#4e79a7",
    "Citeseer": "#f28e2b",
}

# ── label helpers ─────────────────────────────────────────────────────────────
def _cascade_label(r):
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    classes = [s.get("class", "") for s in (sub or [])]
    if any("Saliency" in c for c in classes):
        return "SalUn"
    return "Cascade"

def label_unlearner(r):
    u = r["unlearner"]
    if u == "Cascade":
        return _cascade_label(r)
    return {
        "GoldModelGraph"             : "Gold Model",
        "Scrub"                      : "SCRUB",
        "SelectiveSynapticDampening" : "SSD",
        "IDEA"                       : "IDEA",
        "CGU_edge"                   : "CGU",
        "CEU"                        : "CEU",
    }.get(u, u)


# ── data loading ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    return json.loads("[" + content.rstrip(",") + "]")


def load_ratios():
    """
    Returns dict: ratios[dataset][unlearner] = RunTime / GoldModel RunTime
    Missing values are stored as np.nan.
    """
    ratios = {}
    for dataset in DATASETS:
        path = os.path.join(INPUT_DIR, f"{dataset}_GCN_20.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue

        records = load_json(path)
        labelled = {label_unlearner(r): r.get("RunTime") for r in records}

        gold_rt = labelled.get("Gold Model")
        if not gold_rt:
            print(f"Warning: no Gold Model entry in {path}, skipping.")
            continue

        ratios[dataset] = {}
        for unl in FOCUS:
            rt = labelled.get(unl)
            ratios[dataset][unl] = (rt / gold_rt if (rt and rt > 0) else np.nan)

    return ratios


# ── plotting ──────────────────────────────────────────────────────────────────
def plot(ratios):
    rc = {
        "font.family"      : "serif",
        "font.serif"       : ["Times New Roman", "DejaVu Serif"],
        "font.size"        : 10,
        "axes.labelsize"   : 10,
        "axes.titlesize"   : 11,
        "xtick.labelsize"  : 9,
        "ytick.labelsize"  : 9,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.linewidth"   : 0.8,
        "figure.dpi"       : 600,
    }

    n_unl    = len(FOCUS)
    n_ds     = len(DATASETS)
    width    = 0.28
    x        = np.arange(n_unl)
    offsets  = np.array([-0.5, 0.5]) * width   # left=Cora, right=Citeseer

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(
            "GCN — RunTime relative to Gold Model (retrain from scratch) — 20% forget set",
            fontsize=11, fontweight="bold", y=1.02,
        )

        for di, dataset in enumerate(DATASETS):
            ds_ratios = ratios.get(dataset, {})
            vals      = [ds_ratios.get(unl, np.nan) for unl in FOCUS]
            color     = DATASET_COLORS[dataset]

            for xi, val in enumerate(vals):
                if np.isnan(val):
                    continue
                ax.bar(
                    x[xi] + offsets[di], val, width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=1.0,
                    zorder=3,
                    label=dataset if xi == 0 else "_nolegend_",
                )
                # annotate extreme values (ratio > 30×)
                if val > 30:
                    ax.text(
                        x[xi] + offsets[di], val * 1.08,
                        f"{val:.0f}×",
                        ha="center", va="bottom",
                        fontsize=7, color=color, fontweight="bold",
                    )

        # Gold Model reference line
        ax.axhline(1.0, color="#c0392b", linewidth=1.4,
                   linestyle="--", zorder=4)

        # Shade above the reference line
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.05)
        ax.axhspan(1.0, ax.get_ylim()[1],
                   color="#c0392b", alpha=0.05, zorder=0)

        ax.yaxis.set_major_locator(
            LogLocator(base=10, subs=[1.0], numticks=10))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])
        ax.set_yticklabels(
            ["0.1×", "0.5×", "1×", "2×", "5×",
             "10×", "20×", "50×", "100×", "200×"])
        ax.set_ylabel("RunTime / Gold Model RunTime (log scale)")
        ax.set_xticks(x)
        ax.set_xticklabels(FOCUS, fontsize=9)
        ax.grid(axis="y", which="major", linestyle=":", alpha=0.4, zorder=0)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.15, zorder=0)

        ax.text(
            0.01, 0.72,
            "more expensive\nthan retraining",
            transform=ax.transAxes, fontsize=7.5,
            color="#c0392b", style="italic", va="center",
        )

        # Legend: datasets + gold model reference
        ds_handles = [
            mpatches.Patch(facecolor=DATASET_COLORS[d], edgecolor="black",
                           linewidth=0.5, label=d)
            for d in DATASETS
        ]
        gold_handle = plt.Line2D(
            [0], [0], color="#c0392b", linewidth=1.4,
            linestyle="--", label="Gold Model",
        )
        ax.legend(
            handles=ds_handles + [gold_handle],
            fontsize=8.5, frameon=True, framealpha=0.9,
            edgecolor="#ccc", loc="upper right",
        )

        plt.tight_layout()

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"runtime_vs_goldmodel_gcn.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ratios = load_ratios()
    plot(ratios)
    print("Done.")
