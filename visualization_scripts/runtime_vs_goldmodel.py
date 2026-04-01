"""
runtime_vs_goldmodel.py

Produces one NeurIPS-ready figure:
  output/viz/LinkAttack/edge/runtime_vs_goldmodel.{png,pdf}

For each dataset (Cora, Citeseer) — one subplot each — shows the RunTime of
six focused unlearners (SCRUB, SSD, SalUn, IDEA, CGU, CEU) relative to the
Gold Model (retrain from scratch) cost.  Y-axis is the ratio

    RunTime(unlearner) / RunTime(Gold Model)

on a log scale.  The dashed line at ratio = 1 marks the Gold Model cost.
Anything above it is *more expensive than just retraining*.
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, NullFormatter

# ── paths ───────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer"]
ARCHS    = ["GAT", "GCN", "GraphSAGE", "SGC", "SGCCGU"]
FOCUS    = ["SCRUB", "SSD", "SalUn", "IDEA", "CGU", "CEU"]

ARCH_COLORS = {
    "GAT"      : "#4e79a7",
    "GCN"      : "#f28e2b",
    "GraphSAGE": "#59a14f",
    "SGC"      : "#e15759",
    "SGCCGU"   : "#b07aa1",
}

# ── unlearner label helpers ──────────────────────────────────────────────────
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


# ── data loading ─────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    return json.loads("[" + content.rstrip(",") + "]")


def load_ratios():
    """
    Returns dict: ratios[dataset][arch][unlearner] = RunTime / GoldModel RunTime
    Missing values (NaN RunTime or missing Gold Model) are stored as np.nan.
    """
    ratios = {}
    for dataset in DATASETS:
        ratios[dataset] = {}
        for path in sorted(glob.glob(
                os.path.join(INPUT_DIR, f"{dataset}_*_20.json"))):
            arch = (os.path.basename(path)
                    .replace(f"{dataset}_", "")
                    .replace("_20.json", ""))
            if arch not in ARCHS:
                continue

            records = load_json(path)
            labelled = {label_unlearner(r): r.get("RunTime") for r in records}

            gold_rt = labelled.get("Gold Model")
            if not gold_rt:
                continue

            ratios[dataset][arch] = {}
            for unl in FOCUS:
                rt = labelled.get(unl)
                ratios[dataset][arch][unl] = (rt / gold_rt
                                              if (rt and rt > 0) else np.nan)
    return ratios


# ── plotting ─────────────────────────────────────────────────────────────────
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

    n_unl   = len(FOCUS)
    n_arch  = len(ARCHS)
    width   = 0.14
    x       = np.arange(n_unl)
    offsets = np.linspace(-(n_arch - 1) / 2 * width,
                           (n_arch - 1) / 2 * width,
                           n_arch)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(DATASETS),
                                 figsize=(13, 5),
                                 squeeze=False)
        fig.suptitle(
            "RunTime relative to Gold Model (retrain from scratch) — 20% forget set",
            fontsize=11, fontweight="bold", y=1.02,
        )

        for di, dataset in enumerate(DATASETS):
            ax = axes[0][di]
            ds_ratios = ratios[dataset]

            for ai, arch in enumerate(ARCHS):
                arch_data = ds_ratios.get(arch, {})
                vals = [arch_data.get(unl, np.nan) for unl in FOCUS]

                for xi, val in enumerate(vals):
                    if np.isnan(val):
                        continue
                    color    = ARCH_COLORS[arch]
                    is_above = val >= 1.0
                    ax.bar(
                        x[xi] + offsets[ai], val, width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=1.0,
                        zorder=3,
                    )
                    # annotate extreme values (ratio > 30×)
                    if val > 30:
                        ax.text(
                            x[xi] + offsets[ai], val * 1.08,
                            f"{val:.0f}×",
                            ha="center", va="bottom",
                            fontsize=6.5, color=color, fontweight="bold",
                        )

            # Gold Model reference line and shading
            ax.axhline(1.0, color="#c0392b", linewidth=1.4,
                       linestyle="--", zorder=4,
                       label="Gold Model (retrain from scratch)")
            ax.axhspan(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 200,
                       color="#c0392b", alpha=0.04, zorder=0)

            # axes formatting
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.05)

            # re-draw shade after setting log scale
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
            ax.set_title(dataset, fontsize=11, fontweight="bold")
            ax.grid(axis="y", which="major", linestyle=":", alpha=0.4, zorder=0)
            ax.grid(axis="y", which="minor", linestyle=":", alpha=0.15, zorder=0)

            ax.text(
                0.01, 0.70,
                "more expensive\nthan retraining",
                transform=ax.transAxes, fontsize=7.5,
                color="#c0392b", style="italic", va="center",
            )

        # shared legend for architectures
        arch_handles = [
            mpatches.Patch(facecolor=ARCH_COLORS[a], edgecolor="black",
                           linewidth=0.5, label=a)
            for a in ARCHS
        ]
        gold_handle = plt.Line2D(
            [0], [0], color="#c0392b", linewidth=1.4,
            linestyle="--", label="Gold Model",
        )
        fig.legend(
            handles=arch_handles + [gold_handle],
            loc="lower center", ncol=len(ARCHS) + 1,
            fontsize=8.5, frameon=True, framealpha=0.9,
            edgecolor="#ccc", bbox_to_anchor=(0.5, -0.08),
        )

        plt.tight_layout()

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"runtime_vs_goldmodel.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ratios = load_ratios()
    plot(ratios)
    print("Done.")
