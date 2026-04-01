"""
barplot_per_arch.py

For each dataset (Cora, Citeseer) produces one figure saved as:
  output/viz/LinkAttack/edge/<dataset>_barplot_per_arch.{png,pdf}

Each figure is a grid of subplots — one per architecture (20% forget set).
Every subplot is a grouped barplot:
  X axis  : unlearner methods (using canonical short labels)
  3 bars  : RunTime (normalised to [0,1]), UMIA, Accuracy (forget, on_graph:True)

Special handling (mirrors visualize.ipynb):
  - Identity / GoldModel RunTime is set to 10× the Finetuning runtime so
    trivially-fast or retrain-from-scratch baselines don't compress the scale.
  - RunTime is then normalised by the per-arch maximum.
  - GoldModel reference lines are drawn for UMIA and Accuracy.
  - Identity / GoldModel bars are highlighted with a gold background.
  - Unlearners with NaN/zero UMIA are marked with a red ✕.
"""

import json, glob, math, os
import numpy as np
import pandas as pd
from io import StringIO
from pandas import json_normalize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from string import ascii_lowercase

# ── paths ──────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── unlearner label mapping (matches visualize_linkattack.py) ──────────────
UNLEARNER_LABELS = {
    ("Identity",               -1): "Identity",
    ("GoldModelGraph",         -1): "Gold Model",
    ("Finetuning",             -1): "FT",
    ("SuccessiveRandomLabels", -1): "SRL",
    ("Finetuning",              2): "cfk",
    ("eu_k",                   -1): "eu_k",
    ("NegGrad",                -1): "NegGrad",
    ("AdvancedNegGrad",        -1): "Adv.NegGrad",
    ("Cascade_UNSIR",          -1): "UNSIR→FT",
    ("BadTeaching",            -1): "BadTeaching",
    ("Scrub",                  -1): "SCRUB",
    ("FisherForgetting",       -1): "Fisher",
    ("SelectiveSynapticDampening", -1): "SSD",
    ("Cascade_SalUn",          -1): "SalUn",
    ("IDEA",                   -1): "IDEA",
    ("CGU_edge",               -1): "CGU",
    ("CEU",                    -1): "CEU",
}

UNLEARNER_ORDER = [
    "Identity", "Gold Model", "FT", "SRL", "cfk", "eu_k",
    "NegGrad", "Adv.NegGrad", "UNSIR→FT", "BadTeaching", "SCRUB",
    "Fisher", "SSD", "SalUn", "IDEA", "CGU", "CEU",
]


def label_unlearner(r):
    u   = r["unlearner"]
    ltl = r.get("parameters", {}).get("last_trainable_layers", -1)
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    if u == "Cascade":
        classes = [s.get("class", "") for s in (sub or [])]
        if any("UNSIR" in c for c in classes):
            return "UNSIR→FT"
        if any("Saliency" in c for c in classes):
            return "SalUn"
    key = (u, ltl if u == "Finetuning" else -1)
    return UNLEARNER_LABELS.get(key, u)


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if not content.startswith("["):
        content = "[" + content
    if content.endswith(","):
        content = content[:-1]
    if not content.endswith("]"):
        content = content + "]"
    return json.loads(content)


def load_dataset(dataset, forget_pct=20):
    """Load all architecture files for a dataset at a given forget percentage."""
    pattern = os.path.join(INPUT_DIR, f"{dataset}_*_{forget_pct}.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        basename = os.path.splitext(os.path.basename(path))[0]
        parts = basename.split("_")
        arch = "_".join(parts[1:-1])          # e.g. "GCN", "GraphSAGE", "SGCCGU"

        for r in load_json(path):
            rows.append({
                "unlearner" : label_unlearner(r),
                "arch"      : arch,
                "RunTime"   : r.get("RunTime", np.nan),
                "UMIA"      : r.get("UMIA", np.nan),
                "Accuracy"  : r.get(
                    "sklearn.metrics.accuracy_score.forget.unlearned.on_graph:True",
                    np.nan,
                ),
            })
    return pd.DataFrame(rows)


# ── plot constants ─────────────────────────────────────────────────────────
METRICS       = ["RunTime", "UMIA", "Accuracy"]
DISPLAY_NAME  = {"RunTime": "RunTime (norm.)", "UMIA": "UMIA", "Accuracy": "Accuracy"}
COLOR         = {"RunTime": "#1b9e77", "UMIA": "#d95f02", "Accuracy": "#7570b3"}
HATCH         = {"RunTime": "//",      "UMIA": "\\\\",    "Accuracy": ".."}

HEAD = ["Identity", "Gold Model"]
TAIL = ["IDEA", "CGU", "CEU"]


def plot_dataset(dataset):
    df = load_dataset(dataset, forget_pct=20)
    if df.empty:
        print(f"No data found for {dataset}")
        return

    architectures = sorted(df["arch"].unique())
    n_arch = len(architectures)
    ncols  = 2
    nrows  = math.ceil(n_arch / ncols)

    plt.rcParams.update({
        "font.family"       : "serif",
        "font.serif"        : ["Times New Roman", "DejaVu Serif"],
        "font.size"         : 10,
        "axes.titlesize"    : 11,
        "axes.labelsize"    : 10,
        "xtick.labelsize"   : 8,
        "ytick.labelsize"   : 9,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.linewidth"    : 0.8,
        "figure.dpi"        : 600,
    })

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle(f"{dataset} — 20% forget set (GCN architecture grid)",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, arch in enumerate(architectures):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df[df["arch"] == arch].copy()

        # aggregate (mean across duplicate rows, if any)
        agg = sub.groupby("unlearner")[METRICS].mean()

        # Identity does nothing (pure inference), so its RunTime is meaningless
        # as an unlearning cost — pin it to 10× FT as a display convention.
        # Gold Model's RunTime is the real measured retrain time; keep it as-is.
        if "FT" in agg.index and agg.loc["FT", "RunTime"] > 0:
            base = agg.loc["FT", "RunTime"]
            if "Identity" in agg.index:
                agg.loc["Identity", "RunTime"] = base * 10

        # normalise runtime to [0, 1]
        max_rt = agg["RunTime"].max()
        if max_rt > 0:
            agg["RunTime"] /= max_rt

        # canonical ordering
        present       = list(agg.index)
        head_present  = [u for u in HEAD if u in present]
        tail_present  = [u for u in TAIL if u in present]
        middle        = [u for u in UNLEARNER_ORDER
                         if u in present and u not in head_present + tail_present]
        ordered       = head_present + middle + tail_present
        agg           = agg.reindex(ordered)

        x      = np.arange(len(ordered))
        width  = 0.25
        offsets = np.linspace(-width, width, len(METRICS))

        # gold background for baseline unlearners
        if head_present:
            ax.axvspan(-0.5, len(head_present) - 0.5,
                       color="#FFD700", alpha=0.18, zorder=0)

        # vertical separators
        if head_present and len(head_present) < len(ordered):
            ax.axvline(len(head_present) - 0.5, color="black",
                       linestyle="-", linewidth=0.8)
        if tail_present:
            sep = len(head_present) + len(middle) - 0.5
            ax.axvline(sep, color="black", linestyle="-", linewidth=0.8)

        # UMIA validity mask
        umia_vals  = agg["UMIA"].values
        valid_mask = ~(np.isnan(umia_vals) | (umia_vals <= 0))

        # red ✕ for invalid entries
        if (~valid_mask).any():
            ax.plot(x[~valid_mask], np.zeros((~valid_mask).sum()) + 0.02,
                    marker="x", linestyle="None", color="red",
                    markersize=10, markeredgewidth=2, zorder=5)

        # bars (only for valid entries)
        for i, metric in enumerate(METRICS):
            vals = agg[metric].values
            ax.bar(
                x[valid_mask] + offsets[i],
                vals[valid_mask],
                width,
                label=DISPLAY_NAME[metric],
                color=COLOR[metric],
                edgecolor="black",
                linewidth=0.5,
                hatch=HATCH[metric],
            )

        # GoldModel reference lines
        if "Gold Model" in agg.index:
            gold = agg.loc["Gold Model"]
            if np.isfinite(gold["UMIA"]):
                ax.axhline(gold["UMIA"],   color=COLOR["UMIA"],
                           linestyle="--", linewidth=1.0, zorder=3)
            if np.isfinite(gold["Accuracy"]):
                ax.axhline(gold["Accuracy"], color=COLOR["Accuracy"],
                           linestyle="--", linewidth=1.0, zorder=3)

        ax.set_title(arch, fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Unlearner")

        # panel label (a), (b), …
        ax.text(-0.08, 1.06, f"({ascii_lowercase[idx]})",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=11)

    # remove empty subplots
    for j in range(len(architectures), nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    # shared legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1,
                       facecolor=COLOR[m], edgecolor="black",
                       linewidth=0.5, hatch=HATCH[m])
        for m in METRICS
    ]
    fig.legend(
        handles=legend_handles,
        labels=[DISPLAY_NAME[m] for m in METRICS],
        loc="lower center",
        ncol=len(METRICS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.0,
        columnspacing=1.4,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    for fmt in ("png", "pdf"):
        kw  = {"bbox_inches": "tight", "transparent": True}
        if fmt == "png":
            kw["dpi"] = 600
        out = os.path.join(OUTPUT_DIR, f"{dataset}_barplot_per_arch.{fmt}")
        plt.savefig(out, **kw)
        print(f"Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for dataset in ("Cora", "Citeseer"):
        print(f"\n── {dataset} ──")
        plot_dataset(dataset)
    print("\nDone.")
