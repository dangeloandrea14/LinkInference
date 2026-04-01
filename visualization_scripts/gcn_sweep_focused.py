"""
gcn_sweep_focused.py

Produces one NeurIPS-ready figure:
  output/viz/LinkAttack/edge/gcn_sweep_focused.{png,pdf}

GCN only — Test Accuracy (original graph) and UMIA vs. forget % for a
focused subset of unlearners (SCRUB, SSD, SalUn, IDEA, CEU) plus the
Identity and Gold Model baselines.

Cora and Citeseer are shown together in the same subplots:
  solid lines = Cora, dashed lines = Citeseer.
"""

import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer"]
PCTS     = [5, 20, 50, 99, 100]

FOCUS_UNLEARNERS = ["Identity", "Gold Model", "SCRUB", "SSD", "SalUn", "IDEA", "CEU"]
BOLD_BASELINES   = {"Identity", "Gold Model"}

PLOT_METRICS = [
    ("Acc (test, orig)", "Test Accuracy (original graph)"),
    ("UMIA",             "UMIA"),
]

METRICS_KEYS = {
    "Acc (test, orig)": "sklearn.metrics.accuracy_score.test.unlearned.on_graph:False",
    "UMIA"            : "UMIA",
}

# Colour per unlearner — hand-picked for readability and B&W friendliness
UNLEARNER_COLORS = {
    "Identity"  : "#888888",
    "Gold Model": "#c0392b",
    "SCRUB"     : "#4e79a7",
    "SSD"       : "#f28e2b",
    "SalUn"     : "#59a14f",
    "IDEA"      : "#b07aa1",
    "CEU"       : "#e15759",
}

MARKERS = {
    "Identity"  : "s",
    "Gold Model": "D",
    "SCRUB"     : "o",
    "SSD"       : "^",
    "SalUn"     : "v",
    "IDEA"      : "P",
    "CEU"       : "X",
}

DATASET_LS = {
    "Cora"    : "-",
    "Citeseer": "--",
}

# ── label helpers ─────────────────────────────────────────────────────────────
def _cascade_label(r):
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    classes = [s.get("class", "") for s in (sub or [])]
    if any("Saliency" in c for c in classes):
        return "SalUn"
    if any("UNSIR" in c for c in classes):
        return "UNSIR→FT"
    return "Cascade"

def label_unlearner(r):
    u = r["unlearner"]
    if u == "Cascade":
        return _cascade_label(r)
    ltl = r.get("parameters", {}).get("last_trainable_layers", -1)
    return {
        "Identity"               : "Identity",
        "GoldModelGraph"         : "Gold Model",
        "Finetuning"             : "FT" if ltl == -1 else "cfk",
        "Scrub"                  : "SCRUB",
        "SelectiveSynapticDampening": "SSD",
        "IDEA"                   : "IDEA",
        "CGU_edge"               : "CGU",
        "CEU"                    : "CEU",
    }.get(u, u)


# ── data loading ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return json.loads("[" + content.rstrip(",") + "]")


def load_data():
    rows = []
    for dataset in DATASETS:
        for pct in PCTS:
            path = os.path.join(INPUT_DIR, f"{dataset}_GCN_{pct}.json")
            if not os.path.exists(path):
                continue
            records = load_json(path)
            for r in records:
                unl = label_unlearner(r)
                if unl not in FOCUS_UNLEARNERS:
                    continue
                row = {"dataset": dataset, "forget_pct": pct, "unlearner": unl}
                for short, key in METRICS_KEYS.items():
                    row[short] = r.get(key, np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────
def plot(df):
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
        "axes.grid"        : True,
        "grid.alpha"       : 0.25,
        "grid.linestyle"   : "--",
        "grid.linewidth"   : 0.6,
        "figure.dpi"       : 600,
    }

    x_pos    = list(range(len(PCTS)))
    pct_to_x = {p: i for i, p in enumerate(PCTS)}
    x_labels = [f"{p}%" for p in PCTS]

    # draw order: non-baselines first (behind), baselines on top
    draw_order = [u for u in FOCUS_UNLEARNERS if u not in BOLD_BASELINES] + \
                 [u for u in FOCUS_UNLEARNERS if u in BOLD_BASELINES]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(11, 4.5),
                                 squeeze=False)
        fig.suptitle(
            "GCN — Edge Removal Unlearning: Cora & Citeseer",
            fontsize=11, fontweight="bold", y=1.02,
        )

        for mi, (metric, col_title) in enumerate(PLOT_METRICS):
            ax = axes[0][mi]

            for dataset in DATASETS:
                dsub = df[df["dataset"] == dataset]
                ls   = DATASET_LS[dataset]

                for unl in draw_order:
                    usub = dsub[dsub["unlearner"] == unl].sort_values("forget_pct")
                    if usub.empty or usub[metric].isna().all():
                        continue
                    xs      = [pct_to_x[p] for p in usub["forget_pct"]]
                    is_bold = unl in BOLD_BASELINES
                    ax.plot(
                        xs, usub[metric],
                        color=UNLEARNER_COLORS[unl],
                        linestyle=ls,
                        marker=MARKERS[unl],
                        linewidth=2.4 if is_bold else 1.3,
                        markersize=6  if is_bold else 4,
                        markeredgewidth=0.6,
                        markeredgecolor="white",
                        alpha=1.0 if is_bold else 0.65,
                        zorder=10 if is_bold else 2,
                    )

            if metric == "UMIA":
                ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.9,
                           zorder=0)
                ax.text(0.02, 0.52, "random (0.5)", transform=ax.transAxes,
                        fontsize=7, color="#888", va="bottom", style="italic")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Forget set size")
            ax.set_title(col_title, fontsize=10, pad=5)

            yvals = df[metric].dropna()
            if not yvals.empty:
                pad = (yvals.max() - yvals.min()) * 0.12 or 0.02
                ax.set_ylim(yvals.min() - pad, yvals.max() + pad)

        # ── legend ────────────────────────────────────────────────────────────
        # Row 1: unlearner colours
        unl_handles = [
            plt.Line2D([0], [0],
                       color=UNLEARNER_COLORS[u],
                       marker=MARKERS[u],
                       linestyle="-",
                       linewidth=2.2 if u in BOLD_BASELINES else 1.2,
                       markersize=5 if u in BOLD_BASELINES else 4,
                       markeredgewidth=0.4, markeredgecolor="white",
                       label=u)
            for u in FOCUS_UNLEARNERS
        ]
        # Row 2: dataset line styles
        ds_handles = [
            mlines.Line2D([0], [0], color="black",
                          linestyle=DATASET_LS[d], linewidth=1.4, label=d)
            for d in DATASETS
        ]
        fig.legend(
            handles=unl_handles + ds_handles,
            loc="lower center",
            ncol=len(FOCUS_UNLEARNERS) + len(DATASETS),
            fontsize=8.5,
            frameon=True, framealpha=0.9, edgecolor="#ccc",
            bbox_to_anchor=(0.5, -0.10),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"gcn_sweep_focused.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows")
    plot(df)
    print("Done.")
