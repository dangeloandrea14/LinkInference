"""
intro_figure_pubmed.py  —  TEMPORARY variant of intro_figure.py

Adds Pubmed as a third dataset alongside Cora and Citeseer.
Output: output/viz/LinkAttack/edge/intro_figure_pubmed.{png,pdf}

Dataset encoding:
  Cora     — filled bar (no hatch),  filled marker,  solid line
  Citeseer — hatched bar (//),       open marker,    solid line
  Pubmed   — hatched bar (xx),       filled marker,  dashed line
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer", "Pubmed"]
PCTS     = [5, 20, 50, 99, 100]

FOCUS          = sorted(["SCRUB", "SSD", "SalUn", "IDEA", "CEU"])
SWEEP_EXTRAS   = ["Gold Model"]
BOLD_BASELINES = {"Gold Model"}

VENUE_LABEL = {
    "CEU"  : "(KDD'23)",
    "IDEA" : "(KDD'24)",
    "SSD"  : "(AAAI'24)",
    "SalUn": "(ICLR'24)",
}

UNL_COLOR = {
    "Gold Model": "#c0392b",
    "CEU"       : "#76b7b2",
    "IDEA"      : "#b07aa1",
    "SCRUB"     : "#4e79a7",
    "SSD"       : "#f28e2b",
    "SalUn"     : "#59a14f",
}
UNL_MARKER = {
    "Gold Model": "D",
    "CEU"       : "X",
    "IDEA"      : "P",
    "SCRUB"     : "o",
    "SSD"       : "^",
    "SalUn"     : "v",
}

# Dataset visual encoding
DS_HATCH     = {"Cora": "",   "Citeseer": "//",  "Pubmed": "xx"}
DS_FILLED    = {"Cora": True, "Citeseer": False,  "Pubmed": True}
DS_LINESTYLE = {"Cora": "-",  "Citeseer": "-",    "Pubmed": "--"}


# ── label helpers ─────────────────────────────────────────────────────────────
def _cascade_label(r):
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    classes = [s.get("class", "") for s in (sub or [])]
    if any("Saliency" in c for c in classes):
        return "SalUn"
    if any("UNSIR"    in c for c in classes):
        return "UNSIR→FT"
    return "Cascade"

def label_unlearner(r):
    u = r["unlearner"]
    if u == "Cascade":
        return _cascade_label(r)
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


# ── data loading ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return json.loads("[" + content.rstrip(",") + "]")


def load_runtime_ratios():
    ratios = {}
    for dataset in DATASETS:
        path = os.path.join(INPUT_DIR, f"{dataset}_GCN_20.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        records  = load_json(path)
        labelled = {label_unlearner(r): r.get("RunTime") for r in records}
        gold_rt  = labelled.get("Gold Model")
        if not gold_rt:
            continue
        ratios[dataset] = {
            unl: (labelled[unl] / gold_rt
                  if labelled.get(unl) and labelled[unl] > 0 else np.nan)
            for unl in FOCUS
        }
    return ratios


def load_sweep_df():
    ACC_KEY = "sklearn.metrics.accuracy_score.test.unlearned.on_graph:True"
    want    = set(FOCUS) | set(SWEEP_EXTRAS)
    rows    = []
    for dataset in DATASETS:
        for pct in PCTS:
            path = os.path.join(INPUT_DIR, f"{dataset}_GCN_{pct}.json")
            if not os.path.exists(path):
                continue
            for r in load_json(path):
                unl = label_unlearner(r)
                if unl not in want:
                    continue
                rows.append({
                    "dataset"         : dataset,
                    "forget_pct"      : pct,
                    "unlearner"       : unl,
                    "Acc (test, orig)": r.get(ACC_KEY, np.nan),  # now on unlearned graph
                })
    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────
def make_figure(ratios, sweep_df):
    FS       = 14
    FS_VENUE = 11

    rc = {
        "font.family"      : "serif",
        "font.serif"       : ["Times New Roman", "DejaVu Serif"],
        "font.size"        : FS,
        "axes.labelsize"   : FS,
        "axes.titlesize"   : FS,
        "xtick.labelsize"  : FS,
        "ytick.labelsize"  : FS,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.linewidth"   : 0.8,
        "figure.dpi"       : 600,
    }

    n_unl   = len(FOCUS)
    n_ds    = len(DATASETS)
    width   = 0.25
    x_bar   = np.arange(n_unl)
    offsets = np.linspace(-(n_ds - 1) / 2, (n_ds - 1) / 2, n_ds) * width

    x_pos    = list(range(len(PCTS)))
    pct_to_x = {p: i for i, p in enumerate(PCTS)}
    x_labels = [f"{p}%" for p in PCTS]

    sweep_order = [u for u in SWEEP_EXTRAS + FOCUS if u not in BOLD_BASELINES] + \
                  [u for u in SWEEP_EXTRAS + FOCUS if u in BOLD_BASELINES]

    with plt.rc_context(rc):
        fig, (ax_rt, ax_sw) = plt.subplots(
            1, 2, figsize=(14, 4.8),
            gridspec_kw={"width_ratios": [1.05, 1]},
        )

        # ── LEFT: runtime bar chart ───────────────────────────────────────────
        for di, dataset in enumerate(DATASETS):
            ds_ratios = ratios.get(dataset, {})
            for xi, unl in enumerate(FOCUS):
                val = ds_ratios.get(unl, np.nan)
                if np.isnan(val):
                    continue
                color = UNL_COLOR[unl]
                ax_rt.bar(
                    x_bar[xi] + offsets[di], val, width,
                    color=color,
                    hatch=DS_HATCH[dataset],
                    edgecolor="black", linewidth=0.6,
                    zorder=3,
                )
                if val > 30:
                    ax_rt.text(
                        x_bar[xi] + offsets[di], val * 1.08,
                        f"{val:.0f}×",
                        ha="center", va="bottom",
                        fontsize=6.5, color=color, fontweight="bold",
                    )

        ax_rt.axhline(1.0, color=UNL_COLOR["Gold Model"], linewidth=1.4,
                      linestyle="--", zorder=4)
        ax_rt.set_yscale("log")
        ax_rt.set_ylim(bottom=0.05)
        ax_rt.axhspan(1.0, ax_rt.get_ylim()[1],
                      color=UNL_COLOR["Gold Model"], alpha=0.05, zorder=0)

        ax_rt.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0], numticks=10))
        ax_rt.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
        ax_rt.yaxis.set_minor_formatter(NullFormatter())
        ax_rt.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])
        ax_rt.set_yticklabels(["0.1×", "0.5×", "1×", "2×", "5×",
                                "10×", "20×", "50×", "100×", "200×"])
        ax_rt.set_ylabel("RunTime / Gold Model (log scale)")

        ax_rt.set_xticks(x_bar)
        ax_rt.set_xticklabels(FOCUS)
        trans_rt = blended_transform_factory(ax_rt.transData, ax_rt.transAxes)
        for xi, unl in enumerate(FOCUS):
            if unl in VENUE_LABEL:
                ax_rt.text(xi, -0.08, VENUE_LABEL[unl],
                           transform=trans_rt,
                           fontsize=FS_VENUE, color="#777", style="italic",
                           ha="center", va="top")

        ax_rt.grid(axis="y", which="major", linestyle=":", alpha=0.4, zorder=0)
        ax_rt.grid(axis="y", which="minor", linestyle=":", alpha=0.15, zorder=0)
        ax_rt.set_title("(a) Runtime overhead", pad=6)
        ax_rt.text(0.02, 0.73, "more expensive\nthan retraining",
                   transform=ax_rt.transAxes, fontsize=FS,
                   color=UNL_COLOR["Gold Model"], va="center")

        ds_bar_handles = [
            mpatches.Patch(facecolor="#999", edgecolor="black",
                           linewidth=0.5, label="Cora"),
            mpatches.Patch(facecolor="#999", hatch="//", edgecolor="black",
                           linewidth=0.5, label="Citeseer"),
            mpatches.Patch(facecolor="#999", hatch="xx", edgecolor="black",
                           linewidth=0.5, label="Pubmed"),
        ]
        ax_rt.legend(handles=ds_bar_handles, loc="upper right",
                     fontsize=FS, frameon=True, framealpha=0.9,
                     edgecolor="#ccc", handlelength=1.2)

        # ── RIGHT: accuracy sweep line chart ──────────────────────────────────
        for unl in sweep_order:
            for dataset in DATASETS:
                usub = (sweep_df[(sweep_df["unlearner"] == unl) &
                                  (sweep_df["dataset"]   == dataset)]
                        .sort_values("forget_pct"))
                if usub.empty or usub["Acc (test, orig)"].isna().all():
                    continue
                xs      = [pct_to_x[p] for p in usub["forget_pct"]]
                is_bold = unl in BOLD_BASELINES
                filled  = DS_FILLED[dataset]
                mfc     = UNL_COLOR[unl] if filled else "white"
                mec     = UNL_COLOR[unl]
                ax_sw.plot(
                    xs, usub["Acc (test, orig)"],
                    color=UNL_COLOR[unl],
                    linestyle=DS_LINESTYLE[dataset],
                    marker=UNL_MARKER[unl],
                    linewidth=2.4 if is_bold else 1.3,
                    markersize=6  if is_bold else 4.5,
                    markerfacecolor=mfc,
                    markeredgewidth=1.0 if not filled else 0.4,
                    markeredgecolor=mec,
                    alpha=1.0 if is_bold else 0.70,
                    zorder=10 if is_bold else 2,
                )

        ax_sw.set_xticks(x_pos)
        ax_sw.set_xticklabels(x_labels)
        ax_sw.set_xlabel("Forget set size")
        ax_sw.set_ylabel("Test Accuracy (unlearned graph)")
        ax_sw.set_title("(b) Test accuracy vs. forget size", pad=6)
        ax_sw.grid(True, linestyle="--", alpha=0.25)
        ax_sw.set_ylim(0.6, 1.0)

        ds_line_handles = [
            mlines.Line2D([0], [0], color="black", marker="o", linestyle="-",
                          markerfacecolor="black", markersize=5, label="Cora"),
            mlines.Line2D([0], [0], color="black", marker="o", linestyle="-",
                          markerfacecolor="white", markeredgecolor="black",
                          markersize=5, label="Citeseer"),
            mlines.Line2D([0], [0], color="black", marker="o", linestyle="--",
                          markerfacecolor="black", markersize=5, label="Pubmed"),
        ]
        ax_sw.legend(handles=ds_line_handles, loc="lower left",
                     fontsize=FS, frameon=True, framealpha=0.9, edgecolor="#ccc")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"intro_figure_pubmed.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ratios   = load_runtime_ratios()
    sweep_df = load_sweep_df()
    make_figure(ratios, sweep_df)
    print("Done.")
