"""
intro_figure.py

Produces one NeurIPS-ready combined intro figure:
  output/viz/LinkAttack/edge/intro_figure.{png,pdf}

Two panels side by side — GCN, Cora & Citeseer:

  Left  — RunTime relative to Gold Model (20% forget set), bar chart.
  Right — Test Accuracy (original graph) vs. forget %, line plot.

Unified encoding across both panels:
  Color   = unlearner (shared palette, methods sorted alphabetically)
  Dataset = Cora (filled bar / filled marker) vs Citeseer (hatched bar / open marker)
  Dataset distinction annotated inside each panel; legend shows methods only.
"""

import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
TABLE3_DIR = "output/runs/table3"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
DATASETS       = ["Cora", "Citeseer", "Pubmed"]                              # bar chart
SWEEP_DATASETS = ["AmazonComputers", "AmazonPhotos", "DBLP"]               # line plot
PCTS       = [1, 5, 10, 20, 50]
SWEEP_PCTS = [1, 5, 20, 50]     # table3 datasets have no 10% split

# Methods sorted alphabetically; baselines appended at end of sweep panel only
FOCUS          = ["CEU", "SalUn", "GNNDelete", "IDEA", "SSD"]
SWEEP_EXTRAS   = ["Gold Model"]
BOLD_BASELINES = {"Gold Model"}

# Venue labels shown under method names in the bar chart x-axis
VENUE_LABEL = {
    "CEU"      : "(KDD'23)",
    "IDEA"     : "(KDD'24)",
    "SSD"      : "(AAAI'24)",
    "SalUn"    : "(ICLR'24)",
    "GNNDelete": "(ICLR'23)",
}

# Unified colour palette (unlearner → colour)
UNL_COLOR = {
    "Gold Model": "#c0392b",
    "CEU"       : "#76b7b2",
    "GNNDelete" : "#4e79a7",
    "IDEA"      : "#b07aa1",
    "SSD"       : "#f28e2b",
    "SalUn"     : "#59a14f",
}
UNL_MARKER = {
    "Gold Model": "D",
    "CEU"       : "X",
    "GNNDelete" : "o",
    "IDEA"      : "P",
    "SSD"       : "^",
    "SalUn"     : "v",
}

# Dataset encoding — no legend entries; annotated inside panels instead
DS_HATCH = {"Cora": "",  "Citeseer": "//", "Coauthor": "xx", "Pubmed": "xx"}

# Bar-chart datasets keep grayscale; sweep datasets use distinct colors
DS_COLOR = {
    "Coauthor"        : "#000000",
    "Cora"            : "#404040",
    "Pubmed"          : "#808080",
    "Citeseer"        : "#949494",
    "AmazonComputers" : "#5b8db8",
    "AmazonPhotos"    : "#b85c5c",
    "DBLP"            : "#5a9e6f",
}
DS_MARKER = {
    "Cora"            : "o",
    "Citeseer"        : "s",
    "Coauthor"        : "^",
    "Pubmed"          : "D",
    "AmazonComputers" : "o",
    "AmazonPhotos"    : "s",
    "DBLP"            : "^",
}
# Short display labels for in-plot annotations
DS_LABEL = {
    "AmazonComputers" : "Computers",
    "AmazonPhotos"    : "Photos",
    "DBLP"            : "DBLP",
    "Pubmed"          : "Pubmed",
}


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
        "FisherForgetting"           : "FF",
    }.get(u, u)


# ── data loading ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return json.loads("[" + content.rstrip(",") + "]")


def find_run_file(dataset, pct, run_dir=None):
    if run_dir is None:
        run_dir = INPUT_DIR
    path = os.path.join(run_dir, f"{dataset}_GCN_{pct}.json")
    return path if os.path.exists(path) else None


def load_runtime_ratios():
    """ratios[dataset][unlearner] = RunTime / GoldModel RunTime  (20% forget)
    also returns mean Gold Model RunTime in seconds across datasets"""
    ratios, gold_rts = {}, []
    for dataset in DATASETS:
        path = find_run_file(dataset, 20)
        if path is None:
            continue
        records  = load_json(path)
        labelled = {label_unlearner(r): r.get("RunTime") for r in records}
        gold_rt  = labelled.get("Gold Model")
        if not gold_rt:
            continue
        gold_rts.append(gold_rt)
        ratios[dataset] = {
            unl: (labelled[unl] / gold_rt
                  if labelled.get(unl) and labelled[unl] > 0 else np.nan)
            for unl in FOCUS
        }
    return ratios, float(np.mean(gold_rts)) if gold_rts else 1.0


def load_sweep_df():
    """DataFrame with columns: dataset, forget_pct, unlearner, Acc (test, orig)"""
    ACC_KEY = "sklearn.metrics.accuracy_score.test.unlearned.on_graph:False"
    want    = set(FOCUS) | set(SWEEP_EXTRAS)
    rows    = []
    for dataset in SWEEP_DATASETS:
        run_dir = INPUT_DIR if dataset == "Pubmed" else TABLE3_DIR
        for pct in SWEEP_PCTS:
            path = find_run_file(dataset, pct, run_dir)
            if path is None:
                continue
            for r in load_json(path):
                unl = label_unlearner(r)
                if unl not in want:
                    continue
                rows.append({
                    "dataset"         : dataset,
                    "forget_pct"      : pct,
                    "unlearner"       : unl,
                    "Acc (test, orig)": r.get(ACC_KEY, np.nan),
                })
    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────
def make_figure(ratios, sweep_df, gold_rt_avg=1.0):
    FS       = 9    # single font size used everywhere
    FS_VENUE = 7    # smaller for venue labels (secondary info)

    rc = {
        "text.usetex"      : True,
        "font.family"      : "serif",
        "font.size"        : FS,
        "axes.labelsize"   : FS,
        "axes.titlesize"   : FS,
        "xtick.labelsize"  : FS,
        "ytick.labelsize"  : FS,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.linewidth"   : 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "figure.dpi"       : 600,
    }

    n_unl   = len(FOCUS)
    width   = 0.25
    x_bar   = np.arange(n_unl) * 1.4
    offsets = np.array([-1.0, 0.0, 1.0]) * width

    x_pos    = list(range(len(SWEEP_PCTS)))
    pct_to_x = {p: i for i, p in enumerate(SWEEP_PCTS)}
    x_labels = [f"{p}\\%" for p in SWEEP_PCTS]

    sweep_order = [u for u in SWEEP_EXTRAS + FOCUS if u not in BOLD_BASELINES] + \
                  [u for u in SWEEP_EXTRAS + FOCUS if u in BOLD_BASELINES]

    with plt.rc_context(rc):
        fig, (ax_rt, ax_sw) = plt.subplots(
            1, 2, figsize=(6.75, 2.8),
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
                        f"${val:.0f}\\times$",
                        ha="center", va="bottom",
                        fontsize=5.5, color=color, fontweight="bold",
                    )

        ax_rt.axhline(1.0, color=UNL_COLOR["Gold Model"], linewidth=0.9,
                      linestyle="--", zorder=4)
        ax_rt.set_yscale("log")
        ax_rt.set_ylim(bottom=0.05)
        ax_rt.axhspan(1.0, ax_rt.get_ylim()[1],
                      color=UNL_COLOR["Gold Model"], alpha=0.05, zorder=0)

        ax_rt.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0], numticks=10))
        ax_rt.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
        ax_rt.yaxis.set_minor_formatter(NullFormatter())
        all_vals = [v for ds in ratios.values() for v in ds.values() if not np.isnan(v)]
        max_ratio = max(all_vals) if all_vals else 10
        all_ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
        tick_ratios = [t for t in all_ticks if t <= max_ratio * 1.5]
        tick_secs   = [r * gold_rt_avg for r in tick_ratios]
        tick_labels = [f"{s:.2f}s" if s < 1 else f"{s:.1f}s" if s < 10 else f"{s:.0f}s"
                       for s in tick_secs]
        ax_rt.set_yticks(tick_ratios)
        ax_rt.set_yticklabels(tick_labels)
        ax_rt.set_ylabel("RunTime (log scale)")

        # Method names as tick labels; venue labels as smaller gray italic below
        ax_rt.set_xticks(x_bar)
        ax_rt.set_xticklabels(FOCUS)
        trans_rt = blended_transform_factory(ax_rt.transData, ax_rt.transAxes)
        for xi, unl in enumerate(FOCUS):
            if unl in VENUE_LABEL:
                ax_rt.text(x_bar[xi], -0.14, VENUE_LABEL[unl],
                           transform=trans_rt,
                           fontsize=FS_VENUE, color="#444", style="italic",
                           ha="center", va="top")

        ax_rt.grid(axis="y", which="major", linestyle=":", alpha=0.25, linewidth=0.5, zorder=0)
        ax_rt.grid(axis="y", which="minor", linestyle=":", alpha=0.1,  linewidth=0.4, zorder=0)
        ax_rt.set_title("(a) Runtime", pad=4)
        trans_mixed = blended_transform_factory(ax_rt.transAxes, ax_rt.transData)
        ax_rt.text(0.98, 1.08, "more expensive\nthan retraining",
                   transform=trans_mixed, fontsize=FS - 1,
                   color=UNL_COLOR["Gold Model"], va="bottom", ha="right")

        ds_bar_handles = [
            mpatches.Patch(facecolor="#999", edgecolor="black",
                           linewidth=0.5, label="Cora"),
            mpatches.Patch(facecolor="#999", hatch="//", edgecolor="black",
                           linewidth=0.5, label="Citeseer"),
            mpatches.Patch(facecolor="#999", hatch="xx", edgecolor="black",
                           linewidth=0.5, label="Pubmed"),
        ]
        ax_rt.legend(handles=ds_bar_handles, loc="upper right", ncol=3,
                     fontsize=FS - 2, frameon=True, framealpha=0.9,
                     edgecolor="#ccc", handlelength=1.0,
                     borderpad=0.4, labelspacing=0.2)

        # ── RIGHT: accuracy sweep line chart (Gold Model only) ───────────────
        for dataset in SWEEP_DATASETS:
            usub = (sweep_df[(sweep_df["unlearner"] == "Gold Model") &
                              (sweep_df["dataset"]   == dataset)]
                    .sort_values("forget_pct"))
            if usub.empty or usub["Acc (test, orig)"].isna().all():
                continue
            vals  = usub["Acc (test, orig)"].values
            xs    = [pct_to_x[p] for p in usub["forget_pct"]]
            color = DS_COLOR[dataset]
            delta = vals[0] - vals[-1]

            ax_sw.plot(xs, vals,
                       color=color, linewidth=1.2,
                       marker=DS_MARKER[dataset], markersize=4.0,
                       markerfacecolor=color, markeredgewidth=0.3,
                       markeredgecolor=color)

            mid = len(xs) // 2
            label = DS_LABEL.get(dataset, dataset)
            ax_sw.text(xs[mid], vals[mid] - 0.012, label,
                       va="top", ha="center",
                       color=color, fontsize=FS - 2, clip_on=False)

            ax_sw.text(xs[-1] + 0.15, vals[-1],
                       f"$\\Delta={delta*100:.1f}\\%$",
                       va="center", ha="left",
                       color=color, fontsize=FS - 2, clip_on=False)

        ax_sw.set_xticks(x_pos)
        ax_sw.set_xticklabels(x_labels)
        ax_sw.set_xlim(-0.3, len(x_pos) - 0.2)
        ax_sw.set_xlabel("Forget set size")
        ax_sw.set_ylabel("Test Accuracy")
        ax_sw.set_title("(b) GCN Test accuracy vs. forget size", pad=4)
        ax_sw.grid(True, linestyle=":", alpha=0.25, linewidth=0.5)
        ax_sw.set_ylim(0.5, 1.0)


        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, bottom=0.18)

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"intro_figure.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ratios, gold_rt_avg = load_runtime_ratios()
    sweep_df = load_sweep_df()
    make_figure(ratios, sweep_df, gold_rt_avg)
    print("Done.")
