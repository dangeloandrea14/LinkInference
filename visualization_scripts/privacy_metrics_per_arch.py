"""
privacy_metrics_per_arch.py

For each dataset (Cora, Citeseer) produces one figure:
  output/viz/LinkAttack/edge/<dataset>_privacy_metrics.{png,pdf}

Layout: 3 rows × 5 columns
  rows    → UMIA | LinkTeller AUC | Link Stealing
  columns → GAT | GCN | GraphSAGE | SGC | SGCCGU

Each cell is a bar chart with unlearners on the X-axis.
Reference line at 0.5 (random-chance baseline) in every panel.
Identity and Gold Model bars are highlighted in gold.
Bars with invalid values (UMIA = -1) are shown as red ✕.
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "Citeseer"]
ARCHS    = ["GAT", "GCN", "GraphSAGE", "SGC", "SGCCGU"]

UNLEARNER_ORDER = [
    "Identity", "Gold Model", "FT", "SRL", "cfk", "eu_k",
    "NegGrad", "Adv.NegGrad", "UNSIR→FT", "BadTeaching", "SCRUB",
    "Fisher", "SSD", "SalUn", "IDEA", "CGU", "CEU",
]

METRICS = [
    ("UMIA",        "UMIA",
     "UMIA",
     [0.43, 0.65]),
    ("LinkTeller",  "LinkTeller unlearn auc with sampler bfs+:",
     "LinkTeller AUC (unlearned)",
     [0.0,  0.35]),
    ("LinkSteal",   "Link Stealing Attack unlearned 0 exist/non_exist",
     "Link Stealing (exist/non-exist)",
     [0.45, 1.02]),
]

HEAD = {"Identity", "Gold Model"}

BAR_COLOR    = "#5b8db8"
HEAD_COLOR   = "#e8b84b"
INVALID_COLOR = "red"

# ── label helpers ─────────────────────────────────────────────────────────────
def _cascade_label(r):
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    classes = [s.get("class", "") for s in (sub or [])]
    if any("Saliency" in c for c in classes): return "SalUn"
    if any("UNSIR"    in c for c in classes): return "UNSIR→FT"
    return "Cascade"

_LABEL_MAP = {
    "Identity": "Identity", "GoldModelGraph": "Gold Model",
    "Finetuning": "FT", "SuccessiveRandomLabels": "SRL",
    "eu_k": "eu_k", "NegGrad": "NegGrad",
    "AdvancedNegGrad": "Adv.NegGrad", "BadTeaching": "BadTeaching",
    "Scrub": "SCRUB", "FisherForgetting": "Fisher",
    "SelectiveSynapticDampening": "SSD", "IDEA": "IDEA",
    "CGU_edge": "CGU", "CEU": "CEU",
}

def label_unlearner(r):
    u   = r["unlearner"]
    ltl = r.get("parameters", {}).get("last_trainable_layers", -1)
    if u == "Cascade":
        return _cascade_label(r)
    if u == "Finetuning" and ltl == 2:
        return "cfk"
    return _LABEL_MAP.get(u, u)


# ── data loading ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    return json.loads("[" + content.rstrip(",") + "]")


def load_dataset(dataset):
    """Returns dict: data[arch][unlearner][metric_key] = value | nan"""
    result = {}
    for path in sorted(glob.glob(
            os.path.join(INPUT_DIR, f"{dataset}_*_20.json"))):
        arch = (os.path.basename(path)
                .replace(f"{dataset}_", "")
                .replace("_20.json", ""))
        if arch not in ARCHS:
            continue
        result[arch] = {}
        for r in load_json(path):
            lbl = label_unlearner(r)
            result[arch][lbl] = r
    return result


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_dataset(dataset):
    data = load_dataset(dataset)

    rc = {
        "font.family"      : "serif",
        "font.serif"       : ["Times New Roman", "DejaVu Serif"],
        "font.size"        : 9,
        "axes.labelsize"   : 9,
        "axes.titlesize"   : 10,
        "xtick.labelsize"  : 6.5,
        "ytick.labelsize"  : 8,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.linewidth"   : 0.7,
        "figure.dpi"       : 600,
    }

    n_rows = len(METRICS)
    n_cols = len(ARCHS)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.8 * n_cols, 3.2 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"{dataset} — Privacy metrics per architecture (20% forget set)",
            fontsize=11, fontweight="bold", y=1.01,
        )

        for ri, (mkey, mraw, mlabel, ylim) in enumerate(METRICS):
            for ci, arch in enumerate(ARCHS):
                ax = axes[ri][ci]
                arch_data = data.get(arch, {})

                vals, colors, invalids = [], [], []
                for unl in UNLEARNER_ORDER:
                    r = arch_data.get(unl)
                    v = r.get(mraw) if r else None

                    # treat -1 and None as invalid
                    if v is None or v < 0:
                        vals.append(np.nan)
                        invalids.append(True)
                    else:
                        vals.append(v)
                        invalids.append(False)

                    colors.append(HEAD_COLOR if unl in HEAD else BAR_COLOR)

                x = np.arange(len(UNLEARNER_ORDER))

                # gold background for Identity / Gold Model
                ax.axvspan(-0.5, 1.5, color="#FFD700", alpha=0.15, zorder=0)

                # bars for valid entries
                valid = ~np.array(invalids)
                ax.bar(x[valid], np.array(vals)[valid], 0.7,
                       color=np.array(colors)[valid],
                       edgecolor="black", linewidth=0.4, zorder=2)

                # red ✕ for invalid entries
                if any(invalids):
                    ax.plot(x[np.array(invalids)],
                            np.full(sum(invalids), ylim[0] + 0.01 * (ylim[1]-ylim[0])),
                            marker="x", linestyle="None",
                            color=INVALID_COLOR, markersize=7,
                            markeredgewidth=1.5, zorder=4)

                # 0.5 random baseline
                ax.axhline(0.5, color="#888", linestyle="--",
                           linewidth=0.9, zorder=3)

                # Gold Model reference line
                gold_r = arch_data.get("Gold Model")
                gold_v = gold_r.get(mraw) if gold_r else None
                if gold_v is not None and gold_v >= 0:
                    ax.axhline(gold_v, color="#c0392b", linestyle=":",
                               linewidth=1.2, zorder=4)

                # no-data notice for LinkTeller on SGC / SGCCGU
                all_nan = all(np.isnan(v) or v < 0
                              for v in vals if v is not None) \
                          or all(np.isnan(v) for v in vals)
                if mkey == "LinkTeller" and arch in ("SGC", "SGCCGU"):
                    ax.text(0.5, 0.5, "not available",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=8, color="#999", style="italic")

                ax.set_xlim(-0.6, len(UNLEARNER_ORDER) - 0.4)
                ax.set_ylim(ylim)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    UNLEARNER_ORDER, rotation=45, ha="right", fontsize=6.5)
                ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

                # column title (top row only)
                if ri == 0:
                    ax.set_title(arch, fontsize=10, fontweight="bold")

                # row label (left column only)
                if ci == 0:
                    ax.set_ylabel(mlabel, fontsize=8.5)

        # legend
        handles = [
            mpatches.Patch(facecolor=HEAD_COLOR,  edgecolor="black",
                           linewidth=0.4, label="Identity / Gold Model"),
            mpatches.Patch(facecolor=BAR_COLOR,   edgecolor="black",
                           linewidth=0.4, label="Other unlearners"),
            plt.Line2D([0], [0], color="#888",    linestyle="--",
                       linewidth=0.9,  label="Random baseline (0.5)"),
            plt.Line2D([0], [0], color="#c0392b", linestyle=":",
                       linewidth=1.2,  label="Gold Model value"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=8.5, frameon=True, framealpha=0.9,
                   edgecolor="#ccc", bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.subplots_adjust(hspace=0.55, wspace=0.3)

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"{dataset}_privacy_metrics.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"\n── {dataset} ──")
        plot_dataset(dataset)
    print("\nDone.")
