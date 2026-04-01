"""
Visualize LinkAttack/edge benchmark results.

Generates three figures in output/viz/LinkAttack/edge/:
  fig1_heatmaps.png       — per-dataset heatmaps: unlearner × arch for key metrics (20% forget)
  fig2_forget_pct.png     — GCN forget-% sweep (5/20/50/99/100) for both datasets
  fig3_link_attacks.png   — LinkTeller & LinkStealing bar charts across unlearners (20% forget)
"""

import json, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
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

METRICS = {
    "Acc (test)"          : "sklearn.metrics.accuracy_score.test.unlearned.on_graph:True",
    "Acc (test, orig)"    : "sklearn.metrics.accuracy_score.test.unlearned.on_graph:False",
    "Acc (forget)"        : "sklearn.metrics.accuracy_score.forget.unlearned.on_graph:True",
    "F1 (test)"           : "f1_macro.test.unlearned.on_graph:True",
    "AUS"                 : "AUS",
    "UMIA"                : "UMIA",
    "LinkTeller"          : "LinkTeller unlearn auc with sampler bfs+:",
    "LinkStealing"        : "Link Stealing Attack unlearned 0 exist/non_exist",
    "Runtime (s)"         : "RunTime",
}

# Higher-is-better flags (False → lower is better → reversed colormap)
METRIC_HIB = {
    "Acc (test)": True, "Acc (test, orig)": True, "Acc (forget)": True,
    "F1 (test)": True, "AUS": True, "UMIA": False, "LinkTeller": False,
    "LinkStealing": False, "Runtime (s)": False,
}

ARCH_ORDER = ["GAT", "GCN", "GIN", "GraphSAGE", "MLP", "SGC", "SGCCGU"]


def label_unlearner(r):
    u = r["unlearner"]
    ltl = r.get("parameters", {}).get("last_trainable_layers", -1)
    sub = r.get("parameters", {}).get("sub_unlearner", [])
    if u == "Cascade":
        classes = [s.get("class", "") for s in sub]
        if any("UNSIR" in c for c in classes):
            return "UNSIR→FT"
        if any("Saliency" in c for c in classes):
            return "SalUn"
    key = (u, ltl if u == "Finetuning" else -1)
    return UNLEARNER_LABELS.get(key, u)


def parse_filename(fname):
    name = os.path.splitext(os.path.basename(fname))[0]
    parts = name.split("_")
    dataset = parts[0]
    last = parts[-1]
    if last.isdigit():
        return dataset, "_".join(parts[1:-1]), int(last)
    return dataset, "_".join(parts[1:]), None


def _load_json(path):
    """Load a JSON file that may be a plain array or concatenated top-level objects."""
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    # Wrap concatenated objects into a JSON array
    return json.loads("[" + content.rstrip(",") + "]")


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.json"))):
        dataset, arch, forget_pct = parse_filename(path)
        records = _load_json(path)
        for r in records:
            row = {
                "dataset"    : dataset,
                "arch"       : arch,
                "forget_pct" : forget_pct,
                "unlearner"  : label_unlearner(r),
            }
            for short, key in METRICS.items():
                row[short] = r.get(key, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — heatmaps  (dataset × metric, rows=unlearner, cols=arch, @ 20% forget)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_heatmaps(df):
    datasets = sorted(df["dataset"].unique())
    metric_names = list(METRICS.keys())

    fig, axes = plt.subplots(
        len(datasets), len(metric_names),
        figsize=(3.2 * len(metric_names), 4.5 * len(datasets)),
        squeeze=False,
    )
    fig.suptitle("Key metrics at 20% forget — rows: unlearner, cols: architecture",
                 fontsize=13, y=1.01)

    for di, dataset in enumerate(datasets):
        sub = df[(df["dataset"] == dataset) & (df["forget_pct"] == 20)].copy()
        # keep only standard archs (drop GIN_edge etc.)
        sub = sub[sub["arch"].isin(ARCH_ORDER)]
        for mi, metric in enumerate(metric_names):
            ax = axes[di][mi]
            pivot = (
                sub.pivot_table(index="unlearner", columns="arch", values=metric, aggfunc="first")
                   .reindex(index=UNLEARNER_ORDER, columns=ARCH_ORDER)
            )
            hib  = METRIC_HIB[metric]
            cmap = "RdYlGn" if hib else "RdYlGn_r"
            vals = pivot.values.astype(float)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            ax.set_xticks(range(len(ARCH_ORDER)))
            ax.set_xticklabels(ARCH_ORDER, rotation=45, ha="right", fontsize=7)
            if mi == 0:
                ax.set_yticks(range(len(UNLEARNER_ORDER)))
                ax.set_yticklabels(UNLEARNER_ORDER, fontsize=7)
                ax.set_ylabel(dataset, fontsize=10, fontweight="bold")
            else:
                ax.set_yticks([])
            ax.set_title(metric, fontsize=8)
            # annotate cells
            for r in range(vals.shape[0]):
                for c in range(vals.shape[1]):
                    v = vals[r, c]
                    if not np.isnan(v):
                        txt = f"{v:.2f}" if metric != "Runtime (s)" else f"{v:.1f}"
                        ax.text(c, r, txt, ha="center", va="center", fontsize=5.5,
                                color="black")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_heatmaps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — GCN forget-% sweep
# ══════════════════════════════════════════════════════════════════════════════
def fig2_forget_pct(df):
    datasets  = sorted(df["dataset"].unique())
    pcts      = [5, 20, 50, 99, 100]
    plot_metrics = ["Acc (test)", "AUS", "UMIA", "LinkTeller", "LinkStealing"]

    sub = df[(df["arch"] == "GCN") & (df["forget_pct"].isin(pcts))].copy()

    n_rows = len(datasets)
    n_cols = len(plot_metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.5 * n_rows), squeeze=False)
    fig.suptitle("GCN — metric vs. forget % (edge removal)", fontsize=13)

    cmap_u = plt.get_cmap("tab20")
    unlearner_colors = {u: cmap_u(i / len(UNLEARNER_ORDER)) for i, u in enumerate(UNLEARNER_ORDER)}

    for di, dataset in enumerate(datasets):
        dsub = sub[sub["dataset"] == dataset]
        for mi, metric in enumerate(plot_metrics):
            ax = axes[di][mi]
            for unl in UNLEARNER_ORDER:
                usub = dsub[dsub["unlearner"] == unl].sort_values("forget_pct")
                if usub.empty or usub[metric].isna().all():
                    continue
                ax.plot(usub["forget_pct"], usub[metric],
                        marker="o", markersize=4, linewidth=1.4,
                        label=unl, color=unlearner_colors[unl])
            ax.set_xticks(pcts)
            ax.set_xlabel("Forget %", fontsize=8)
            if mi == 0:
                ax.set_ylabel(dataset, fontsize=10, fontweight="bold")
            ax.set_title(metric, fontsize=9)
            ax.grid(True, alpha=0.3)
            hib = METRIC_HIB[metric]
            ax.set_ylim(bottom=0)

    # shared legend
    handles = [mpatches.Patch(color=unlearner_colors[u], label=u) for u in UNLEARNER_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=9,
               fontsize=7, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_forget_pct.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Link-privacy attack focus (LinkTeller + LinkStealing)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_link_attacks(df):
    datasets      = sorted(df["dataset"].unique())
    attack_metrics = {
        "LinkTeller AUC\n(unlearned)": "LinkTeller",
        "Link Stealing\n(exist/non-exist)": "LinkStealing",
    }

    sub = df[df["forget_pct"] == 20].copy()
    sub = sub[sub["arch"].isin(ARCH_ORDER)]

    n_rows = len(datasets) * len(attack_metrics)
    fig, axes = plt.subplots(
        len(datasets), len(attack_metrics),
        figsize=(14, 5 * len(datasets)), squeeze=False,
    )
    fig.suptitle("Link privacy attacks at 20% forget — lower is better", fontsize=13)

    x = np.arange(len(UNLEARNER_ORDER))
    width = 0.12
    arch_colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(ARCH_ORDER)))

    for di, dataset in enumerate(datasets):
        dsub = sub[sub["dataset"] == dataset]
        for mi, (title, metric) in enumerate(attack_metrics.items()):
            ax = axes[di][mi]
            for ai, arch in enumerate(ARCH_ORDER):
                asub = dsub[dsub["arch"] == arch]
                vals = []
                for unl in UNLEARNER_ORDER:
                    row = asub[asub["unlearner"] == unl]
                    vals.append(row[metric].values[0] if not row.empty else np.nan)
                offset = (ai - len(ARCH_ORDER) / 2) * width + width / 2
                ax.bar(x + offset, vals, width, label=arch, color=arch_colors[ai], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(UNLEARNER_ORDER, rotation=45, ha="right", fontsize=7.5)
            ax.set_title(f"{dataset} — {title}", fontsize=10)
            ax.set_ylabel(metric, fontsize=8)
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="random (0.5)")
            ax.grid(axis="y", alpha=0.3)
            if di == 0 and mi == len(attack_metrics) - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig3_link_attacks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Radar / spider chart: per-unlearner profile (Cora GCN 20%)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_radar(df):
    """Radar chart: one spoke per metric, one line per unlearner. Cora+Citeseer, GCN 20%."""
    radar_metrics = ["Acc (test)", "F1 (test)", "AUS", "UMIA", "LinkTeller", "LinkStealing"]
    # For radar: normalise all metrics to [0,1] where 1 = best
    datasets = sorted(df["dataset"].unique())

    n_cols = len(datasets)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7),
                             subplot_kw=dict(polar=True))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Unlearner profile — GCN, 20% forget\n(each axis normalised to [0,1], 1=best)",
                 fontsize=12)

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    cmap_u = plt.get_cmap("tab20")

    for di, dataset in enumerate(datasets):
        ax = axes[di]
        sub = df[(df["dataset"] == dataset) & (df["arch"] == "GCN") & (df["forget_pct"] == 20)]

        # normalise per metric
        normed = {}
        for m in radar_metrics:
            col = sub[m].values.astype(float)
            mn, mx = np.nanmin(col), np.nanmax(col)
            hib = METRIC_HIB[m]
            if mx == mn:
                normed[m] = {u: 0.5 for u in sub["unlearner"]}
            else:
                normed[m] = {}
                for _, row in sub.iterrows():
                    v = (row[m] - mn) / (mx - mn)
                    normed[m][row["unlearner"]] = v if hib else 1 - v

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, size=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], size=6)
        ax.set_ylim(0, 1)
        ax.set_title(dataset, size=11, pad=15, fontweight="bold")

        for i, unl in enumerate(UNLEARNER_ORDER):
            row = sub[sub["unlearner"] == unl]
            if row.empty:
                continue
            vals = [normed[m].get(unl, np.nan) for m in radar_metrics]
            if all(np.isnan(v) for v in vals):
                continue
            vals_plot = [v if not np.isnan(v) else 0 for v in vals]
            vals_plot += vals_plot[:1]
            color = cmap_u(i / len(UNLEARNER_ORDER))
            ax.plot(angles, vals_plot, linewidth=1.5, label=unl, color=color)
            ax.fill(angles, vals_plot, alpha=0.05, color=color)

    # legend outside
    handles = [mpatches.Patch(color=plt.get_cmap("tab20")(i / len(UNLEARNER_ORDER)), label=u)
               for i, u in enumerate(UNLEARNER_ORDER)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig4_radar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Paper-ready: Acc(test) & UMIA vs. forget% for GCN on Cora & Citeseer
# ══════════════════════════════════════════════════════════════════════════════
def fig_paper_gcn_sweep(df):
    """NeurIPS-ready 2×2 figure: Acc(test) & UMIA vs. forget% for GCN on Cora & Citeseer."""
    datasets     = ["Cora", "Citeseer"]
    pcts         = [5, 20, 50, 99, 100]
    plot_metrics = [
        ("Acc (test)", "Test Accuracy"),
        ("UMIA",       "UMIA"),
    ]

    sub = df[
        (df["arch"] == "GCN") &
        (df["dataset"].isin(datasets)) &
        (df["forget_pct"].isin(pcts))
    ].copy()

    # ── NeurIPS-style rc ────────────────────────────────────────────────────
    rc = {
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "DejaVu Serif"],
        "font.size"          : 11,
        "axes.labelsize"     : 11,
        "axes.titlesize"     : 12,
        "xtick.labelsize"    : 10,
        "ytick.labelsize"    : 10,
        "legend.fontsize"    : 8,
        "lines.linewidth"    : 1.5,
        "lines.markersize"   : 5,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "grid.alpha"         : 0.25,
        "grid.linestyle"     : "--",
        "grid.linewidth"     : 0.6,
        "figure.dpi"         : 600,
    }

    BOLD_BASELINES = {"Identity", "Gold Model"}

    # Colour + line-style per unlearner (aids B&W reproduction)
    cmap_u       = plt.get_cmap("tab20")
    n_u          = len(UNLEARNER_ORDER)
    ul_color     = {u: cmap_u(i / n_u) for i, u in enumerate(UNLEARNER_ORDER)}
    _ls_cycle    = ["-", "--", "-.", ":"]
    ul_ls        = {u: _ls_cycle[i % 4]  for i, u in enumerate(UNLEARNER_ORDER)}
    _mk_cycle    = ["o", "s", "^", "D", "v", "P", "X"]
    ul_mk        = {u: _mk_cycle[i % 7]  for i, u in enumerate(UNLEARNER_ORDER)}

    # Use evenly-spaced discrete positions so 99% and 100% don't overlap
    x_pos   = list(range(len(pcts)))
    pct_to_x = {p: i for i, p in enumerate(pcts)}
    x_labels = [f"{p}%" for p in pcts]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            len(datasets), len(plot_metrics),
            figsize=(12.0, 8.0),
            squeeze=False,
        )
        fig.suptitle("GCN Architecture — Edge Removal Unlearning",
                     fontsize=11, fontweight="bold", y=1.01)

        for di, dataset in enumerate(datasets):
            dsub = sub[sub["dataset"] == dataset]
            for mi, (metric, col_title) in enumerate(plot_metrics):
                ax = axes[di][mi]

                # Draw regular unlearners first, baselines last (on top)
                draw_order = [u for u in UNLEARNER_ORDER if u not in BOLD_BASELINES] + \
                             [u for u in UNLEARNER_ORDER if u in BOLD_BASELINES]
                for unl in draw_order:
                    usub = dsub[dsub["unlearner"] == unl].sort_values("forget_pct")
                    if usub.empty or usub[metric].isna().all():
                        continue
                    xs = [pct_to_x[p] for p in usub["forget_pct"]]
                    is_bold = unl in BOLD_BASELINES
                    ax.plot(
                        xs, usub[metric],
                        marker=ul_mk[unl], label=unl,
                        color=ul_color[unl], linestyle=ul_ls[unl],
                        linewidth=2.8 if is_bold else 1.5,
                        markersize=7 if is_bold else 5,
                        markeredgewidth=0.8 if is_bold else 0.4,
                        markeredgecolor="white",
                        zorder=10 if is_bold else 2,
                        alpha=0.55 if not is_bold else 1.0,
                    )

                # Reference line for UMIA (random baseline = 0.5)
                if metric == "UMIA":
                    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.9,
                               zorder=0, label="_nolegend_")

                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels)

                if di == len(datasets) - 1:
                    ax.set_xlabel("Forget set size")

                # Row label = dataset name (left column only)
                if mi == 0:
                    ax.set_ylabel(dataset, fontsize=9.5, fontweight="bold", labelpad=6)

                # Column title (top row only)
                if di == 0:
                    ax.set_title(col_title, fontsize=9.5, pad=4)

                # Tighter y-limits
                yvals = sub[sub["dataset"] == dataset][metric].dropna()
                if not yvals.empty:
                    pad = (yvals.max() - yvals.min()) * 0.15 or 0.02
                    ax.set_ylim(yvals.min() - pad, yvals.max() + pad)

        # ── shared legend below the figure ─────────────────────────────────
        # Only include unlearners that actually appear in this subset
        present = set(sub["unlearner"].unique())
        handles = [
            plt.Line2D(
                [0], [0],
                color=ul_color[u], linestyle=ul_ls[u],
                marker=ul_mk[u],
                markersize=6 if u in BOLD_BASELINES else 4,
                linewidth=2.5 if u in BOLD_BASELINES else 1.2,
                markeredgewidth=0.5, markeredgecolor="white",
                label=u,
            )
            for u in UNLEARNER_ORDER if u in present
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=6,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
            framealpha=0.9,
            edgecolor="#ccc",
        )

        plt.tight_layout(rect=[0, 0.09, 1, 1])
        plt.subplots_adjust(hspace=0.35, wspace=0.30)

        for fmt in ("png", "pdf"):
            kw  = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"fig_paper_gcn_sweep.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Same as fig_paper_gcn_sweep but Acc uses on_graph:False
# ══════════════════════════════════════════════════════════════════════════════
def fig_paper_gcn_sweep_orig_graph(df):
    """Like fig_paper_gcn_sweep but accuracy evaluated on original graph (on_graph:False)."""
    datasets     = ["Cora", "Citeseer"]
    pcts         = [5, 20, 50, 99, 100]
    plot_metrics = [
        ("Acc (test, orig)", "Test Accuracy (original graph)"),
        ("UMIA",             "UMIA"),
    ]

    sub = df[
        (df["arch"] == "GCN") &
        (df["dataset"].isin(datasets)) &
        (df["forget_pct"].isin(pcts))
    ].copy()

    rc = {
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "DejaVu Serif"],
        "font.size"          : 11,
        "axes.labelsize"     : 11,
        "axes.titlesize"     : 12,
        "xtick.labelsize"    : 10,
        "ytick.labelsize"    : 10,
        "legend.fontsize"    : 8,
        "lines.linewidth"    : 1.5,
        "lines.markersize"   : 5,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "grid.alpha"         : 0.25,
        "grid.linestyle"     : "--",
        "grid.linewidth"     : 0.6,
        "figure.dpi"         : 600,
    }

    BOLD_BASELINES = {"Identity", "Gold Model"}

    cmap_u    = plt.get_cmap("tab20")
    n_u       = len(UNLEARNER_ORDER)
    ul_color  = {u: cmap_u(i / n_u) for i, u in enumerate(UNLEARNER_ORDER)}
    _ls_cycle = ["-", "--", "-.", ":"]
    ul_ls     = {u: _ls_cycle[i % 4] for i, u in enumerate(UNLEARNER_ORDER)}
    _mk_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    ul_mk     = {u: _mk_cycle[i % 7] for i, u in enumerate(UNLEARNER_ORDER)}

    x_pos    = list(range(len(pcts)))
    pct_to_x = {p: i for i, p in enumerate(pcts)}
    x_labels = [f"{p}%" for p in pcts]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            len(datasets), len(plot_metrics),
            figsize=(12.0, 8.0),
            squeeze=False,
        )
        fig.suptitle("GCN Architecture — Edge Removal Unlearning (accuracy on original graph)",
                     fontsize=11, fontweight="bold", y=1.01)

        for di, dataset in enumerate(datasets):
            dsub = sub[sub["dataset"] == dataset]
            for mi, (metric, col_title) in enumerate(plot_metrics):
                ax = axes[di][mi]

                draw_order = [u for u in UNLEARNER_ORDER if u not in BOLD_BASELINES] + \
                             [u for u in UNLEARNER_ORDER if u in BOLD_BASELINES]
                for unl in draw_order:
                    usub = dsub[dsub["unlearner"] == unl].sort_values("forget_pct")
                    if usub.empty or usub[metric].isna().all():
                        continue
                    xs = [pct_to_x[p] for p in usub["forget_pct"]]
                    is_bold = unl in BOLD_BASELINES
                    ax.plot(
                        xs, usub[metric],
                        marker=ul_mk[unl], label=unl,
                        color=ul_color[unl], linestyle=ul_ls[unl],
                        linewidth=2.8 if is_bold else 1.5,
                        markersize=7 if is_bold else 5,
                        markeredgewidth=0.8 if is_bold else 0.4,
                        markeredgecolor="white",
                        zorder=10 if is_bold else 2,
                        alpha=0.55 if not is_bold else 1.0,
                    )

                if metric == "UMIA":
                    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.9,
                               zorder=0, label="_nolegend_")

                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels)

                if di == len(datasets) - 1:
                    ax.set_xlabel("Forget set size")

                if mi == 0:
                    ax.set_ylabel(dataset, fontsize=9.5, fontweight="bold", labelpad=6)

                if di == 0:
                    ax.set_title(col_title, fontsize=9.5, pad=4)

                yvals = sub[sub["dataset"] == dataset][metric].dropna()
                if not yvals.empty:
                    pad = (yvals.max() - yvals.min()) * 0.15 or 0.02
                    ax.set_ylim(yvals.min() - pad, yvals.max() + pad)

        present = set(sub["unlearner"].unique())
        handles = [
            plt.Line2D(
                [0], [0],
                color=ul_color[u], linestyle=ul_ls[u],
                marker=ul_mk[u],
                markersize=6 if u in BOLD_BASELINES else 4,
                linewidth=2.5 if u in BOLD_BASELINES else 1.2,
                markeredgewidth=0.5, markeredgecolor="white",
                label=u,
            )
            for u in UNLEARNER_ORDER if u in present
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=6,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
            framealpha=0.9,
            edgecolor="#ccc",
        )

        plt.tight_layout(rect=[0, 0.09, 1, 1])
        plt.subplots_adjust(hspace=0.35, wspace=0.30)

        for fmt in ("png", "pdf"):
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = 600
            out = os.path.join(OUTPUT_DIR, f"fig_paper_gcn_sweep_orig_graph.{fmt}")
            plt.savefig(out, **kw)
            print(f"Saved {out}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_all()
    print(f"Loaded {len(df)} rows, datasets={sorted(df['dataset'].unique())}, "
          f"archs={sorted(df['arch'].unique())}")
    fig1_heatmaps(df)
    fig2_forget_pct(df)
    fig3_link_attacks(df)
    fig4_radar(df)
    fig_paper_gcn_sweep(df)
    fig_paper_gcn_sweep_orig_graph(df)
    print("Done.")
