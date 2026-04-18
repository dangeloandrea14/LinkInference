"""
goldmodel_accuracy_table.py  —  Gold Model test accuracy across architectures

For each (dataset, architecture) pair, reports:
  - Identity accuracy  (original model = upper bound)
  - Gold Model accuracy at 5% and 20% forget set

Metric: sklearn accuracy_score on test set, original graph (on_graph:False).
"""

import json, os
import numpy as np
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "output/runs/LinkAttack/edge"
OUTPUT_DIR = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["Cora", "Citeseer", "Coauthor", "Pubmed"]
ARCHS    = ["GCN", "GIN", "GAT", "GraphSAGE", "SGC", "SGCCGU"]
PCTS     = [5, 20]

ACC_KEY  = "sklearn.metrics.accuracy_score.test.unlearned.on_graph:False"

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
        "Identity"      : "Identity",
        "GoldModelGraph": "Gold Model",
        "Finetuning"    : "FT" if ltl == -1 else "cfk",
        "Scrub"         : "SCRUB",
        "SelectiveSynapticDampening": "SSD",
        "IDEA"          : "IDEA",
        "CGU_edge"      : "CGU",
        "CEU"           : "CEU",
    }.get(u, u)

def get_acc(dataset, arch, pct):
    path = os.path.join(INPUT_DIR, f"{dataset}_{arch}_{pct}.json")
    if not os.path.exists(path):
        return None, None
    records  = load_json(path)
    labelled = {label_unlearner(r): r.get(ACC_KEY) for r in records}
    return labelled.get("Identity"), labelled.get("Gold Model")

# ── collect data ──────────────────────────────────────────────────────────────
# data[dataset][arch] = {"identity": float, 5: float, 20: float}
data = defaultdict(dict)
for ds in DATASETS:
    for arch in ARCHS:
        # Identity is stable across pcts — take from 5% file if available, else 20%
        identity = None
        gold = {}
        for pct in PCTS:
            idt, gld = get_acc(ds, arch, pct)
            if identity is None and idt is not None:
                identity = idt
            if gld is not None:
                gold[pct] = gld
        if identity is not None or gold:
            data[ds][arch] = {"identity": identity, **gold}

# ── format helpers ────────────────────────────────────────────────────────────
def fmt_acc(val):
    if val is None:
        return "--"
    return f"{val * 100:.1f}"

def fmt_delta(gold, identity):
    if gold is None or identity is None:
        return "--"
    d = (gold - identity) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}"

# ── build LaTeX table ─────────────────────────────────────────────────────────
# Columns: Arch | Identity | Gold@5% | Δ@5% | Gold@20% | Δ@20%
col_spec = r"ll r rr rr"

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Gold Model test accuracy (\%) across architectures and datasets",
    r"         at 5\% and 20\% forget-set size.",
    r"         Original shows the accuracy of the unmodified model (upper bound).",
    r"         $\Delta$ is the accuracy change relative to Original (negative = drop).}",
    r"\label{tab:goldmodel_accuracy}",
    r"\small",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{@{}" + col_spec + r"@{}}",
    r"\toprule",
    r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Architecture} & \multirow{2}{*}{Original} "
    r"& \multicolumn{2}{c}{Gold Model (5\%)} & \multicolumn{2}{c}{Gold Model (20\%)} \\",
    r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r" & & & Acc & $\Delta$ & Acc & $\Delta$ \\",
    r"\midrule",
]

for di, ds in enumerate(DATASETS):
    ds_data = data.get(ds, {})
    archs_present = [a for a in ARCHS if a in ds_data]
    if not archs_present:
        continue
    if di > 0:
        lines.append(r"\midrule")
    for ai, arch in enumerate(archs_present):
        row = ds_data[arch]
        identity = row.get("identity")
        gold5    = row.get(5)
        gold20   = row.get(20)
        ds_cell  = r"\multirow{" + str(len(archs_present)) + r"}{*}{" + ds + r"}" if ai == 0 else ""
        lines.append(
            f"{ds_cell} & {arch} & {fmt_acc(identity)}"
            f" & {fmt_acc(gold5)} & {fmt_delta(gold5, identity)}"
            f" & {fmt_acc(gold20)} & {fmt_delta(gold20, identity)}"
            r" \\"
        )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

latex = "\n".join(lines)
print("\n" + "=" * 70)
print(latex)
print("=" * 70)

out_path = os.path.join(OUTPUT_DIR, "goldmodel_accuracy_table.tex")
with open(out_path, "w") as f:
    f.write(latex + "\n")
print(f"\nSaved to {out_path}")
