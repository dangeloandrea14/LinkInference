"""
infected_nodes_table.py  —  Section 4.1 infected-nodes statistics

Loads the real cached PyG graphs, runs infected_nodes (the same BFS used
in GraphUnlearner) at every forget-set percentage used by the benchmark,
and emits a LaTeX table showing what fraction of the graph is "infected"
under k-hop neighbourhood expansion.
"""

import os, time, random
import torch
import networkx as nx
import numpy as np

# ── dataset paths ─────────────────────────────────────────────────────────────
BASE = "resources/data"
DATASET_PATHS = {
    "Cora"     : os.path.join(BASE, "Cora/Cora/processed/data.pt"),
    "Citeseer" : os.path.join(BASE, "CiteSeer/CiteSeer/processed/data.pt"),
    "Coauthor" : os.path.join(BASE, "coauthor/CS/processed/data.pt"),
    "Pubmed"   : os.path.join(BASE, "PubMed/PubMed/processed/data.pt"),
}

FORGET_PCTS = [1, 5, 10, 20, 50]
HOPS        = [2, 3]          # 1-hidden-layer GCN → hops=2; 2-hidden-layer → hops=3
N_SEEDS     = 5               # average over this many random forget-set samples
SEED_BASE   = 42

OUTPUT_DIR  = "output/viz/LinkAttack/edge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── infected_nodes (identical to GraphUnlearner implementation) ───────────────
def infected_nodes(edge_list, forget_edges, hops):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    endpoints = set()
    for u, v in forget_edges:
        endpoints.add(u); endpoints.add(v)
    infected = set()
    for node in endpoints:
        if node in G:
            infected.update(
                nx.single_source_shortest_path_length(G, node, cutoff=hops).keys()
            )
    return infected

# ── load graph ────────────────────────────────────────────────────────────────
def load_graph(path):
    raw = torch.load(path, weights_only=False)
    # data.pt is a (dict, None, class) tuple; graph lives in raw[0]
    d  = raw[0] if isinstance(raw, tuple) else raw
    ei = d["edge_index"].t().tolist()            # directed edge list
    n_nodes = d["x"].shape[0]
    # deduplicate to undirected
    undirected = list({ (min(u,v), max(u,v)) for u, v in ei if u != v })
    return n_nodes, undirected

# ── compute stats ─────────────────────────────────────────────────────────────
rows = []   # (dataset, n_nodes, n_edges, pct, hops, mean_infected, std_infected, mean_time)

for ds_name, path in DATASET_PATHS.items():
    print(f"Loading {ds_name} ...", flush=True)
    n_nodes, edge_list = load_graph(path)
    n_edges = len(edge_list)
    print(f"  {n_nodes} nodes, {n_edges} undirected edges")

    for pct in FORGET_PCTS:
        n_forget = max(1, int(n_edges * pct / 100))

        for hops in HOPS:
            counts = []
            times  = []
            for seed in range(N_SEEDS):
                random.seed(SEED_BASE + seed)
                forget = random.sample(edge_list, n_forget)

                t0 = time.perf_counter()
                inf = infected_nodes(edge_list, forget, hops)
                times.append(time.perf_counter() - t0)
                counts.append(len(inf))

            rows.append({
                "dataset"   : ds_name,
                "n_nodes"   : n_nodes,
                "n_edges"   : n_edges,
                "pct"       : pct,
                "n_forget"  : n_forget,
                "hops"      : hops,
                "mean_inf"  : np.mean(counts),
                "std_inf"   : np.std(counts),
                "mean_time" : np.mean(times),
            })
            pct_inf = np.mean(counts) / n_nodes * 100
            print(f"  pct={pct:2}% hops={hops}  infected={np.mean(counts):.0f}/{n_nodes} "
                  f"({pct_inf:.1f}%)  t={np.mean(times):.4f}s")

# ── build LaTeX table ─────────────────────────────────────────────────────────
# Layout: one block per dataset.
# Columns: |V|  |E|  || for each pct: infected% (hops=2) / infected% (hops=3)

def fmt_pct(mean, n):
    p = mean / n * 100
    if p >= 99.9:
        return r"$\approx$100"
    return f"{p:.1f}"

lines = []
n_pct_cols = len(FORGET_PCTS)

# Column spec: l r r | (r r) * n_pcts
col_spec = "l rr " + "".join(["r" * len(HOPS) for _ in FORGET_PCTS])
# Build header
pct_headers = " & ".join(
    r"\multicolumn{" + str(len(HOPS)) + r"}{c}{" + f"{p}\\%" + "}"
    for p in FORGET_PCTS
)
hop_sub = " & ".join(
    " & ".join(f"$k={h}$" for h in HOPS)
    for _ in FORGET_PCTS
)

lines += [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Fraction of nodes (\%) infected by $k$-hop neighbourhood expansion",
    r"         when removing a random subset of edges. Values averaged over",
    f"         {N_SEEDS} random forget-set samples.",
    r"         Almost all nodes are infected even at small forget-set sizes,",
    r"         confirming that edge unlearning is structurally degenerate on",
    r"         feature-rich datasets.}",
    r"\label{tab:infected_nodes}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}" + col_spec + r"@{}}",
    r"\toprule",
    r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{$|V|$} & \multirow{2}{*}{$|E|$} & "
    + pct_headers + r" \\",
    r"\cmidrule(lr){4-" + str(3 + n_pct_cols * len(HOPS)) + r"}",
    r" & & & " + hop_sub + r" \\",
    r"\midrule",
]

# Group rows by dataset
from collections import defaultdict
by_ds = defaultdict(dict)
for r in rows:
    by_ds[r["dataset"]][(r["pct"], r["hops"])] = r

for ds_name in DATASET_PATHS:
    ds_rows = by_ds[ds_name]
    if not ds_rows:
        continue
    # get n_nodes, n_edges from first row
    first = next(iter(ds_rows.values()))
    n_nodes = first["n_nodes"]
    n_edges = first["n_edges"]

    cells = []
    for pct in FORGET_PCTS:
        for hops in HOPS:
            key = (pct, hops)
            if key in ds_rows:
                cells.append(fmt_pct(ds_rows[key]["mean_inf"], n_nodes))
            else:
                cells.append("--")

    line = (f"{ds_name} & {n_nodes:,} & {n_edges:,} & "
            + " & ".join(cells) + r" \\")
    lines.append(line)

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

latex = "\n".join(lines)
print("\n" + "="*70)
print(latex)
print("="*70)

out_path = os.path.join(OUTPUT_DIR, "infected_nodes_table.tex")
with open(out_path, "w") as f:
    f.write(latex + "\n")
print(f"\nSaved to {out_path}")
