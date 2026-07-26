"""Generate the forget-set-strategy configs (Reviewer Vdzo Q3).

Grid: 2 datasets x 4 forget sets x 2 seeds = 16 runs, GCN, all 16 unlearner entries.

    AmazonPhotos   random | easy | hard | group_intra       (cohort deletes its links)
    ogbn-arxiv     random | easy | hard | temporal_window   (one year deleted)

Each config is its EdgeUnbench counterpart with exactly one thing changed in the data
block -- the `forget`/`retain` splitter -- plus the evaluator measure list, the SaveValues
path, the seed and the predictor alias. Predictor, unlearners and every other partition are
copied verbatim, so results stay comparable to Table 3.

    python generate_forget_strategy_configs.py

Writes to configs/benchmark/forget_strategies/. Existing benchmark configs are read only.

Cohort ids and the temporal window are PINNED here rather than resolved at run time, so a
run cannot drift if a dataset is reprocessed. Regenerate them with:
    python rebuttal/forget_strategies/verify_node_year.py
    python rebuttal/forget_strategies/verify_splitters.py
"""

import json
import os
import shutil

from jsonc_parser.parser import JsoncParser

SRC_DIR = "configs/benchmark/EdgeUnbench"
OUT_DIR = "configs/benchmark/forget_strategies"
RUN_DIR = "output/rebuttal/forget_strategies/runs"

SPLITTER = "erasure.data.datasets.DataSplitterGraph"
PERCENTAGE = 0.05

# Cohort chosen by select='nearest' = the smallest cohort that still funds the 5% budget,
# so the forget set is as close as possible to a *complete* cohort deletion.
#   AmazonPhotos class 0: 5,953 intra edges vs a 5,954 budget -> deleted in full.
COHORT = {
    "AmazonPhotos": {"intra": 0, "incident": 0},
    "DBLP": {"intra": 2, "incident": 2},
    "Flickr": {"intra": 4, "incident": 5},
    "ogbn-arxiv": {"intra": 30, "incident": 10},
}

# On ogbn-arxiv each of 2016-2020 holds more than the 5% budget (57,889 edges), so a
# one-year window needs no padding. 2018 holds 246,382 and is not the truncated final year.
ARXIV_WINDOW = 2018

SEEDS = [0, 1]

# (dataset, strategies, seeds, with_linkteller, ls_ratio, device)
#
# easy/hard are NOT here: the paper already ran them and those results are reused from
# output/runs/EdgeUnbench/{AmazonPhotos,ogbn-arxiv}_GCN_{easy,hard}.json.
#
# `random` IS here, because no existing run covers it. The nearest thing,
# configs/benchmark/table3/*_GCN_5, sets shuffle:false and therefore takes the
# lowest-node-id 5% rather than a random sample, and it has no arxiv counterpart.
#
# CAVEAT to carry into the tables: the reused easy/hard runs come from the A100 cluster,
# these will not. LinkTeller's absolute AUC is backend-dependent
# (HANDOVER_adversary_ablation.md sec. 4), so the LinkTeller column mixes two machines and
# must be read relatively. Link Stealing, E-UMIA and accuracy are backend-stable.
#
# device=None leaves the backend to the framework, i.e. CUDA on the cluster. Do not pin it
# here; if you ever do, clear resources/cached/*fstrat* too, because Saveable keys the
# model cache on the alias and `device` is not part of the alias.
GRIDS = [
    ("AmazonPhotos", ["random", "group_intra"],     SEEDS, True, None, None),
    ("ogbn-arxiv",   ["random", "temporal_window"], SEEDS, True, None, None),
]


def splitter(strategy, dataset):
    """The `forget`/`retain` partition block for one strategy."""
    base = {"parts_names": ["forget", "retain"], "percentage": PERCENTAGE, "ref_data": "all"}
    cohort = COHORT.get(dataset, {})

    if strategy == "random":
        return {"class": f"{SPLITTER}.DataSplitterEdgeDifficulty",
                "parameters": dict(base, mode="simple")}
    if strategy in ("easy", "hard"):
        return {"class": f"{SPLITTER}.DataSplitterEdgeDifficulty",
                "parameters": dict(base, mode=strategy)}
    if strategy == "group_intra":
        return {"class": f"{SPLITTER}.DataSplitterEdgeGroup",
                "parameters": dict(base, mode="intra", group_attr="y", group=cohort["intra"])}
    if strategy == "group_incident":
        return {"class": f"{SPLITTER}.DataSplitterEdgeGroup",
                "parameters": dict(base, mode="incident", group_attr="y",
                                   group=cohort["incident"])}
    if strategy == "user_random":
        return {"class": f"{SPLITTER}.DataSplitterEdgeUserDeletion",
                "parameters": dict(base, mode="random")}
    if strategy == "user_group":
        return {"class": f"{SPLITTER}.DataSplitterEdgeUserDeletion",
                "parameters": dict(base, mode="random", group_attr="y",
                                   group=cohort["incident"])}
    if strategy in ("temporal_recent", "temporal_oldest"):
        return {"class": f"{SPLITTER}.DataSplitterEdgeTemporal",
                "parameters": dict(base, mode=strategy.split("_")[1],
                                   time_attr="node_year", edge_time="max")}
    if strategy == "temporal_window":
        return {"class": f"{SPLITTER}.DataSplitterEdgeTemporal",
                "parameters": dict(base, mode="window", time_attr="node_year",
                                   edge_time="max", window=ARXIV_WINDOW)}
    raise ValueError(f"unknown strategy {strategy!r}")


def measures_for(dataset, strategy, seed, with_linkteller, ls_ratio):
    """Utility metrics, then the three link inference attacks, then SaveValues.

    Every attack runs at the submission's own threat model (graph_knowledge='retain'),
    because the axis under study here is the forget set, not the adversary.
    """
    out = [
        {"class": "erasure.evaluations.running.RunTime"},
        {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
         "parameters": {"partition": "test", "target": "unlearned"}},
        {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
         "parameters": {"partition": "forget", "target": "unlearned"}},
        # on_graph:False is the accuracy key compute_stats.py uses for the paper's F-test.
        {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
         "parameters": {"partition": "test", "target": "unlearned", "unlearned_graph": False}},
        {"compose_umia": "configs/snippets/e_umia_graph.json"},
    ]

    ls_params = {"target": "unlearned", "graph_knowledge": "retain"}
    if ls_ratio is not None:
        ls_params["ratio"] = ls_ratio
    out.append({"class": "erasure.evaluations.link_stealing_attack."
                         "link_stealing_attack_0.LinkStealing0",
                "parameters": ls_params})

    if with_linkteller:
        out.append({"class": "erasure.evaluations.LinkTeller.LinkTeller.LinkTeller",
                    "parameters": {"target": "unlearn", "graph_knowledge": "retain"}})

    name = f"{dataset}_GCN_{strategy}_seed{seed}"
    out.append({"class": "erasure.evaluations.measures.SaveValues",
                "parameters": {"path": f"{RUN_DIR}/{name}.json"}})
    return out


def main():
    # Wipe first: a stale config from an earlier grid would otherwise be submitted too.
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    written = []

    for dataset, strategies, seeds, with_lt, ls_ratio, device in GRIDS:
        src = f"{SRC_DIR}/{dataset}_GCN_easy.jsonc"
        if not os.path.exists(src):
            print(f"  SKIP (no source config): {src}")
            continue

        for strategy in strategies:
            for seed in seeds:
                cfg = JsoncParser.parse_file(src)

                partitions = cfg["data"]["parameters"]["partitions"]
                forget_at = [i for i, p in enumerate(partitions)
                             if p["parameters"].get("parts_names", [None])[0] == "forget"]
                if len(forget_at) != 1:
                    raise RuntimeError(f"{src}: expected one 'forget' splitter, "
                                       f"found {len(forget_at)}")
                partitions[forget_at[0]] = splitter(strategy, dataset)

                cfg["evaluator"]["parameters"]["measures"] = measures_for(
                    dataset, strategy, seed, with_lt, ls_ratio)
                cfg["globals"] = dict(cfg.get("globals", {}), seed=seed)
                # Distinct alias per strategy and seed: the cached Original/Gold models are
                # specific to a forget set, so sharing them would reuse the wrong reference.
                cfg["predictor"]["parameters"]["alias"] = (
                    f"{cfg['predictor']['parameters']['alias']}_fstrat_{strategy}_seed{seed}")
                if device is not None:
                    cfg["predictor"]["parameters"]["device"] = device

                dst = f"{OUT_DIR}/{dataset}_GCN_{strategy}_seed{seed}.jsonc"
                with open(dst, "w") as fh:
                    json.dump(cfg, fh, indent=4)
                written.append(dst)
                print(f"  wrote {dst}")

    print(f"\n{len(written)} configs written to {OUT_DIR}/")
    print(f"Results will land in {RUN_DIR}/")
    print("\nNext:  ./submit_forget_strategies_cluster.sh --check")


if __name__ == "__main__":
    main()
