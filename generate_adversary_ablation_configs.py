"""Generate the adversary-knowledge ablation configs for the NeurIPS rebuttal.

Each config is its EdgeUnbench counterpart with two things changed: the evaluator's
measure list, and the SaveValues output path.  Data, predictor and unlearner blocks are
copied verbatim so the ablation stays comparable to Table 3 of the submission.

    python generate_adversary_ablation_configs.py

Writes to configs/benchmark/adversary_ablation/.  Existing benchmark configs are read
only, never modified.
"""

import json
import os

from jsonc_parser.parser import JsoncParser

SRC_DIR = "configs/benchmark/EdgeUnbench"
OUT_DIR = "configs/benchmark/adversary_ablation"
RUN_DIR = "output/rebuttal/adversary_ablation/runs"

DATASETS = ["AmazonPhotos", "DBLP", "Flickr"]
SETTINGS = ["easy", "hard"]          # easy = low centrality, hard = high centrality
SEEDS = [0, 1]

# The knowledge ladder. `tag` becomes part of the result key, so every rung lands in its
# own entry of the flat result dict instead of overwriting the previous one.
LADDER = [
    # (tag,        knowledge,  fraction)
    ("k=none",     "none",     1.0),
    ("k=p25",      "partial",  0.25),
    ("k=p50",      "partial",  0.50),
    ("k=retain",   "retain",   1.0),   # the submission's threat model
    ("k=oracle",   "oracle",   1.0),   # upper bound: forgotten edges still present
]

LABELLED_SIZES = [100, 1000, 5000]     # adversary's labelled edge budget (attack-3)


def measures_for(dataset, setting, seed):
    """Build the evaluator measure list: utility metrics, then the attack ladder."""
    out = [
        {"class": "erasure.evaluations.running.RunTime"},
        # Utility arm -- the comparison side of the F-test at every knowledge level.
        {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
         "parameters": {"partition": "test", "target": "unlearned"}},
        {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
         "parameters": {"partition": "forget", "target": "unlearned"}},
        # on_graph:False is the accuracy key compute_stats.py uses for the paper's
        # F-test; kept so the ablation's utility arm is directly comparable.
        {"class": "erasure.evaluations.measures.TorchSKLearnGraph",
         "parameters": {"partition": "test", "target": "unlearned", "unlearned_graph": False}},
        {"compose_umia": "configs/snippets/e_umia_graph.json"},
    ]

    for tag, knowledge, fraction in LADDER:
        common = {"graph_knowledge": knowledge, "knowledge_fraction": fraction,
                  "knowledge_seed": 42, "tag": tag}
        out.append({"class": "erasure.evaluations.LinkTeller.LinkTeller.LinkTeller",
                    "parameters": dict(common, target="unlearn")})
        out.append({"class": "erasure.evaluations.link_stealing_attack."
                             "link_stealing_attack_0.LinkStealing0",
                    "parameters": dict(common, target="unlearned")})

    # Attack variant: harder negatives (2-hop, degree-matched, >=2 shared neighbours),
    # at the submission's own knowledge level.
    out.append({"class": "erasure.evaluations.LinkTeller.LinkTeller.LinkTeller",
                "parameters": {"target": "unlearn", "edge_sampler": "bfs+",
                               "graph_knowledge": "retain", "tag": "k=retain+hardneg"}})

    # Supervised link stealing: the adversary also holds labelled edge/non-edge pairs.
    for n in LABELLED_SIZES:
        out.append({"class": "erasure.evaluations.link_stealing_attack."
                             "link_stealing_attack_3.LinkStealingSupervised",
                    "parameters": {"target": "unlearned", "n_labelled": n,
                                   "graph_knowledge": "retain", "knowledge_seed": 42,
                                   "tag": f"k=retain+lab{n}"}})

    name = f"{dataset}_GCN_{setting}_seed{seed}"
    out.append({"class": "erasure.evaluations.measures.SaveValues",
                "parameters": {"path": f"{RUN_DIR}/{name}.json"}})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = 0

    for dataset in DATASETS:
        for setting in SETTINGS:
            src = f"{SRC_DIR}/{dataset}_GCN_{setting}.jsonc"
            if not os.path.exists(src):
                print(f"  SKIP (no source config): {src}")
                continue

            for seed in SEEDS:
                cfg = JsoncParser.parse_file(src)
                cfg["evaluator"]["parameters"]["measures"] = measures_for(dataset, setting, seed)
                cfg["globals"] = dict(cfg.get("globals", {}), seed=seed)
                # Distinct alias per seed so cached models are not shared across seeds.
                cfg["predictor"]["parameters"]["alias"] = (
                    f"{cfg['predictor']['parameters']['alias']}_advabl_seed{seed}")

                dst = f"{OUT_DIR}/{dataset}_GCN_{setting}_seed{seed}.jsonc"
                with open(dst, "w") as fh:
                    json.dump(cfg, fh, indent=4)
                print(f"  wrote {dst}")
                written += 1

    print(f"\n{written} configs written to {OUT_DIR}/")
    print(f"Results will land in {RUN_DIR}/")


if __name__ == "__main__":
    main()
