#!/usr/bin/env bash
# run_edgeunbench_local.sh
# Runs EdgeUnbench configs for Pubmed, AmazonComputers, and DBLP locally (no SLURM).
# Hard configs run first so the model cache is populated before the easy variant starts.
# Usage: bash run_edgeunbench_local.sh

set -e
cd "$(dirname "$0")"

mkdir -p output/runs/EdgeUnbench

DATASETS=(Pubmed AmazonComputers DBLP)
ARCHS_HARD_ONLY=(GAT GraphSAGE SGC MLP SGC_CGU)
ARCHS_WITH_EASY=(GCN GIN)

run() {
    local cfg="$1"
    echo ""
    echo "=== $(date '+%H:%M:%S') Running: $cfg ==="
    python main.py "$cfg"
}

for ds in "${DATASETS[@]}"; do
    for arch in "${ARCHS_HARD_ONLY[@]}"; do
        run "configs/benchmark/EdgeUnbench/${ds}_${arch}_hard.jsonc"
    done
    for arch in "${ARCHS_WITH_EASY[@]}"; do
        run "configs/benchmark/EdgeUnbench/${ds}_${arch}_hard.jsonc"
        run "configs/benchmark/EdgeUnbench/${ds}_${arch}_easy.jsonc"
    done
done

echo ""
echo "All EdgeUnbench local experiments done ($(date '+%H:%M:%S'))."
