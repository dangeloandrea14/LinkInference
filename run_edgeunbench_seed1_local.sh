#!/usr/bin/env bash
# run_edgeunbench_seed1_local.sh
# Runs seed=1 EdgeUnbench configs for AmazonComputers, AmazonPhotos, DBLP, Flickr.
# Architectures: GCN and GIN (SGC_CGU omitted — handled separately).
# Hard configs run before easy so the model cache is populated first.
# Intended for the second (light) machine.
# Usage: bash run_edgeunbench_seed1_local.sh

set -e
cd "$(dirname "$0")"

mkdir -p output/runs/EdgeUnbench_seed1

run() {
    local cfg="$1"
    echo ""
    echo "=== $(date '+%H:%M:%S') Running: $cfg ==="
    python main.py "$cfg"
}

DATASETS=(AmazonComputers AmazonPhotos DBLP Flickr)

for ds in "${DATASETS[@]}"; do
    for arch in GCN GIN; do
        run "configs/benchmark/EdgeUnbench_seed1/${ds}_${arch}_hard.jsonc"
        run "configs/benchmark/EdgeUnbench_seed1/${ds}_${arch}_easy.jsonc"
    done
done

echo ""
echo "All seed=1 EdgeUnbench experiments done ($(date '+%H:%M:%S'))."
