#!/usr/bin/env bash
# run_edgeunbench_missing_local.sh
# Runs the remaining EdgeUnbench configs for AmazonPhotos and Flickr that are
# not yet in output/runs/EdgeUnbench/ (SGC_CGU runs assumed in-flight separately).
# Usage: bash run_edgeunbench_missing_local.sh

set -e
cd "$(dirname "$0")"

mkdir -p output/runs/EdgeUnbench

run() {
    local cfg="$1"
    echo ""
    echo "=== $(date '+%H:%M:%S') Running: $cfg ==="
    python main.py "$cfg"
}

# AmazonPhotos — GAT and GraphSAGE (hard-only, no easy split exists for these)
run configs/benchmark/EdgeUnbench/AmazonPhotos_GAT_hard.jsonc
run configs/benchmark/EdgeUnbench/AmazonPhotos_GraphSAGE_hard.jsonc

# Flickr — GIN easy (hard result already exists, so cache is ready)
run configs/benchmark/EdgeUnbench/Flickr_GIN_easy.jsonc

# Flickr — GAT and GraphSAGE (hard-only)
run configs/benchmark/EdgeUnbench/Flickr_GAT_hard.jsonc
run configs/benchmark/EdgeUnbench/Flickr_GraphSAGE_hard.jsonc

echo ""
echo "All missing local EdgeUnbench experiments done ($(date '+%H:%M:%S'))."
