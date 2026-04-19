#!/usr/bin/env bash
# run_dataset_selection.sh
# Runs Identity + Gold Model at 1%, 5%, 20% for all candidate datasets.
# Usage: bash run_dataset_selection.sh [optional: single config path]

set -e
cd "$(dirname "$0")"

mkdir -p output/runs/dataset_selection

CONFIGS=(
    configs/dataset_selection/RomanEmpire/RomanEmpire_GCN_1.jsonc
    configs/dataset_selection/RomanEmpire/RomanEmpire_GCN_5.jsonc
    configs/dataset_selection/RomanEmpire/RomanEmpire_GCN_20.jsonc

    configs/dataset_selection/AmazonRatings/AmazonRatings_GCN_1.jsonc
    configs/dataset_selection/AmazonRatings/AmazonRatings_GCN_5.jsonc
    configs/dataset_selection/AmazonRatings/AmazonRatings_GCN_20.jsonc

    configs/dataset_selection/Minesweeper/Minesweeper_GCN_1.jsonc
    configs/dataset_selection/Minesweeper/Minesweeper_GCN_5.jsonc
    configs/dataset_selection/Minesweeper/Minesweeper_GCN_20.jsonc

    configs/dataset_selection/Flickr/Flickr_GCN_1.jsonc
    configs/dataset_selection/Flickr/Flickr_GCN_5.jsonc
    configs/dataset_selection/Flickr/Flickr_GCN_20.jsonc

    configs/dataset_selection/Penn94/Penn94_GCN_1.jsonc
    configs/dataset_selection/Penn94/Penn94_GCN_5.jsonc
    configs/dataset_selection/Penn94/Penn94_GCN_20.jsonc

    configs/dataset_selection/ogbn-arxiv/ogbn_arxiv_GCN_1.jsonc
    configs/dataset_selection/ogbn-arxiv/ogbn_arxiv_GCN_5.jsonc
    configs/dataset_selection/ogbn-arxiv/ogbn_arxiv_GCN_20.jsonc
)

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    echo "[$((i+1))/$TOTAL] Submitting $cfg ..."
    sbatch launchers/new_launcher_fast.sh main.py "$cfg"
    echo "  submitted."
done

echo "All dataset selection experiments complete."
