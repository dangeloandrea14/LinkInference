#!/usr/bin/env bash
# run_edgeunbench.sh
# Runs all EdgeUnbench benchmark configs: 5 datasets × 7 architectures × hard split,
# plus GIN easy split for each dataset.
# Usage: bash run_edgeunbench.sh

set -e
cd "$(dirname "$0")"

mkdir -p output/runs/EdgeUnbench

CONFIGS=(
    # AmazonPhotos
    configs/benchmark/EdgeUnbench/AmazonPhotos_GCN_hard.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_GIN_hard.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_GIN_easy.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_GAT_hard.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_GraphSAGE_hard.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_SGC_hard.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_MLP_hard.jsonc
    configs/benchmark/EdgeUnbench/AmazonPhotos_SGC_CGU_hard.jsonc

    # Flickr
    configs/benchmark/EdgeUnbench/Flickr_GCN_hard.jsonc
    configs/benchmark/EdgeUnbench/Flickr_GIN_hard.jsonc
    configs/benchmark/EdgeUnbench/Flickr_GIN_easy.jsonc
    configs/benchmark/EdgeUnbench/Flickr_GAT_hard.jsonc
    configs/benchmark/EdgeUnbench/Flickr_GraphSAGE_hard.jsonc
    configs/benchmark/EdgeUnbench/Flickr_SGC_hard.jsonc
    configs/benchmark/EdgeUnbench/Flickr_MLP_hard.jsonc
    configs/benchmark/EdgeUnbench/Flickr_SGC_CGU_hard.jsonc

    # Reddit
    configs/benchmark/EdgeUnbench/Reddit_GCN_hard.jsonc
    configs/benchmark/EdgeUnbench/Reddit_GIN_hard.jsonc
    configs/benchmark/EdgeUnbench/Reddit_GIN_easy.jsonc
    configs/benchmark/EdgeUnbench/Reddit_GAT_hard.jsonc
    configs/benchmark/EdgeUnbench/Reddit_GraphSAGE_hard.jsonc
    configs/benchmark/EdgeUnbench/Reddit_SGC_hard.jsonc
    configs/benchmark/EdgeUnbench/Reddit_MLP_hard.jsonc
    configs/benchmark/EdgeUnbench/Reddit_SGC_CGU_hard.jsonc

    # RomanEmpire
    configs/benchmark/EdgeUnbench/RomanEmpire_GCN_hard.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_GIN_hard.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_GIN_easy.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_GAT_hard.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_GraphSAGE_hard.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_SGC_hard.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_MLP_hard.jsonc
    configs/benchmark/EdgeUnbench/RomanEmpire_SGC_CGU_hard.jsonc

    # ogbn-arxiv
    configs/benchmark/EdgeUnbench/ogbn-arxiv_GCN_hard.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_GIN_hard.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_GIN_easy.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_GAT_hard.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_GraphSAGE_hard.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_SGC_hard.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_MLP_hard.jsonc
    configs/benchmark/EdgeUnbench/ogbn-arxiv_SGC_CGU_hard.jsonc
)

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    echo "[$((i+1))/$TOTAL] Submitting $cfg ..."
    sbatch launchers/new_launcher_fast.sh main.py "$cfg"
    echo "  submitted."
done

echo "All EdgeUnbench experiments submitted ($TOTAL jobs)."
