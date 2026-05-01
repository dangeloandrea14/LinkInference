#!/usr/bin/env bash
# run_edgeunbench.sh
# Runs all EdgeUnbench benchmark configs: 5 datasets × 7 architectures × hard split,
# plus GCN and GIN easy splits for each dataset.
# Hard configs are submitted first; easy configs depend on the corresponding hard job
# so the model cache is always populated before the easy variant starts.
# Usage: bash run_edgeunbench.sh

set -e
cd "$(dirname "$0")"

mkdir -p output/runs/EdgeUnbench

submit() {
    # Submits a config and prints the job id.
    local cfg="$1"
    local dep="$2"
    local dep_flag=""
    [ -n "$dep" ] && dep_flag="--dependency=afterok:${dep}"
    local out
    out=$(sbatch $dep_flag launchers/new_launcher_fast.sh main.py "$cfg")
    echo "$out"
    echo "$out" | awk '{print $NF}'   # return job id
}

DATASETS=(AmazonPhotos Flickr Reddit RomanEmpire ogbn-arxiv)
ARCHS_HARD_ONLY=(GAT GraphSAGE SGC MLP SGC_CGU)
ARCHS_WITH_EASY=(GCN GIN)

total=0

for ds in "${DATASETS[@]}"; do
    # Archs that only have a hard split — submit immediately
    for arch in "${ARCHS_HARD_ONLY[@]}"; do
        cfg="configs/benchmark/EdgeUnbench/${ds}_${arch}_hard.jsonc"
        echo "Submitting $cfg ..."
        submit "$cfg" > /dev/null
        (( total++ ))
    done

    # Archs with easy split — hard first, easy depends on hard job
    for arch in "${ARCHS_WITH_EASY[@]}"; do
        hard_cfg="configs/benchmark/EdgeUnbench/${ds}_${arch}_hard.jsonc"
        easy_cfg="configs/benchmark/EdgeUnbench/${ds}_${arch}_easy.jsonc"

        echo "Submitting $hard_cfg ..."
        job_id=$(submit "$hard_cfg")
        (( total++ ))

        echo "Submitting $easy_cfg (depends on job $job_id) ..."
        submit "$easy_cfg" "$job_id" > /dev/null
        (( total++ ))
    done
done

echo ""
echo "All EdgeUnbench experiments submitted ($total jobs)."
