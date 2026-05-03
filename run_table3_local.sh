#!/usr/bin/env bash
# run_table3_local.sh
# Runs Table 3 configs (Identity + GoldModel, accuracy only) for all missing
# architecture × dataset × forget-% combinations.
# GCN results already exist in output/runs/dataset_selection/.
# Usage: bash run_table3_local.sh [--dry-run]

cd "$(dirname "$0")"
mkdir -p output/runs/table3

DRY_RUN=0
[[ "$1" == "--dry-run" ]] && DRY_RUN=1

ERRORS=()
COUNT=0

run() {
    local cfg="$1"
    if [[ ! -f "$cfg" ]]; then
        echo "  [SKIP] config not found: $cfg"
        return
    fi
    echo ""
    echo "=== $(date '+%H:%M:%S') Running: $cfg ==="
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [DRY-RUN] would run: python main.py $cfg"
        return
    fi
    python main.py "$cfg"
    local status=$?
    COUNT=$((COUNT + 1))
    if [[ $status -ne 0 ]]; then
        ERRORS+=("$cfg (exit $status)")
        echo "  [ERROR] $cfg failed with exit $status"
    fi
}

DATASETS=(AmazonPhotos AmazonComputers DBLP Flickr ogbn_arxiv)
ARCHS=(GIN GAT GraphSAGE SGC SGC_CGU)
PCTS=(5 20)

for ds in "${DATASETS[@]}"; do
    for arch in "${ARCHS[@]}"; do
        for pct in "${PCTS[@]}"; do
            run "configs/benchmark/table3/${ds}_${arch}_${pct}.jsonc"
        done
    done
done

# Pubmed: only SGC_CGU 20% is missing
run "configs/benchmark/table3/Pubmed_SGC_CGU_20.jsonc"

echo ""
echo "=== $(date '+%H:%M:%S') table3 runs done. Completed: $COUNT ==="
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "Errors (${#ERRORS[@]}):"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
    exit 1
fi
