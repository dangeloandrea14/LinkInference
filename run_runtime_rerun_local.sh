#!/usr/bin/env bash
# run_runtime_rerun_local.sh
# Re-runs CGU and ScaleGUN on SGCCGU for Cora, Citeseer, Pubmed (5% forget set).
# Results are written to output/runs/LinkAttack/edge/{Dataset}_SGCCGU_5.json.
# Usage: bash run_runtime_rerun_local.sh [--dry-run]

cd "$(dirname "$0")"
mkdir -p output/runs/runtime_rerun
export PYTORCH_ENABLE_MPS_FALLBACK=1

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

run "configs/benchmark/runtime_rerun/Cora_SGCCGU_5.jsonc"
run "configs/benchmark/runtime_rerun/Citeseer_SGCCGU_5.jsonc"

echo ""
echo "=== $(date '+%H:%M:%S') runtime_rerun done. Completed: $COUNT ==="
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "Errors (${#ERRORS[@]}):"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
    exit 1
fi
