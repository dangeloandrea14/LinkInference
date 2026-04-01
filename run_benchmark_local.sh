#!/bin/bash
# Run benchmark configs sequentially with plain python.
# Must be run from the project root directory.
#
# Usage:
#   ./run_benchmark_local.sh              # run all missing configs
#   ./run_benchmark_local.sh --dry-run    # print what would run without running

CONFIG_DIR="configs/benchmark/edge_removal"
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No configs will be executed."
fi

mkdir -p output/logs

ran=0
skipped=0
errors=0
failed=0

while IFS= read -r -d '' config; do
    output_path=$(grep -o '"path":"output/runs/LinkAttack/edge/[^"]*\.json"' "$config" \
                  | sed 's/"path":"//;s/"$//')

    if [[ -z "$output_path" ]]; then
        echo "WARN  no SaveValues path found in: $config"
        ((errors++))
        continue
    fi

    if [[ -f "$output_path" ]]; then
        echo "SKIP  $output_path"
        ((skipped++))
        continue
    fi

    echo "RUN   $config -> $output_path"
    if [[ "$DRY_RUN" == false ]]; then
        log_file="output/logs/$(basename "${config%.jsonc}").log"
        PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py "$config" 2>&1 | tee "$log_file"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "FAIL  $config (see $log_file)"
            ((failed++))
        else
            ((ran++))
        fi
    else
        ((ran++))
    fi

done < <(find "$CONFIG_DIR" -name "*.jsonc" -print0 | sort -z)

echo ""
echo "Done. Ran: $ran  |  Skipped (already done): $skipped  |  Failed: $failed  |  Warnings: $errors"
