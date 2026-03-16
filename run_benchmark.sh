#!/bin/bash
# Submit one SLURM job per benchmark config whose output file does not yet exist.
# Must be run from the project root directory.
#
# Usage:
#   ./run_benchmark.sh              # submit all missing jobs
#   ./run_benchmark.sh --dry-run    # print what would be submitted without submitting

CONFIG_DIR="configs/benchmark/edge_removal"
LAUNCHER="new_launcher_fast.sh"
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No jobs will be submitted."
fi

mkdir -p output/logs

submitted=0
skipped=0
errors=0

while IFS= read -r -d '' config; do
    # Extract the SaveValues output path from the config
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
    else
        echo "SUBMIT $config -> $output_path"
        if [[ "$DRY_RUN" == false ]]; then
            sbatch "$LAUNCHER" main.py "$config"
        fi
        ((submitted++))
    fi

done < <(find "$CONFIG_DIR" -name "*.jsonc" -print0 | sort -z)

echo ""
echo "Done. Submitted: $submitted  |  Skipped (already done): $skipped  |  Warnings: $errors"
