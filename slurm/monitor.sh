#!/bin/bash
# Watch ablation progress: formats new CSV lines as they arrive.
# Usage: bash slurm/monitor.sh

CSV="results/benchmark_ablation_naive.csv"
TOTAL=1600

if [ ! -f "$CSV" ]; then
    echo "Waiting for $CSV to appear..."
    while [ ! -f "$CSV" ]; do sleep 1; done
fi

HEADER=$(head -1 "$CSV")
DONE=$(($(wc -l < "$CSV") - 1))
echo "Progress: ${DONE}/${TOTAL} runs completed"
echo "========================================"

tail -n 0 -f "$CSV" | while IFS=, read -r method t_value image true_label iterations success adv_class switch_iter locked_class timestamp; do
    DONE=$(($(wc -l < "$CSV") - 1))
    [ "$success" = "True" ] && status="OK" || status="FAIL"
    extra=""
    [ -n "$switch_iter" ] && extra=" (switch@${switch_iter}, locked=${locked_class})"
    printf "[%d/%d] %s T=%-3s | %s | %s iters | %s%s\n" \
        "$DONE" "$TOTAL" "$method" "$t_value" "$image" "$iterations" "$status" "$extra"
done
