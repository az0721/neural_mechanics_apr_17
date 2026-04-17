#!/bin/bash
#SBATCH --job-name=auto_st2
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/auto_st2_%j.out
#SBATCH --error=logs/auto_st2_%j.err

# ══════════════════════════════════════════════════════════════════════
# Auto-submitter for steering Phase 2b
# Partitions: gpu (8h, max 8) + multigpu (8h, max 8) = 16 slots
# GPU: h200 only
# Output: per_user_v2/
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7

SCRIPTS_DIR="scripts/steering_jobs_v2"
PERUSER_DIR="outputs_v7/steering/per_user_v2"
MAX_GPU=8
MAX_MULTI=8
INTERVAL=90

mkdir -p "$PERUSER_DIR" logs

echo "$(date) | Steering Phase 2b Auto-Submitter"
echo "$(date) | Partitions: gpu($MAX_GPU) multigpu($MAX_MULTI)"
echo "$(date) | Output: $PERUSER_DIR"
echo "==========================================================="

while true; do
    QCACHE=$(squeue -u "$USER" -h -o "%.18i %.200j %.8T %.10M %.10l %P" 2>/dev/null)

    # Count running st2_ jobs per partition
    n_gpu=$(echo "$QCACHE" | grep "st2_" | grep " gpu " 2>/dev/null | wc -l)
    n_multi=$(echo "$QCACHE" | grep "st2_" | grep "multigpu" 2>/dev/null | wc -l)
    n_gpu=${n_gpu:-0}
    n_multi=${n_multi:-0}
    avail_gpu=$((MAX_GPU - n_gpu))
    avail_multi=$((MAX_MULTI - n_multi))
    [ "$avail_gpu" -lt 0 ] && avail_gpu=0
    [ "$avail_multi" -lt 0 ] && avail_multi=0

    # Count done
    n_done=$(find "$PERUSER_DIR" -name "*.json" 2>/dev/null | wc -l)
    n_total=$(ls "$SCRIPTS_DIR"/*.sbatch 2>/dev/null | wc -l)

    # Find fresh (not done, not running)
    n_submitted=0
    for script in "$SCRIPTS_DIR"/*.sbatch; do
        [ ! -f "$script" ] && continue
        uid=$(basename "$script" .sbatch | sed 's/^st2_//')
        # Check if done
        if find "$PERUSER_DIR" -name "${uid}*.json" 2>/dev/null | grep -q .; then
            continue
        fi
        # Check if running
        if echo "$QCACHE" | grep -q "st2_${uid}"; then
            continue
        fi
        # Submit to partition with availability
        if [ "$avail_gpu" -gt 0 ]; then
            jobid=$(sbatch --partition=gpu "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $uid → gpu → $jobid"
            avail_gpu=$((avail_gpu - 1))
            n_submitted=$((n_submitted + 1))
        elif [ "$avail_multi" -gt 0 ]; then
            jobid=$(sbatch --partition=multigpu "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $uid → multigpu → $jobid"
            avail_multi=$((avail_multi - 1))
            n_submitted=$((n_submitted + 1))
        else
            break
        fi
    done

    n_running=$((n_gpu + n_multi + n_submitted))
    echo "$(date) | Done:$n_done/$n_total Run:$n_running | gpu=$avail_gpu multi=$avail_multi"

    # Exit if all done
    if [ "$n_done" -ge "$n_total" ] && [ "$n_running" -eq 0 ]; then
        echo "$(date) | ALL DONE"
        break
    fi

    sleep "$INTERVAL"
done
