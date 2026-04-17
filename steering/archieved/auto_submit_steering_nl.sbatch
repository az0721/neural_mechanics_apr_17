#!/bin/bash
#SBATCH --job-name=auto_nl
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/auto_nl_%j.out
#SBATCH --error=logs/auto_nl_%j.err
# ══════════════════════════════════════════════════════════════════════
# Auto-submitter for NL-based steering experiments.
# Generates sbatch scripts, then loops submitting & monitoring.
#
# 8 jobs total: 2 iters (v7,v8) × 2 exps (exp2,exp1a) × 2 cfgs
#   v7: exp2 cfg5/6, exp1a cfg3/4
#   v8: exp2 cfg5/6, exp1a cfg3/4
#
# Partitions: gpu (8h, max 8) + multigpu (8h, max 8) = 16 slots
# Each job: ~35 min (well under 8h limit)
#
# Usage:
#   sbatch steering/auto_submit_steering_nl.sbatch
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

SCRIPTS_DIR="scripts/nl_steering_jobs"
MAX_GPU=8
MAX_MULTI=8
INTERVAL=60

mkdir -p logs "$SCRIPTS_DIR"

echo "$(date) | NL Steering Auto-Submitter"
echo "$(date) | Partitions: gpu($MAX_GPU) multigpu($MAX_MULTI)"
echo "==========================================================="

# Step 1: Generate sbatch scripts
echo "$(date) | Generating sbatch scripts..."
python -u steering/generate_steering_jobs_nl.py
echo "$(date) | Scripts generated in $SCRIPTS_DIR/"

n_total=$(ls "$SCRIPTS_DIR"/*.sbatch 2>/dev/null | wc -l)
echo "$(date) | Total jobs: $n_total"
echo "==========================================================="

# Step 2: Loop — submit and monitor
while true; do
    QCACHE=$(squeue -u "$USER" -h -o "%.18i %.200j %.8T %.10M %.10l %P" 2>/dev/null)

    n_gpu=$(echo "$QCACHE" | grep "nl_" | grep " gpu " 2>/dev/null | wc -l)
    n_multi=$(echo "$QCACHE" | grep "nl_" | grep "multigpu" 2>/dev/null | wc -l)
    n_gpu=${n_gpu:-0}
    n_multi=${n_multi:-0}
    avail_gpu=$((MAX_GPU - n_gpu))
    avail_multi=$((MAX_MULTI - n_multi))
    [ "$avail_gpu" -lt 0 ] && avail_gpu=0
    [ "$avail_multi" -lt 0 ] && avail_multi=0

    # Count completed jobs by checking output JSON files
    n_done=0
    for script in "$SCRIPTS_DIR"/*.sbatch; do
        [ ! -f "$script" ] && continue
        base=$(basename "$script" .sbatch)
        # Parse: nl_{iter}_{exp}_c{cfg} → outputs_{iter}/steering/nl_results/{exp}_cfg{cfg}_nl.json
        iter=$(echo "$base" | sed 's/nl_//' | cut -d'_' -f1)
        # Handle exp names with underscores: exp1a, exp2
        rest=$(echo "$base" | sed "s/nl_${iter}_//" | sed 's/_c[0-9]*$//')
        cfg=$(echo "$base" | grep -o 'c[0-9]*$' | sed 's/c//')
        outfile="outputs_${iter}/steering/nl_results/${rest}_cfg${cfg}_nl.json"
        if [ -f "$outfile" ]; then
            n_done=$((n_done + 1))
        fi
    done

    n_submitted=0
    for script in "$SCRIPTS_DIR"/*.sbatch; do
        [ ! -f "$script" ] && continue
        base=$(basename "$script" .sbatch)

        # Parse iter/exp/cfg
        iter=$(echo "$base" | sed 's/nl_//' | cut -d'_' -f1)
        rest=$(echo "$base" | sed "s/nl_${iter}_//" | sed 's/_c[0-9]*$//')
        cfg=$(echo "$base" | grep -o 'c[0-9]*$' | sed 's/c//')
        outfile="outputs_${iter}/steering/nl_results/${rest}_cfg${cfg}_nl.json"

        # Skip if output already exists
        if [ -f "$outfile" ]; then
            continue
        fi

        # Skip if already in queue
        if echo "$QCACHE" | grep -q "$base"; then
            continue
        fi

        # Skip v8 jobs if v8 hidden states don't exist yet
        if [ "$iter" = "v8" ]; then
            # Check if at least one v8 hidden state file exists
            v8_check=$(ls outputs_v8/hidden_states/gemma3_12b_it/${rest}_cfg${cfg}.npz 2>/dev/null | head -1)
            if [ -z "$v8_check" ]; then
                # V8 not ready yet — skip silently
                continue
            fi
        fi

        # Submit to available partition
        if [ "$avail_gpu" -gt 0 ]; then
            jobid=$(sbatch --partition=gpu "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → gpu → $jobid"
            avail_gpu=$((avail_gpu - 1))
            n_submitted=$((n_submitted + 1))
        elif [ "$avail_multi" -gt 0 ]; then
            jobid=$(sbatch --partition=multigpu "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → multigpu → $jobid"
            avail_multi=$((avail_multi - 1))
            n_submitted=$((n_submitted + 1))
        else
            break
        fi
    done

    n_running=$((n_gpu + n_multi + n_submitted))
    echo "$(date) | Done:$n_done/$n_total Run:$n_running Sub:$n_submitted | gpu_avail=$avail_gpu multi_avail=$avail_multi"

    # Check completion
    if [ "$n_done" -ge "$n_total" ] && [ "$n_running" -eq 0 ]; then
        echo "$(date) | ALL NL STEERING JOBS COMPLETE"
        echo "$(date) | Running analysis..."
        python -u steering/analyze_steering_nl.py
        echo "$(date) | Analysis done → outputs_v8/steering/results/nl_steering_results.pdf"
        break
    fi

    sleep "$INTERVAL"
done

echo "==========================================================="
echo "$(date) | Auto-submitter finished"
echo "==========================================================="
