#!/bin/bash
#SBATCH --job-name=auto_nl_g4
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/auto_nl_g4_%j.out
#SBATCH --error=logs/auto_nl_g4_%j.err
# ══════════════════════════════════════════════════════════════════════
# Auto-submitter for NL-based steering — Gemma 4 31B.
#
# 12 jobs total: 2 iters × (4 exp2 + 2 exp1a) CoT-only cfgs
#   exp2:  cfg 2,4,6,8  × v7,v8 = 8 jobs
#   exp1a: cfg 2,4      × v7,v8 = 4 jobs
#
# Partitions: gpu (max 8) + multigpu (max 8) + gpu-short (max 4) = 20 slots
# Each job: ~60-90 min (well under 2h limit)
#
# Usage:
#   sbatch auto_submit_steering_nl.sbatch
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env

MODEL="gemma4_31b"
MODEL_SHORT="g4"
MODEL_DIR="gemma-4-31B-it-unsloth-bnb-4bit"
SCRIPTS_DIR="scripts/nl_steering_jobs"
MAX_GPU=8
MAX_MULTI=8
MAX_SHORT=4
INTERVAL=60
TIME_LIMIT="00:30:00"

mkdir -p logs "$SCRIPTS_DIR"

echo "$(date) | NL Steering Auto-Submitter (Gemma 4 31B)"
echo "$(date) | Partitions: gpu($MAX_GPU) multigpu($MAX_MULTI) gpu-short($MAX_SHORT)"
echo "==========================================================="

# Step 1: Generate sbatch scripts
echo "$(date) | Generating sbatch scripts..."
python -u steering/generate_steering_jobs_nl.py --model "$MODEL" --time-limit "$TIME_LIMIT"
echo "$(date) | Scripts generated in $SCRIPTS_DIR/"

n_total=$(ls "$SCRIPTS_DIR"/nl_${MODEL_SHORT}_*.sbatch 2>/dev/null | wc -l)
echo "$(date) | Total jobs: $n_total"
echo "==========================================================="

# Step 2: Loop — submit and monitor
while true; do
    QCACHE=$(squeue -u "$USER" -h -o "%.18i %.200j %.8T %.10M %.10l %P" 2>/dev/null)

    n_gpu=$(echo "$QCACHE" | grep "nl_${MODEL_SHORT}_" | grep " gpu " 2>/dev/null | wc -l)
    n_multi=$(echo "$QCACHE" | grep "nl_${MODEL_SHORT}_" | grep "multigpu" 2>/dev/null | wc -l)
    n_short=$(echo "$QCACHE" | grep "nl_${MODEL_SHORT}_" | grep "gpu-short" 2>/dev/null | wc -l)
    n_gpu=${n_gpu:-0}
    n_multi=${n_multi:-0}
    n_short=${n_short:-0}
    avail_gpu=$((MAX_GPU - n_gpu))
    avail_multi=$((MAX_MULTI - n_multi))
    avail_short=$((MAX_SHORT - n_short))
    [ "$avail_gpu" -lt 0 ] && avail_gpu=0
    [ "$avail_multi" -lt 0 ] && avail_multi=0
    [ "$avail_short" -lt 0 ] && avail_short=0

    # Count completed jobs by checking output JSON files
    n_done=0
    for script in "$SCRIPTS_DIR"/nl_${MODEL_SHORT}_*.sbatch; do
        [ ! -f "$script" ] && continue
        base=$(basename "$script" .sbatch)
        # Parse: nl_g4_{iter}_{exp}_c{cfg}
        iter=$(echo "$base" | sed "s/nl_${MODEL_SHORT}_//" | cut -d'_' -f1)
        rest=$(echo "$base" | sed "s/nl_${MODEL_SHORT}_${iter}_//" | sed 's/_c[0-9]*$//')
        cfg=$(echo "$base" | grep -o 'c[0-9]*$' | sed 's/c//')
        outfile="${MODEL_DIR}/outputs_${iter}/steering/nl_results/${rest}_cfg${cfg}_nl.json"
        if [ -f "$outfile" ]; then
            n_done=$((n_done + 1))
        fi
    done

    n_submitted=0
    for script in "$SCRIPTS_DIR"/nl_${MODEL_SHORT}_*.sbatch; do
        [ ! -f "$script" ] && continue
        base=$(basename "$script" .sbatch)

        # Parse iter/exp/cfg
        iter=$(echo "$base" | sed "s/nl_${MODEL_SHORT}_//" | cut -d'_' -f1)
        rest=$(echo "$base" | sed "s/nl_${MODEL_SHORT}_${iter}_//" | sed 's/_c[0-9]*$//')
        cfg=$(echo "$base" | grep -o 'c[0-9]*$' | sed 's/c//')
        outfile="${MODEL_DIR}/outputs_${iter}/steering/nl_results/${rest}_cfg${cfg}_nl.json"

        # Skip if output already exists
        if [ -f "$outfile" ]; then
            continue
        fi

        # Skip if already in queue
        if echo "$QCACHE" | grep -q "$base"; then
            continue
        fi

        # Check hidden states exist for this config
        hidden_check="${MODEL_DIR}/outputs_${iter}/hidden_states/gemma4_31b_it_bnb4/${rest}_cfg${cfg}.npz"
        if [ ! -f "$hidden_check" ]; then
            continue
        fi

        # Submit to available partition (prefer gpu-short for 2h jobs)
        if [ "$avail_short" -gt 0 ]; then
            jobid=$(sbatch --partition=gpu-short --time="$TIME_LIMIT" "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → gpu-short → $jobid"
            avail_short=$((avail_short - 1))
            n_submitted=$((n_submitted + 1))
        elif [ "$avail_gpu" -gt 0 ]; then
            jobid=$(sbatch --partition=gpu --time="$TIME_LIMIT" "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → gpu → $jobid"
            avail_gpu=$((avail_gpu - 1))
            n_submitted=$((n_submitted + 1))
        elif [ "$avail_multi" -gt 0 ]; then
            jobid=$(sbatch --partition=multigpu --time="$TIME_LIMIT" "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → multigpu → $jobid"
            avail_multi=$((avail_multi - 1))
            n_submitted=$((n_submitted + 1))
        else
            break
        fi
    done

    n_running=$((n_gpu + n_multi + n_short + n_submitted))
    echo "$(date) | Done:$n_done/$n_total Run:$n_running Sub:$n_submitted | gpu=$avail_gpu multi=$avail_multi short=$avail_short"

    # Check completion
    if [ "$n_done" -ge "$n_total" ] && [ "$n_running" -eq 0 ]; then
        echo "$(date) | ALL NL STEERING JOBS COMPLETE"
        echo "$(date) | Running analysis..."
        python -u steering/analyze_steering_nl.py --model "$MODEL"
        echo "$(date) | Analysis done"
        break
    fi

    sleep "$INTERVAL"
done

echo "==========================================================="
echo "$(date) | Auto-submitter finished"
echo "==========================================================="
