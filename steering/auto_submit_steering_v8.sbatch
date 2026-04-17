#!/bin/bash
#SBATCH --job-name=auto_stv8
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/auto_stv8_%j.out
#SBATCH --error=logs/auto_stv8_%j.err
# ══════════════════════════════════════════════════════════════════════
# Auto-submitter for trajectory steering Phase 2c (Mar 30 redesign).
#
# 80 jobs: 2 iters × 2 layers (26,33) × 2 cfgs (5,6) × 10 users
# Each job: 240 gen × ~1.5 min = ~6h (under 8h limit)
# Both methods: old (coeffs) + new Matteo (raw V_emp/V_unemp)
# RAW unnormalized vectors
#
# Partitions: gpu (max 8) + multigpu (max 8) = 16 slots
#
# Usage:
#   sbatch steering/auto_submit_steering_v8.sbatch
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

SCRIPTS_DIR="scripts/steering_jobs_v8"
MAX_GPU=8
MAX_MULTI=8
INTERVAL=90

mkdir -p logs "$SCRIPTS_DIR"

echo "$(date) | Trajectory Steering Auto-Submitter (Mar 30)"
echo "$(date) | 80 jobs: 2 iters × 2 layers × 2 cfgs × 10 users"
echo "$(date) | Partitions: gpu($MAX_GPU) multigpu($MAX_MULTI)"
echo "==========================================================="

# Step 1: Generate scripts + prebuild prompts
python -u steering/generate_steering_jobs_v8.py
python -u steering/prebuild_steering_prompts_v8.py

n_total=$(ls "$SCRIPTS_DIR"/*.sbatch 2>/dev/null | wc -l)
echo "$(date) | Total jobs: $n_total"
echo "==========================================================="

# Step 2: Loop
while true; do
    QCACHE=$(squeue -u "$USER" -h -o "%.18i %.200j %.8T %.10M %.10l %P" 2>/dev/null)

    n_gpu=$(echo "$QCACHE" | grep "stv8_" | grep " gpu " 2>/dev/null | wc -l)
    n_multi=$(echo "$QCACHE" | grep "stv8_" | grep "multigpu" 2>/dev/null | wc -l)
    n_gpu=${n_gpu:-0}; n_multi=${n_multi:-0}
    avail_gpu=$((MAX_GPU - n_gpu)); avail_multi=$((MAX_MULTI - n_multi))
    [ "$avail_gpu" -lt 0 ] && avail_gpu=0
    [ "$avail_multi" -lt 0 ] && avail_multi=0

    # Count done
    n_done=0
    for script in "$SCRIPTS_DIR"/*.sbatch; do
        [ ! -f "$script" ] && continue
        base=$(basename "$script" .sbatch)
        # Parse: stv8_{uid12}_{iter}_c{cfg}_L{layer}
        uid12=$(echo "$base" | sed 's/^stv8_//' | sed 's/_v[78]_c[0-9]*_L[0-9]*//')
        iter_name=$(echo "$base" | grep -o 'v[78]')
        cfg_num=$(echo "$base" | grep -o 'c[0-9]*' | head -1 | sed 's/c//')
        layer_num=$(echo "$base" | grep -o 'L[0-9]*' | sed 's/L//')
        outfile="outputs_${iter_name}/steering/per_user_v8/${uid12}*_cfg${cfg_num}_L${layer_num}.json"
        if ls $outfile 1>/dev/null 2>&1; then
            n_done=$((n_done + 1))
        fi
    done

    n_submitted=0
    for script in "$SCRIPTS_DIR"/*.sbatch; do
        [ ! -f "$script" ] && continue
        base=$(basename "$script" .sbatch)
        uid12=$(echo "$base" | sed 's/^stv8_//' | sed 's/_v[78]_c[0-9]*_L[0-9]*//')
        iter_name=$(echo "$base" | grep -o 'v[78]')
        cfg_num=$(echo "$base" | grep -o 'c[0-9]*' | head -1 | sed 's/c//')
        layer_num=$(echo "$base" | grep -o 'L[0-9]*' | sed 's/L//')

        # Skip if done
        outfile="outputs_${iter_name}/steering/per_user_v8/${uid12}*_cfg${cfg_num}_L${layer_num}.json"
        if ls $outfile 1>/dev/null 2>&1; then continue; fi

        # Skip if in queue
        if echo "$QCACHE" | grep -q "$base"; then continue; fi

        # Skip v8 if hidden states not ready
        if [ "$iter_name" = "v8" ]; then
            if [ ! -f "outputs_v8/hidden_states/gemma3_12b_it/exp2_cfg${cfg_num}.npz" ]; then
                continue
            fi
        fi

        # Submit
        if [ "$avail_gpu" -gt 0 ]; then
            jobid=$(sbatch --partition=gpu "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → gpu → $jobid"
            avail_gpu=$((avail_gpu - 1)); n_submitted=$((n_submitted + 1))
        elif [ "$avail_multi" -gt 0 ]; then
            jobid=$(sbatch --partition=multigpu "$script" 2>&1 | grep -o '[0-9]*')
            echo "$(date) | SUBMIT $base → multigpu → $jobid"
            avail_multi=$((avail_multi - 1)); n_submitted=$((n_submitted + 1))
        else
            break
        fi
    done

    n_running=$((n_gpu + n_multi + n_submitted))
    echo "$(date) | Done:$n_done/$n_total Run:$n_running Sub:$n_submitted"

    if [ "$n_done" -ge "$n_total" ] && [ "$n_running" -eq 0 ]; then
        echo "$(date) | ALL TRAJECTORY STEERING DONE"
        echo "$(date) | Running analysis..."
        python -u steering/analyze_steering_v8.py
        echo "$(date) | Complete"
        break
    fi

    sleep "$INTERVAL"
done

echo "==========================================================="
echo "$(date) | Auto-submitter finished"
echo "==========================================================="