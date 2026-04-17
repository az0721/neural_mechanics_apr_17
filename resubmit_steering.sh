#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Resubmit Phase 2 steering with updated config:
#   - 8 layers: [6, 12, 20, 25, 28, 33, 40, 45]
#   - Coeffs: [-10, -5, 0, 5, 10] (Method A single vector)
#   - 30 reps per condition
#   - 1 user × 1 cfg per job = 40 jobs
#   - Per job: 8 × 5 × 30 = 1,200 gen ≈ 6.7h
#
# Hooks into existing dependency chain:
#   p2_vectors (5502145) → this script
#
# Usage:
#   bash resubmit_steering.sh
#   bash resubmit_steering.sh --after 5502145   # explicit vectors job
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7

# ── Parse args ──
VEC_JOB=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --after) VEC_JOB="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$VEC_JOB" ]; then
    VEC_JOB=$(squeue -u "$USER" -h -n p2_vectors -o "%i" | head -1)
    if [ -z "$VEC_JOB" ]; then
        echo "ERROR: p2_vectors not found. Use --after JOBID"
        exit 1
    fi
fi

echo "================================================================"
echo "Resubmit Steering v8 (updated: 8 layers, 5 coeffs, split jobs)"
echo "Depends on: p2_vectors job $VEC_JOB"
echo "================================================================"

mkdir -p logs

STEER_JOB=$(sbatch --dependency=afterok:${VEC_JOB} \
    --job-name=p2_steer_v8b \
    --partition=short \
    --time=2-00:00:00 \
    --mem=2G \
    --cpus-per-task=1 \
    --output=logs/p2_steer_v8b_%j.out \
    --error=logs/p2_steer_v8b_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo \"\$(date) | Phase 2 Steering v8b — 8 layers, coeffs [-10..10], 30 reps\"
echo \"\$(date) | 1 user × 1 cfg per job = 40 jobs total\"

# Step A: Generate 40 sbatch scripts
python -u steering/generate_steering_jobs_v8.py --n-reps 30 --cfgs 5 6
echo \"\$(date) | Generated 40 sbatch scripts\"

# Step B: Prebuild prompts (in case not done yet)
python -u steering/prebuild_steering_prompts_v8.py
echo \"\$(date) | Prompts prebuilt\"

# Step C: Auto-submit loop
SCRIPTS_DIR=scripts/steering_jobs_v8
PERUSER_DIR=outputs_v8/steering/per_user_v8
MAX_GPU=8
MAX_MULTI=8
INTERVAL=90

mkdir -p \"\$PERUSER_DIR\"

while true; do
    QCACHE=\$(squeue -u \"\$USER\" -h -o \"%.18i %.200j %.8T %.10M %.10l %P\" 2>/dev/null)

    n_gpu=\$(echo \"\$QCACHE\" | grep \"stv8_\" | grep \" gpu \" 2>/dev/null | wc -l)
    n_multi=\$(echo \"\$QCACHE\" | grep \"stv8_\" | grep \"multigpu\" 2>/dev/null | wc -l)
    n_gpu=\${n_gpu:-0}; n_multi=\${n_multi:-0}
    avail_gpu=\$((MAX_GPU - n_gpu)); avail_multi=\$((MAX_MULTI - n_multi))
    [ \"\$avail_gpu\" -lt 0 ] && avail_gpu=0
    [ \"\$avail_multi\" -lt 0 ] && avail_multi=0

    n_done=\$(find \"\$PERUSER_DIR\" -name \"*_cfg*.json\" 2>/dev/null | wc -l)
    n_total=\$(ls \"\$SCRIPTS_DIR\"/*.sbatch 2>/dev/null | wc -l)

    for script in \"\$SCRIPTS_DIR\"/*.sbatch; do
        [ ! -f \"\$script\" ] && continue
        base=\$(basename \"\$script\" .sbatch)
        # Check if output exists: stv8_{uid}_c{cfg} → {uid}_cfg{cfg}.json
        uid=\$(echo \"\$base\" | sed 's/^stv8_//' | sed 's/_c[0-9]*\$//')
        cfg=\$(echo \"\$base\" | grep -o 'c[0-9]*\$' | sed 's/c//')
        outfile=\"\${PERUSER_DIR}/\${uid}*_cfg\${cfg}.json\"
        if ls \$outfile 1>/dev/null 2>&1; then continue; fi
        if echo \"\$QCACHE\" | grep -q \"\$base\"; then continue; fi
        if [ \"\$avail_gpu\" -gt 0 ]; then
            jobid=\$(sbatch --partition=gpu \"\$script\" 2>&1 | grep -o '[0-9]*')
            echo \"\$(date) | SUBMIT \$base → gpu → \$jobid\"
            avail_gpu=\$((avail_gpu - 1))
        elif [ \"\$avail_multi\" -gt 0 ]; then
            jobid=\$(sbatch --partition=multigpu \"\$script\" 2>&1 | grep -o '[0-9]*')
            echo \"\$(date) | SUBMIT \$base → multigpu → \$jobid\"
            avail_multi=\$((avail_multi - 1))
        else
            break
        fi
    done

    n_running=\$((n_gpu + n_multi))
    echo \"\$(date) | Done:\$n_done/\$n_total Run:\$n_running\"

    if [ \"\$n_done\" -ge \"\$n_total\" ] && [ \"\$n_running\" -eq 0 ]; then
        echo \"\$(date) | ALL STEERING DONE\"
        break
    fi
    sleep \"\$INTERVAL\"
done

echo \"\$(date) | Running analysis...\"
python -u steering/analyze_steering_v8.py
echo \"\$(date) | COMPLETE — check outputs_v8/steering/results/steering_results_v8.pdf\"
" 2>&1 | grep -o '[0-9]*')

echo ""
echo "  p2_vectors:     $VEC_JOB (existing, pending on merge)"
echo "       │"
echo "  └→ p2_steer_v8b: $STEER_JOB (afterok:$VEC_JOB)"
echo "       ├→ generate 40 sbatch (1 user × 1 cfg)"
echo "       ├→ prebuild prompts"
echo "       ├→ auto-submit 40 GPU jobs (gpu+multigpu)"
echo "       ├→ wait for all 40 to complete"
echo "       └→ analyze → steering_results_v8.pdf"
echo ""
echo "  Config: 8 layers × 5 coeffs × 30 reps = 1,200 gen/job ≈ 6.7h"
echo "  Total: 40 jobs × 1,200 = 48,000 generations"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  tail -f logs/p2_steer_v8b_*.out"
echo "================================================================"