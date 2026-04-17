#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Master Dependency Chain — Phase 1 (merge → probe → geometry → prediction)
#                          + Phase 2 (vectors → steering jobs)
#
# Submits ALL steps as dependent SLURM jobs.
# Each step starts only after the previous one completes successfully.
#
# Usage:
#   bash submit_full_pipeline.sh                    # auto-detect auto_v7
#   bash submit_full_pipeline.sh --after 5490473    # explicit dependency
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7

# ── Parse args ──
AFTER_JOB=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --after) AFTER_JOB="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect auto_v7 job if not specified
if [ -z "$AFTER_JOB" ]; then
    AFTER_JOB=$(squeue -u "$USER" -h -n auto_v7 -o "%i" | head -1)
    if [ -z "$AFTER_JOB" ]; then
        echo "ERROR: auto_v7 not found in queue. Use --after JOBID"
        exit 1
    fi
fi

echo "================================================================"
echo "Master Pipeline — Phase 1 + Phase 2"
echo "Waiting for: job $AFTER_JOB (auto_v7 extraction)"
echo "================================================================"

mkdir -p logs

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Merge per-user results (CPU, ~10 min)
# ══════════════════════════════════════════════════════════════════════
MERGE_JOB=$(sbatch --dependency=afterany:${AFTER_JOB} \
    --job-name=p1_merge \
    --partition=short \
    --time=3:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --output=logs/p1_merge_%j.out \
    --error=logs/p1_merge_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
echo \"\$(date) | Step 1: Merge\"
python -u merge_user_results.py --exp exp1a exp2
echo \"\$(date) | Merge done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 1 MERGE:      job $MERGE_JOB (after $AFTER_JOB)"

# ══════════════════════════════════════════════════════════════════════
# STEP 2a: Probe exp1a (CPU, ~6h)
# ══════════════════════════════════════════════════════════════════════
PROBE1A_JOB=$(sbatch --dependency=afterok:${MERGE_JOB} \
    --job-name=p1_probe1a \
    --partition=short \
    --time=20:00:00 \
    --mem=16G \
    --cpus-per-task=8 \
    --output=logs/p1_probe_exp1a_%j.out \
    --error=logs/p1_probe_exp1a_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
echo \"\$(date) | Step 2a: Probe exp1a (v4: MLP all layers)\"
python -u prob_results.py --exp exp1a --force
echo \"\$(date) | Probe exp1a done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 2a PROBE_1a:  job $PROBE1A_JOB (after merge $MERGE_JOB)"

# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Probe exp2 (CPU, ~12h) — runs in PARALLEL with 2a
# ══════════════════════════════════════════════════════════════════════
PROBE2_JOB=$(sbatch --dependency=afterok:${MERGE_JOB} \
    --job-name=p1_probe2 \
    --partition=short \
    --time=24:00:00 \
    --mem=16G \
    --cpus-per-task=8 \
    --output=logs/p1_probe_exp2_%j.out \
    --error=logs/p1_probe_exp2_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
echo \"\$(date) | Step 2b: Probe exp2 (v4: MLP all layers)\"
python -u prob_results.py --exp exp2 --force
echo \"\$(date) | Probe exp2 done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 2b PROBE_2:   job $PROBE2_JOB (after merge $MERGE_JOB)"

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Geometry (CPU, ~10 min) — after merge
# ══════════════════════════════════════════════════════════════════════
GEO_JOB=$(sbatch --dependency=afterok:${MERGE_JOB} \
    --job-name=p1_geometry \
    --partition=short \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output=logs/p1_geometry_%j.out \
    --error=logs/p1_geometry_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
echo \"\$(date) | Step 3: Geometry (PCA + DA + Sep)\"
python -u geometry_v7.py --exp exp1a exp2
echo \"\$(date) | Geometry done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 3 GEOMETRY:   job $GEO_JOB (after merge $MERGE_JOB)"

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Prediction accuracy + GoF (CPU, ~5 min) — after merge
# ══════════════════════════════════════════════════════════════════════
PRED_JOB=$(sbatch --dependency=afterok:${MERGE_JOB} \
    --job-name=p1_predict \
    --partition=short \
    --time=1:00:00 \
    --mem=8G \
    --cpus-per-task=4 \
    --output=logs/p1_prediction_%j.out \
    --error=logs/p1_prediction_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
echo \"\$(date) | Step 4: Prediction accuracy + GoF\"
python -u prediction_results.py
echo \"\$(date) | Prediction done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 4 PREDICTION: job $PRED_JOB (after merge $MERGE_JOB)"

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Compute steering vectors (CPU, ~2 min) — after merge
# ══════════════════════════════════════════════════════════════════════
VEC_JOB=$(sbatch --dependency=afterok:${MERGE_JOB} \
    --job-name=p2_vectors \
    --partition=short \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output=logs/p2_vectors_%j.out \
    --error=logs/p2_vectors_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
echo \"\$(date) | Step 5: Compute steering vectors (Method A + B, cfg 5 6)\"
python -u steering/compute_vectors.py --cfgs 5 6 --method both
echo \"\$(date) | Vectors done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 5 VECTORS:    job $VEC_JOB (after merge $MERGE_JOB)"

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Generate steering sbatch + submit auto-submitter
#         — after vectors are computed
# ══════════════════════════════════════════════════════════════════════
STEER_JOB=$(sbatch --dependency=afterok:${VEC_JOB} \
    --job-name=p2_steer_launch \
    --partition=short \
    --time=2-00:00:00 \
    --mem=2G \
    --cpus-per-task=1 \
    --output=logs/p2_steer_launch_%j.out \
    --error=logs/p2_steer_launch_%j.err \
    --wrap="
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo \"\$(date) | Step 6: Generate steering jobs + auto-submit\"

# Generate 20 sbatch scripts (30 reps, cfgs 5 6)
python -u steering/generate_steering_jobs_v8.py --n-reps 30 --cfgs 5 6

# Now run the auto-submitter inline (it loops until all done)
echo \"\$(date) | Starting auto-submitter for 20 steering users\"

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
    n_gpu=\${n_gpu:-0}
    n_multi=\${n_multi:-0}
    avail_gpu=\$((MAX_GPU - n_gpu))
    avail_multi=\$((MAX_MULTI - n_multi))
    [ \"\$avail_gpu\" -lt 0 ] && avail_gpu=0
    [ \"\$avail_multi\" -lt 0 ] && avail_multi=0

    n_done=\$(find \"\$PERUSER_DIR\" -name \"*.json\" 2>/dev/null | wc -l)
    n_total=\$(ls \"\$SCRIPTS_DIR\"/*.sbatch 2>/dev/null | wc -l)

    for script in \"\$SCRIPTS_DIR\"/*.sbatch; do
        [ ! -f \"\$script\" ] && continue
        uid=\$(basename \"\$script\" .sbatch | sed 's/^stv8_//')
        if find \"\$PERUSER_DIR\" -name \"\${uid}*.json\" 2>/dev/null | grep -q .; then continue; fi
        if echo \"\$QCACHE\" | grep -q \"stv8_\${uid}\"; then continue; fi
        if [ \"\$avail_gpu\" -gt 0 ]; then
            jobid=\$(sbatch --partition=gpu \"\$script\" 2>&1 | grep -o '[0-9]*')
            echo \"\$(date) | SUBMIT \$uid → gpu → \$jobid\"
            avail_gpu=\$((avail_gpu - 1))
        elif [ \"\$avail_multi\" -gt 0 ]; then
            jobid=\$(sbatch --partition=multigpu \"\$script\" 2>&1 | grep -o '[0-9]*')
            echo \"\$(date) | SUBMIT \$uid → multigpu → \$jobid\"
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

echo \"\$(date) | Step 6 complete — running analysis\"
python -u steering/analyze_steering_v8.py
echo \"\$(date) | Analysis done\"
" 2>&1 | grep -o '[0-9]*')

echo "  Step 6 STEERING:   job $STEER_JOB (after vectors $VEC_JOB)"

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "DEPENDENCY CHAIN SUBMITTED"
echo "================================================================"
echo ""
echo "  auto_v7 extraction:  $AFTER_JOB (RUNNING)"
echo "       │"
echo "       └→ Step 1 MERGE:       $MERGE_JOB (afterany:$AFTER_JOB)"
echo "              │"
echo "              ├→ Step 2a PROBE exp1a: $PROBE1A_JOB (afterok:$MERGE_JOB)"
echo "              ├→ Step 2b PROBE exp2:  $PROBE2_JOB  (afterok:$MERGE_JOB)"
echo "              ├→ Step 3 GEOMETRY:     $GEO_JOB     (afterok:$MERGE_JOB)"
echo "              ├→ Step 4 PREDICTION:   $PRED_JOB    (afterok:$MERGE_JOB)"
echo "              └→ Step 5 VECTORS:      $VEC_JOB     (afterok:$MERGE_JOB)"
echo "                     │"
echo "                     └→ Step 6 STEERING: $STEER_JOB (afterok:$VEC_JOB)"
echo "                        (generates jobs + auto-submits 20 users"
echo "                         + runs analyze_steering_v8.py when done)"
echo ""
echo "  Phase 1 analysis (steps 2-4) runs in PARALLEL after merge."
echo "  Phase 2 steering (step 6) starts after vectors are computed."
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  tail -f logs/p1_merge_*.out"
echo "  tail -f logs/p2_steer_launch_*.out"
echo "================================================================"