#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Auto-submit NL steering jobs.
# Generates sbatch scripts then submits all 8 jobs to gpu + multigpu.
#
# V7 jobs can run immediately (v7 vectors already exist).
# V8 jobs need v8 hidden states (must wait for merge).
#
# Usage:
#   bash submit_nl_steering.sh                    # submit all 8
#   bash submit_nl_steering.sh --v7-only          # only v7 (4 jobs)
#   bash submit_nl_steering.sh --after 5502140    # depend on merge job
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7

V7_ONLY=false
AFTER_JOB=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --v7-only) V7_ONLY=true; shift ;;
        --after) AFTER_JOB="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "================================================================"
echo "NL Steering Pipeline"
echo "================================================================"

mkdir -p logs

# Generate all sbatch scripts
python steering/generate_steering_jobs_nl.py

SCRIPTS_DIR=scripts/nl_steering_jobs
MAX_GPU=4
MAX_MULTI=4

echo ""
echo "Submitting jobs..."
echo ""

submitted=0
for script in "$SCRIPTS_DIR"/*.sbatch; do
    [ ! -f "$script" ] && continue
    base=$(basename "$script" .sbatch)

    # Skip v8 if --v7-only
    if $V7_ONLY && [[ "$base" == *"_v8_"* ]]; then
        echo "  SKIP $base (v7-only mode)"
        continue
    fi

    # Check if output already exists
    # Parse iter/exp/cfg from filename: nl_{iter}_{exp}_c{cfg}
    iter=$(echo "$base" | sed 's/nl_//' | cut -d'_' -f1)
    exp=$(echo "$base" | sed 's/nl_//' | sed "s/${iter}_//" | sed 's/_c[0-9]*//')
    cfg=$(echo "$base" | grep -o 'c[0-9]*$' | sed 's/c//')

    outfile="outputs_${iter}/steering/nl_results/${exp}_cfg${cfg}_nl.json"
    if [ -f "$outfile" ]; then
        echo "  SKIP $base (output exists: $outfile)"
        continue
    fi

    # Add dependency if specified (for v8 jobs that need merge)
    DEP=""
    if [ -n "$AFTER_JOB" ] && [[ "$base" == *"_v8_"* ]]; then
        DEP="--dependency=afterok:${AFTER_JOB}"
    fi

    # Try gpu first, then multigpu
    n_gpu=$(squeue -u "$USER" -h -o "%j %P" 2>/dev/null | grep "nl_" | grep " gpu " | wc -l)
    n_multi=$(squeue -u "$USER" -h -o "%j %P" 2>/dev/null | grep "nl_" | grep "multigpu" | wc -l)

    if [ "$n_gpu" -lt "$MAX_GPU" ]; then
        jobid=$(sbatch --partition=gpu $DEP "$script" 2>&1 | grep -o '[0-9]*')
        echo "  SUBMIT $base → gpu → $jobid ${DEP:+(dep: $AFTER_JOB)}"
    elif [ "$n_multi" -lt "$MAX_MULTI" ]; then
        jobid=$(sbatch --partition=multigpu $DEP "$script" 2>&1 | grep -o '[0-9]*')
        echo "  SUBMIT $base → multigpu → $jobid ${DEP:+(dep: $AFTER_JOB)}"
    else
        jobid=$(sbatch --partition=gpu $DEP "$script" 2>&1 | grep -o '[0-9]*')
        echo "  SUBMIT $base → gpu (queue) → $jobid ${DEP:+(dep: $AFTER_JOB)}"
    fi
    submitted=$((submitted + 1))
done

echo ""
echo "================================================================"
echo "Submitted $submitted NL steering jobs"
echo ""
if $V7_ONLY; then
    echo "  Mode: v7-only (4 jobs)"
    echo "  V7 vectors exist → jobs can run immediately"
else
    echo "  Mode: all (8 jobs)"
    if [ -n "$AFTER_JOB" ]; then
        echo "  V8 jobs depend on merge job $AFTER_JOB"
    fi
fi
echo ""
echo "  Each job: ~30-40 min (5 questions × 48 layers × ~10 conditions)"
echo "  Monitor: squeue -u $USER | grep nl_"
echo ""
echo "  After all complete, run analysis:"
echo "    python steering/analyze_steering_nl.py"
echo "================================================================"
