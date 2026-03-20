#!/bin/bash
#SBATCH --job-name=analysis_v7
#SBATCH --partition=short
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/analysis_v7_%j.out
#SBATCH --error=logs/analysis_v7_%j.err

# ══════════════════════════════════════════════════════════════════════
# V7 Full Analysis Pipeline
# Run after all 240 users are extracted.
# Steps: verify → merge → export → probing → geometry → prediction
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "$(date) | ═══ Step 1: Check Progress ═══"
python3 -u check_progress.py

echo ""
echo "$(date) | ═══ Step 2: Verify Answers ═══"
python3 -u verify_answers.py

echo ""
echo "$(date) | ═══ Step 3: Merge Per-User → Per-Config ═══"
python3 -u merge_user_results.py --exp exp1a exp2

echo ""
echo "$(date) | ═══ Step 4: Export Raw Answers ═══"
python3 -u export_raw_answers.py

echo ""
echo "$(date) | ═══ Step 5: Probing (exp1a — all configs) ═══"
python3 -u prob_results.py --exp exp1a

echo ""
echo "$(date) | ═══ Step 6: Probing (exp2 — all configs) ═══"
python3 -u prob_results.py --exp exp2

echo ""
echo "$(date) | ═══ Step 7: Geometry (exp1a + exp2) ═══"
python3 -u geometry_v7.py --exp exp1a exp2

echo ""
echo "$(date) | ═══ Step 8: Prediction Accuracy ═══"
python3 -u prediction_results.py --model 12b --exp exp1a --config all
python3 -u prediction_results.py --model 12b --exp exp2 --config all

echo ""
echo "$(date) | ═══ ALL DONE ═══"
echo "Results in: outputs_v7/results/gemma3_12b_it/"
ls -lh outputs_v7/results/gemma3_12b_it/
