#!/bin/bash
#SBATCH --job-name=sae_analysis
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sae_analysis_%j.out
#SBATCH --error=logs/sae_analysis_%j.err
# ══════════════════════════════════════════════════════════════════════
# SAE Feature Analysis — CPU only (no GPU needed)
# Loads GemmaScope 2 SAEs + Phase 1 hidden states
# Checks if employment/weekday features activate on GPS data
# ══════════════════════════════════════════════════════════════════════
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "$(date) | SAE Feature Analysis"
python -u sae/sae_analysis.py
echo "$(date) | Done → sae/output/sae_feature_analysis.pdf"
