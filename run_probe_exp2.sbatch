#!/bin/bash
#SBATCH --job-name=probe4_exp2
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/probe4_exp2_%j.out
#SBATCH --error=logs/probe4_exp2_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "$(date) | Probe v4: exp2 (8 cfgs × ~86min = ~11.5h)"
echo "  MLP: ALL 49 layers, max_iter=500, no early_stopping"
python -u prob_results.py --exp exp2 --force
echo "$(date) | Done"
