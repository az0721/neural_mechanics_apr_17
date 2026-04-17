#!/bin/bash
#SBATCH --job-name=probe4_1a
#SBATCH --partition=short
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/probe4_exp1a_%j.out
#SBATCH --error=logs/probe4_exp1a_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "$(date) | Probe v4: exp1a (4 cfgs × ~86min = ~5.7h)"
echo "  MLP: ALL 49 layers, max_iter=500, no early_stopping"
python -u prob_results.py --exp exp1a --n-rand 5 --force
echo "$(date) | Done"
