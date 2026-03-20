#!/bin/bash
#SBATCH --job-name=probe3_exp2
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/probe3_exp2_%j.out
#SBATCH --error=logs/probe3_exp2_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "$(date) | Probe v3: exp2 (8 configs × ~68min = ~9h)"
python -u prob_results_v3.py --exp exp2 --n-rand 5 --n-mlp-layers 5
echo "$(date) | Done"
