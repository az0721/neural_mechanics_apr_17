#!/bin/bash
#SBATCH --job-name=probe_exp1a
#SBATCH --partition=short
#SBATCH --time=18:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/probe_exp1a_%j.out
#SBATCH --error=logs/probe_exp1a_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

echo "$(date) | Starting probe analysis: exp1a"
python -u prob_results_v2.py --exp exp1a --n-rand 5 --n-mlp-layers 5

# Rename to avoid overwrite
mv outputs_v7/results/gemma3_12b_it/probe_results.pdf \
   outputs_v7/results/gemma3_12b_it/probe_results_exp1a.pdf 2>/dev/null

echo "$(date) | Done"
