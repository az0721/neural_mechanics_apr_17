#!/bin/bash
#SBATCH --job-name=nl_g4_v7_exp1a_c2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/nl_g4_v7_exp1a_c2_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/nl_g4_v7_exp1a_c2_%j.err

set -euo pipefail
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "$(date) | NL Steering: gemma4_31b v7 exp1a cfg2"

# Step 1: Compute vectors (CPU, fast)
python -u steering/compute_vectors_nl.py --model gemma4_31b --iter v7 --exp exp1a --cfgs 2

# Step 2: Run steering (GPU)
python -u steering/run_steering_nl.py --model gemma4_31b --iter v7 --exp exp1a --cfg 2

echo "$(date) | Done"
