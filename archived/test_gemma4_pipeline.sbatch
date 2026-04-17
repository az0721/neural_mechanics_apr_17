#!/bin/bash
#SBATCH --job-name=test_g4a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/test_g4pipe_%j.out
#SBATCH --error=logs/test_g4pipe_%j.err

# Change gpu type above to test different cards:
#   --gres=gpu:v100-sxm2:1   (V100 32GB)
#   --gres=gpu:h200:1        (H200 140GB)

source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7

python -u test_gemma4_pipeline.py
