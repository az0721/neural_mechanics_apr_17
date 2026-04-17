#!/bin/bash
#SBATCH --job-name=g4_956b2cf13cd8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/g4_956b2cf13cd8_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/g4_956b2cf13cd8_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py \
    --user 956b2cf13cd87a76c3ade28064e64a3deea43b70b68583e5a2274aab624121e2 \
    --model gemma4_31b \
    --cot-only --align-v7
