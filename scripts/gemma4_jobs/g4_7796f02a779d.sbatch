#!/bin/bash
#SBATCH --job-name=g4_7796f02a779d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/g4_7796f02a779d_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/g4_7796f02a779d_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py \
    --user 7796f02a779d3f2139969acf072c62379862632db030983ab0d055214ca9d36f \
    --model gemma4_31b \
    --cot-only --align-v7
