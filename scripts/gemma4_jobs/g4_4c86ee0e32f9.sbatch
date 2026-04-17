#!/bin/bash
#SBATCH --job-name=g4_4c86ee0e32f9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/g4_4c86ee0e32f9_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/g4_4c86ee0e32f9_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py \
    --user 4c86ee0e32f914c6815389a3bbb7354360bc0e8d0ac51784efadbfc0a849524d \
    --model gemma4_31b \
    --cot-only --align-v7
