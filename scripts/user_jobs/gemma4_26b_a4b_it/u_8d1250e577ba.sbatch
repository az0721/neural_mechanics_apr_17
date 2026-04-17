#!/bin/bash
#SBATCH --job-name=u_8d1250e577ba
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_8d1250e577ba_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_8d1250e577ba_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py --user 8d1250e577ba3f0d60c9d8a5bdf554092c3113eb964d0b5b1aff94be11d6d8c4 --model gemma4_26b
