#!/bin/bash
#SBATCH --job-name=u_ae8acf6860f9
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_ae8acf6860f9_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_ae8acf6860f9_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py --user ae8acf6860f900d86cb74382f5d12f345e5cf1b40dbe50b62584151aa007ebc2 --model gemma4_26b
