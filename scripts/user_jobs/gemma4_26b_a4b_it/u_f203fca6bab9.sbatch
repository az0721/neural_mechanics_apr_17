#!/bin/bash
#SBATCH --job-name=u_f203fca6bab9
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_f203fca6bab9_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_f203fca6bab9_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py --user f203fca6bab9e4ed4502cee5aa4d55e8321e25e0b63b4f2bff030c08436e591c --model gemma4_26b
