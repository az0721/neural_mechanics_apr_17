#!/bin/bash
#SBATCH --job-name=u_bb16777dcff1
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/u_bb16777dcff1_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/u_bb16777dcff1_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py --user bb16777dcff18375803d1617f3a32d5cdc666a5e097bfd9444dad2399c6ae8bb --align-v7
