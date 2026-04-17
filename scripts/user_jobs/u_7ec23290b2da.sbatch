#!/bin/bash
#SBATCH --job-name=u_7ec23290b2da
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/u_7ec23290b2da_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/u_7ec23290b2da_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py --user 7ec23290b2da6a7bf121ae873c6b4a19426278b9a9034e559a0976c605d57b82 --align-v7
