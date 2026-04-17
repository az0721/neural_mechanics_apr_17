#!/bin/bash
#SBATCH --job-name=u_b840ecf96b6a
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_b840ecf96b6a_%j.out
#SBATCH --error=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/logs/gemma4_26b_a4b_it/u_b840ecf96b6a_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
python -u run_extract_per_user.py --user b840ecf96b6a57684dbf6be40d4a357c2e03406681a46f84e865b0365a3e6456 --model gemma4_26b
