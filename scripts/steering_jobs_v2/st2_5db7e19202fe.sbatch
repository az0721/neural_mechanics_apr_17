#!/bin/bash
#SBATCH --job-name=st2_5db7e19202fe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/st2_5db7e19202fe_%j.out
#SBATCH --error=logs/st2_5db7e19202fe_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user 5db7e19202fefe287fd8ad03bb8a64aadc6b871661e38353de40537a4f0539d7 --cfgs 5 6
