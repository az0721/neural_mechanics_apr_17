#!/bin/bash
#SBATCH --job-name=st2_0601371662aa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/st2_0601371662aa_%j.out
#SBATCH --error=logs/st2_0601371662aa_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user 0601371662aa79ff901e33f1578b85ce7ac75a63b8f6502b41f5111203c46cc2 --cfgs 5 6
