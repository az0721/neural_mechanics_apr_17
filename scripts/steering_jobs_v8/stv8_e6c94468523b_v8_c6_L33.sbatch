#!/bin/bash
#SBATCH --job-name=stv8_e6c94468523b_v8_c6_L33
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stv8_e6c94468523b_v8_c6_L33_%j.out
#SBATCH --error=logs/stv8_e6c94468523b_v8_c6_L33_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "$(date) | Traj Steering: v8 cfg6 L33 e6c94468523b"

# Compute raw vectors if not exist (CPU, fast)
python -u steering/compute_vectors_nl.py --iter v8 --exp exp2 --cfgs 6

# Run steering (GPU, ~6h)
python -u steering/run_steering_per_user_v8.py \
    --user e6c94468523b61980648dfef98a59495acc7284dfe7d676c2598e264f13a4a12 --iter v8 --layer 33 --cfg 6 --n-reps 30

echo "$(date) | Done"
