#!/bin/bash
#SBATCH --job-name=stv8_0cbc2c943d98_v7_c5_L26
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stv8_0cbc2c943d98_v7_c5_L26_%j.out
#SBATCH --error=logs/stv8_0cbc2c943d98_v7_c5_L26_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "$(date) | Traj Steering: v7 cfg5 L26 0cbc2c943d98"

# Compute raw vectors if not exist (CPU, fast)
python -u steering/compute_vectors_nl.py --iter v7 --exp exp2 --cfgs 5

# Run steering (GPU, ~6h)
python -u steering/run_steering_per_user_v8.py \
    --user 0cbc2c943d988da8a24349bf46722683110753cc88d4fe1266fbacb1a03a3836 --iter v7 --layer 26 --cfg 5 --n-reps 30

echo "$(date) | Done"
