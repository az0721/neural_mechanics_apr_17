#!/bin/bash
#SBATCH --job-name=traj_366a2822c013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/traj_366a2822c013_%j.out
#SBATCH --error=logs/traj_366a2822c013_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_traj_steering.py --user 366a2822c0136dca582cb4ba15935b5e317f6b395284a0531ce8c0f0dc790e28 --n-reps 50 --cfg 5
