#!/bin/bash
#SBATCH --job-name=traj_0fe75a6fa7db
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/traj_0fe75a6fa7db_%j.out
#SBATCH --error=logs/traj_0fe75a6fa7db_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_traj_steering.py --user 0fe75a6fa7db61783bf4ad62406503d0dce1d3ece6f135ac61c4abdc1fa1ae76 --n-reps 50 --cfg 5
