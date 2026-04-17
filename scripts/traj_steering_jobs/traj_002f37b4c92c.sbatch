#!/bin/bash
#SBATCH --job-name=traj_002f37b4c92c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/traj_002f37b4c92c_%j.out
#SBATCH --error=logs/traj_002f37b4c92c_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_traj_steering.py --user 002f37b4c92c7b83eab112b8190d468c48c1898c40a9f1b613eadeedb698cb5f --n-reps 50 --cfg 5
