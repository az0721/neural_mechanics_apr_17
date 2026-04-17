#!/bin/bash
#SBATCH --job-name=traj_508ba0c962d2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/traj_508ba0c962d2_%j.out
#SBATCH --error=logs/traj_508ba0c962d2_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_traj_steering.py --user 508ba0c962d21258a2ae53d42b0dc68d6830f834db03463043d1af4b4a2802c6 --n-reps 50 --cfg 5
