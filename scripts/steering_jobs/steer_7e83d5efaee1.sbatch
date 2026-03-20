#!/bin/bash
#SBATCH --job-name=steer_7e83d5efaee1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/steer_7e83d5efaee1_%j.out
#SBATCH --error=logs/steer_7e83d5efaee1_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user 7e83d5efaee1653b748640e7a7b5b69db4c1d8a100054717d187cf7cfc39a567 --cfgs 5 6 
