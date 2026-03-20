#!/bin/bash
#SBATCH --job-name=steer_f203fca6bab9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/steer_f203fca6bab9_%j.out
#SBATCH --error=logs/steer_f203fca6bab9_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user f203fca6bab9e4ed4502cee5aa4d55e8321e25e0b63b4f2bff030c08436e591c --cfgs 5 6 
