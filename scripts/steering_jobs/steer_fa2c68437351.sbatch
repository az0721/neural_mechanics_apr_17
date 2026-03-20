#!/bin/bash
#SBATCH --job-name=steer_fa2c68437351
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/steer_fa2c68437351_%j.out
#SBATCH --error=logs/steer_fa2c68437351_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user fa2c684373517499803f3bf1ef0f7de88aae14acd6cb4e2d94a9474d1b100826 --cfgs 5 6 
