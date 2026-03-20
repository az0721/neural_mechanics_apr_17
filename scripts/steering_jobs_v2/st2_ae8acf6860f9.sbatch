#!/bin/bash
#SBATCH --job-name=st2_ae8acf6860f9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/st2_ae8acf6860f9_%j.out
#SBATCH --error=logs/st2_ae8acf6860f9_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user ae8acf6860f900d86cb74382f5d12f345e5cf1b40dbe50b62584151aa007ebc2 --cfgs 5 6
