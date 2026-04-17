#!/bin/bash
#SBATCH --job-name=traj_453d92fa46e3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/traj_453d92fa46e3_%j.out
#SBATCH --error=logs/traj_453d92fa46e3_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_traj_steering.py --user 453d92fa46e3eb826c925127f2fabe535380b804cebfda82c99b5e1524f79242 --n-reps 50 --cfg 5
