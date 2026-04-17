#!/bin/bash
#SBATCH --job-name=traj_cbe8574bbca7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/traj_cbe8574bbca7_%j.out
#SBATCH --error=logs/traj_cbe8574bbca7_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_traj_steering.py --user cbe8574bbca716a3644386ef06d3f0d1878a916e5f289299dafbf33106df5b30 --n-reps 50 --cfg 5
