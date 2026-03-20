#!/bin/bash
#SBATCH --job-name=st2_f1e263019db4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/st2_f1e263019db4_%j.out
#SBATCH --error=logs/st2_f1e263019db4_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user f1e263019db4d1d6ca781296e3b0d5479190e72a64caf1619682f93f0f9a653c --cfgs 5 6
