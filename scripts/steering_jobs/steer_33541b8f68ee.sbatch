#!/bin/bash
#SBATCH --job-name=steer_33541b8f68ee
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/steer_33541b8f68ee_%j.out
#SBATCH --error=logs/steer_33541b8f68ee_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user 33541b8f68eed966b8e9609041a2451b9c85ff9551691ee713832f0822f06346 --cfgs 5 6 
