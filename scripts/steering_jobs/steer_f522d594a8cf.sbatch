#!/bin/bash
#SBATCH --job-name=steer_f522d594a8cf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/steer_f522d594a8cf_%j.out
#SBATCH --error=logs/steer_f522d594a8cf_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user f522d594a8cf77eeb34d94338a2ddc565b539936bba7c94180f206be54eb32fa --cfgs 5 6 
