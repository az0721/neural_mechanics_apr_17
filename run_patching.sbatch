#!/bin/bash
#SBATCH --job-name=patching
#SBATCH --partition=short
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/patching_%j.out
#SBATCH --error=logs/patching_%j.err
# ══════════════════════════════════════════════════════════════════════
# Run both patching analyses (CPU only — no GPU needed)
#
# Step 1: Probe-based patching (per-layer concept sensitivity)
# Step 2: Activation patching (residual stream approximation)
# ══════════════════════════════════════════════════════════════════════
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env

mkdir -p patching/output

echo "$(date) | Step 2: Activation Patching (residual stream approx)"
for iter in v7 v8; do
    for cfg in 5 6; do
        echo "$(date) | exp2 ${iter} cfg${cfg}"
        python -u patching/activation_patching.py --exp exp2 --iter $iter --cfg $cfg --n-pairs 50
    done
done

for iter in v7 v8; do
    for cfg in 3 4; do
        echo "$(date) | exp1a ${iter} cfg${cfg}"
        python -u patching/activation_patching.py --exp exp1a --iter $iter --cfg $cfg --n-pairs 50
    done
done

echo "$(date) | All patching done"
echo "$(date) | Results in patching/output/"
