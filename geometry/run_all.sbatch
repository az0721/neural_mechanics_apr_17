#!/bin/bash
#SBATCH --job-name=geo_all
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/geo_all_%j.out
#SBATCH --error=logs/geo_all_%j.err

set -euo pipefail
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
mkdir -p geometry/outputs logs

echo "$(date) | Starting geometry analysis"
echo "============================================================"

# ── Script 1: PCA + DA + Sep (~5 min each, no GPU needed) ──
for exp in exp1a exp2; do
    for iter in v7 v8; do
        echo "$(date) | PCA+DA: ${iter}/${exp}"
        python -u geometry/run_pca_da.py --exp $exp --iter $iter
    done
done

# ── Script 3: Direction Analysis (~10-15 min each) ──
for exp in exp1a exp2; do
    for iter in v7 v8; do
        echo "$(date) | Direction analysis: ${iter}/${exp}"
        python -u geometry/run_direction_analysis.py --exp $exp --iter $iter
    done
done

# ── Script 2: t-SNE + UMAP (~1-2h each — run last) ──
for exp in exp1a exp2; do
    for iter in v7 v8; do
        echo "$(date) | t-SNE+UMAP: ${iter}/${exp}"
        python -u geometry/run_tsne_umap.py --exp $exp --iter $iter
    done
done

echo "$(date) | ALL DONE"
echo "============================================================"
echo "Outputs in: geometry/outputs/"
ls -la geometry/outputs/*.pdf 2>/dev/null || echo "No PDFs found"
