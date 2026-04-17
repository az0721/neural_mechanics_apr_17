#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate sbatch scripts for NL-based steering experiments.

Gemma 4 31B: 16 jobs = 2 iters × (4 exp2 cfgs + 2 exp1a cfgs) × 2 = wait no
  exp2 CoT-only: cfg 2,4,6,8 → 4 cfgs × 2 iters = 8
  exp1a CoT-only: cfg 2,4         → 2 cfgs × 2 iters = 4
  Total: 12 jobs

Each job: ~20 questions × 60 layers × 10 passes ≈ 12,000 fwd passes ≈ 60-90 min

Usage:
    python steering/generate_steering_jobs_nl.py --model gemma4_31b
    python steering/generate_steering_jobs_nl.py --model 12b
"""
import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_REGISTRY, COT_ONLY_IDS, get_iter_output_dir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts', 'nl_steering_jobs')
WORK_DIR = '/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7'

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=nl_{model_short}_{iter}_{exp}_c{cfg}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
#SBATCH --output={work_dir}/logs/nl_{model_short}_{iter}_{exp}_c{cfg}_%j.out
#SBATCH --error={work_dir}/logs/nl_{model_short}_{iter}_{exp}_c{cfg}_%j.err

set -euo pipefail
cd {work_dir}
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate {conda_env}
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "$(date) | NL Steering: {model} {iter} {exp} cfg{cfg}"

# Step 1: Compute vectors (CPU, fast)
python -u steering/compute_vectors_nl.py --model {model} --iter {iter} --exp {exp} --cfgs {cfg}

# Step 2: Run steering (GPU)
python -u steering/run_steering_nl.py --model {model} --iter {iter} --exp {exp} --cfg {cfg}

echo "$(date) | Done"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma4_31b',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--partition', default='gpu')
    parser.add_argument('--gpu-type', default='h200')
    parser.add_argument('--time-limit', default='00:30:00')
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    conda_env = cfg['conda_env']
    model_short = 'g4' if 'gemma4' in args.model else 'g3'
    mem = '64G' if 'gemma4' in args.model else '14G'

    # Build all combos: iter × exp × CoT-only cfgs
    combos = []
    for it in ['v7', 'v8']:
        for exp in ['exp2', 'exp1a']:
            for cfg_id in COT_ONLY_IDS[exp]:
                combos.append((it, exp, cfg_id))

    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

    scripts = []
    skipped = 0
    for it, exp, cfg_id in combos:
        # Check if output already exists
        # out_base = get_iter_output_dir(args.model, it)
        # out_path = os.path.join(out_base, 'steering', 'nl_results',
        #                         f"{exp}_cfg{cfg_id}_nl.json")
        # if os.path.exists(out_path):
        #     skipped += 1
        #     continue

        script = SBATCH_TEMPLATE.format(
            model_short=model_short,
            iter=it, exp=exp, cfg=cfg_id,
            model=args.model,
            partition=args.partition,
            gpu_type=args.gpu_type,
            time_limit=args.time_limit,
            mem=mem,
            conda_env=conda_env,
            work_dir=WORK_DIR,
        )

        fname = f"nl_{model_short}_{it}_{exp}_c{cfg_id}.sbatch"
        path = os.path.join(SCRIPTS_DIR, fname)
        with open(path, 'w') as f:
            f.write(script)
        scripts.append((path, it, exp, cfg_id))

    print(f"{'='*60}")
    print(f"NL Steering Job Generator")
    print(f"{'='*60}")
    print(f"  Model: {args.model} ({cfg['tag']})")
    print(f"  Conda env: {conda_env}")
    print(f"  Partition: {args.partition} ({args.gpu_type})")
    print(f"  Time: {args.time_limit}")
    print(f"  Total combos: {len(combos)}")
    print(f"  Already done: {skipped}")
    print(f"  Generated: {len(scripts)} scripts")
    print(f"  Scripts dir: {SCRIPTS_DIR}/")
    print()

    for path, it, exp, cfg_id in scripts:
        print(f"  {os.path.basename(path)}: {it}/{exp}/cfg{cfg_id}")

    print(f"\nTo submit all:")
    print(f"  for f in {SCRIPTS_DIR}/nl_{model_short}_*.sbatch; do sbatch $f; sleep 0.3; done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()