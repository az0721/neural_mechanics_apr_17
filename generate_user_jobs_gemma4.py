#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate per-user SBATCH scripts for Gemma 4 31B-it (BnB 4-bit).
CoT-only, stops format, single H200.

Usage:
    python generate_user_jobs_gemma4.py                   # generate scripts
    python generate_user_jobs_gemma4.py --submit          # generate + submit
    python generate_user_jobs_gemma4.py --dry-run         # show plan
    python generate_user_jobs_gemma4.py --gpu-type v100   # try V100
    python generate_user_jobs_gemma4.py --align-v7        # V8 uses V7 targets
"""
import argparse, os, random, subprocess, time
import pandas as pd

from config import (
    DATA_DIR, LOGS_DIR, RANDOM_SEED, MODEL_REGISTRY,
    user_output_path_iter,
)

PROJECT = os.path.dirname(os.path.abspath(__file__))
MODEL_KEY = 'gemma4_31b'
SCRIPTS_DIR = os.path.join(PROJECT, 'scripts', 'gemma4_jobs')

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=g4_{uid_short}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output={logs}/g4_{uid_short}_%j.out
#SBATCH --error={logs}/g4_{uid_short}_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gemma4_env
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd {project}
python -u run_extract_per_user.py \\
    --user {uid} \\
    --model {model_key} \\
    --cot-only{align_flag}{iter_flag}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--align-v7', action='store_true',
                        help='V8 uses V7 targets for alignment')
    parser.add_argument('--partition', default='gpu',
                        choices=['gpu', 'multigpu', 'gpu-short'])
    parser.add_argument('--gpu-type', default='h200',
                        choices=['h200', 'a100'])
    parser.add_argument('--time-limit', default='2:00:00',
                        help='SLURM time limit (default 4h)')
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'],
                        help='Iterations to run')
    args = parser.parse_args()

    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    emp_ids = users[users['is_employed'] == 1]['cuebiq_id'].tolist()
    unemp_ids = users[users['is_employed'] == 0]['cuebiq_id'].tolist()

    random.seed(RANDOM_SEED)
    all_ids = emp_ids + unemp_ids
    random.shuffle(all_ids)

    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    scripts = []
    skipped = 0
    for uid in all_ids:
        # Skip if all requested iterations already done
        all_done = all(
            os.path.exists(user_output_path_iter(uid, MODEL_KEY, it))
            for it in args.iter)
        if all_done:
            skipped += 1
            continue

        uid_short = uid[:12]
        align_flag = ' --align-v7' if args.align_v7 else ''
        iter_flag = f" --iter {' '.join(args.iter)}" if args.iter != ['v7', 'v8'] else ''

        script = TEMPLATE.format(
            uid_short=uid_short, uid=uid,
            partition=args.partition,
            gpu_type=args.gpu_type,
            time_limit=args.time_limit,
            logs=LOGS_DIR, project=PROJECT,
            model_key=MODEL_KEY,
            align_flag=align_flag,
            iter_flag=iter_flag)

        path = os.path.join(SCRIPTS_DIR, f"g4_{uid_short}.sbatch")
        scripts.append((path, uid))
        if not args.dry_run:
            with open(path, 'w') as f:
                f.write(script)

    is_emp = users.set_index('cuebiq_id')['is_employed'].to_dict()
    n_emp = sum(1 for _, uid in scripts if is_emp.get(uid, 0))
    n_unemp = len(scripts) - n_emp

    print(f"{'='*60}")
    print(f"Gemma 4 31B Per-User Job Generator")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_KEY} ({MODEL_REGISTRY[MODEL_KEY]['tag']})")
    print(f"  Total users: {len(users)} (120 emp + 120 unemp)")
    print(f"  Already done: {skipped}")
    print(f"  To generate: {len(scripts)} ({n_emp} emp + {n_unemp} unemp)")
    print(f"  Partition: {args.partition}")
    print(f"  GPU type: {args.gpu_type}")
    print(f"  Time limit: {args.time_limit}")
    print(f"  Iterations: {', '.join(args.iter)}")
    print(f"  Format: stops (CoT-only)")
    print(f"  Scripts dir: {SCRIPTS_DIR}/")

    if args.dry_run:
        print(f"\n  DRY RUN — no files written")
        return

    if args.submit:
        print(f"\n  Submitting {len(scripts)} jobs...")
        for i, (path, uid) in enumerate(scripts):
            result = subprocess.run(['sbatch', path],
                                    capture_output=True, text=True)
            if 'Submitted' in result.stdout:
                jid = result.stdout.strip().split()[-1]
                tag = "emp" if is_emp.get(uid, 0) else "unemp"
                print(f"  [{i+1:3d}/{len(scripts)}] {uid[:12]} ({tag}) -> {jid}")
            else:
                print(f"  [{i+1:3d}/{len(scripts)}] FAIL: {result.stderr.strip()}")
            time.sleep(0.3)
        print(f"\n  All submitted. Monitor with: python check_progress.py")
    else:
        print(f"\n  To submit:")
        print(f"    python generate_user_jobs_gemma4.py --submit")
        print(f"  Or manually:")
        print(f"    for f in {SCRIPTS_DIR}/g4_*.sbatch; do sbatch $f; sleep 0.3; done")


if __name__ == "__main__":
    main()
