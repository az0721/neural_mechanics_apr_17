#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate per-user SBATCH scripts for all 240 users.
Supports multiple models via --model flag.

Usage:
    python generate_user_jobs.py --model gemma4_26b              # generate scripts
    python generate_user_jobs.py --model gemma4_26b --submit     # generate + submit
    python generate_user_jobs.py --model 12b --iter v8           # single iteration
"""
import argparse, os, random, subprocess, time
import pandas as pd

from config import (DATA_DIR, SCRIPTS_DIR, LOGS_DIR, RANDOM_SEED,
                    MODEL_REGISTRY, user_output_path_iter,
                    get_iter_per_user_dir)

PROJECT = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=u_{uid_short}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:h200:1
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output={logs}/u_{uid_short}_%j.out
#SBATCH --error={logs}/u_{uid_short}_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate {conda_env}
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd {project}
python -u run_extract_per_user.py --user {uid} --model {model_key}{iter_flag}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma4_26b',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'],
                        help='Iterations (default: v7 v8)')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--partition', default='multigpu',
                        choices=['multigpu', 'gpu', 'gpu-short'])
    parser.add_argument('--time-limit', default='3:00:00',
                        help='Per-job time limit (default: 3h for dual iter)')
    args = parser.parse_args()

    model_key = args.model
    model_cfg = MODEL_REGISTRY[model_key]
    conda_env = model_cfg['conda_env']
    iters = args.iter
    iter_flag = ' --iter ' + ' '.join(iters) if iters != ['v7', 'v8'] else ''

    # Load users
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    emp_ids = users[users['is_employed'] == 1]['cuebiq_id'].tolist()
    unemp_ids = users[users['is_employed'] == 0]['cuebiq_id'].tolist()
    random.seed(RANDOM_SEED)
    all_ids = emp_ids + unemp_ids
    random.shuffle(all_ids)

    # Model-specific script/log dirs
    script_dir = os.path.join(SCRIPTS_DIR, model_cfg['tag'])
    log_dir = os.path.join(LOGS_DIR, model_cfg['tag'])
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    scripts = []
    skipped = 0
    for uid in all_ids:
        # Skip if ALL requested iterations are done
        all_done = all(
            os.path.exists(user_output_path_iter(uid, model_key, it))
            for it in iters)
        if all_done:
            skipped += 1
            continue

        uid_short = uid[:12]
        script = TEMPLATE.format(
            uid_short=uid_short, uid=uid,
            partition=args.partition,
            time_limit=args.time_limit,
            logs=log_dir, project=PROJECT,
            conda_env=conda_env,
            model_key=model_key,
            iter_flag=iter_flag)

        path = os.path.join(script_dir, f"u_{uid_short}.sbatch")
        scripts.append((path, uid))
        if not args.dry_run:
            with open(path, 'w') as f:
                f.write(script)

    is_emp = users.set_index('cuebiq_id')['is_employed'].to_dict()
    n_emp = sum(1 for _, uid in scripts if is_emp.get(uid, 0))
    n_unemp = len(scripts) - n_emp

    print(f"{'='*60}")
    print(f"Job Generator: {model_cfg['tag']}")
    print(f"{'='*60}")
    print(f"  Model: {model_key} ({model_cfg['tag']})")
    print(f"  Conda env: {conda_env}")
    print(f"  Iterations: {iters}")
    print(f"  Total users: {len(users)} (120 emp + 120 unemp)")
    print(f"  Already done: {skipped}")
    print(f"  To generate: {len(scripts)} ({n_emp} emp + {n_unemp} unemp)")
    print(f"  Partition: {args.partition}")
    print(f"  Time limit: {args.time_limit}")
    print(f"  Scripts → {script_dir}/")

    if args.dry_run:
        print("\n  DRY RUN — no files written")
        return

    if args.submit:
        print(f"\n  Submitting {len(scripts)} jobs...")
        for i, (path, uid) in enumerate(scripts):
            result = subprocess.run(['sbatch', path],
                                    capture_output=True, text=True)
            if 'Submitted' in result.stdout:
                jid = result.stdout.strip().split()[-1]
                tag = "emp" if is_emp.get(uid, 0) else "unemp"
                print(f"  [{i+1:3d}/{len(scripts)}] {uid[:12]} ({tag}) → {jid}")
            else:
                print(f"  [{i+1:3d}/{len(scripts)}] FAIL: {result.stderr.strip()}")
            time.sleep(0.3)
        print(f"\n  All submitted. Monitor with: python check_progress.py")
    else:
        print(f"\n  To submit:")
        print(f"    python generate_user_jobs.py --model {model_key} --submit")
        print(f"  Or manually:")
        print(f"    for f in {script_dir}/u_*.sbatch; do sbatch $f; sleep 0.3; done")


if __name__ == "__main__":
    main()
