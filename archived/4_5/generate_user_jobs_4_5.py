#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate per-user SBATCH scripts for all 240 users.
All jobs → multigpu partition with H200 (V100 can't do bfloat16).
~35 min/employed user, ~18 min/unemployed user, request 2h for safety.
Shuffles user order so partial results are representative.

Usage:
    python generate_user_jobs.py              # generate scripts
    python generate_user_jobs.py --submit     # generate + submit all
    python generate_user_jobs.py --dry-run    # show plan only
"""
import argparse, os, random, subprocess, time
import pandas as pd

from config import DATA_DIR, SCRIPTS_DIR, LOGS_DIR, PER_USER_DIR, RANDOM_SEED, user_output_path

PROJECT = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=u_{uid_short}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output={logs}/u_{uid_short}_%j.out
#SBATCH --error={logs}/u_{uid_short}_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
cd {project}
python -u run_extract_per_user.py --user {uid}{align_flag}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--align-v7', action='store_true',
                        help='Add --align-v7 flag to extraction scripts')
    parser.add_argument('--partition', default='multigpu',
                        choices=['multigpu', 'gpu', 'gpu-short'],
                        help='Default partition for all jobs')
    args = parser.parse_args()

    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    emp_ids = users[users['is_employed'] == 1]['cuebiq_id'].tolist()
    unemp_ids = users[users['is_employed'] == 0]['cuebiq_id'].tolist()

    # Shuffle for early partial results (Matteo's suggestion)
    random.seed(RANDOM_SEED)
    all_ids = emp_ids + unemp_ids
    random.shuffle(all_ids)

    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    scripts = []
    skipped = 0
    for uid in all_ids:
        if os.path.exists(user_output_path(uid)):
            skipped += 1
            continue
        uid_short = uid[:12]
        align_flag = ' --align-v7' if args.align_v7 else ''
        script = TEMPLATE.format(
            uid_short=uid_short, uid=uid,
            partition=args.partition,
            logs=LOGS_DIR, project=PROJECT,
            align_flag=align_flag)
        path = os.path.join(SCRIPTS_DIR, f"u_{uid_short}.sbatch")
        scripts.append((path, uid))
        if not args.dry_run:
            with open(path, 'w') as f:
                f.write(script)

    is_emp = users.set_index('cuebiq_id')['is_employed'].to_dict()
    n_emp = sum(1 for _, uid in scripts if is_emp.get(uid, 0))
    n_unemp = len(scripts) - n_emp

    print(f"{'='*60}")
    print(f"V7 Per-User Job Generator")
    print(f"{'='*60}")
    print(f"  Total users: {len(users)} (120 emp + 120 unemp)")
    print(f"  Already done: {skipped}")
    print(f"  To generate: {len(scripts)} ({n_emp} emp + {n_unemp} unemp)")
    print(f"  Partition: {args.partition}")
    print(f"  Time limit: 2h per job")
    print(f"  Est time/emp user: ~35 min (64 prompts)")
    print(f"  Est time/unemp user: ~18 min (32 prompts)")

    if args.dry_run:
        print("\n  DRY RUN — no files written")
        return

    print(f"\n  Scripts → {SCRIPTS_DIR}/")

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
        print(f"    python generate_user_jobs.py --submit")
        print(f"  Or manually:")
        print(f"    for f in {SCRIPTS_DIR}/u_*.sbatch; do sbatch $f; sleep 0.3; done")


if __name__ == "__main__":
    main()
