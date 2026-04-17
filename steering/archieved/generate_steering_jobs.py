#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate sbatch scripts for steering Phase 2b.
20 users, one job per user. Partition set by auto-submitter at runtime.

Usage:
    python steering/generate_steering_jobs.py
"""
import sys, os, random, argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, RANDOM_SEED

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'steering_jobs_v2')
WORK_DIR = '/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7'

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=st2_{short_uid}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/st2_{short_uid}_%j.out
#SBATCH --error=logs/st2_{short_uid}_%j.err

cd {work_dir}
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate unsloth_env
module purge && module load cuda/12.1.1
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u steering/run_steering_per_user.py --user {uid} --cfgs {cfgs}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-per-class', type=int, default=10)
    parser.add_argument('--cfgs', nargs='+', type=int, default=[5, 6])
    args = parser.parse_args()

    os.makedirs(SCRIPTS_DIR, exist_ok=True)

    users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    random.seed(RANDOM_SEED + 999)  # same 20 users
    emp = random.sample(users_df[users_df['is_employed'] == 1]['cuebiq_id'].tolist(),
                        args.n_per_class)
    unemp = random.sample(users_df[users_df['is_employed'] == 0]['cuebiq_id'].tolist(),
                          args.n_per_class)
    all_users = emp + unemp
    cfgs_str = ' '.join(str(c) for c in args.cfgs)

    for uid in all_users:
        short = uid[:12]
        script = SBATCH_TEMPLATE.format(
            short_uid=short, work_dir=WORK_DIR, uid=uid, cfgs=cfgs_str)
        path = os.path.join(SCRIPTS_DIR, f"st2_{short}.sbatch")
        with open(path, 'w') as f:
            f.write(script)

    print(f"Generated {len(all_users)} sbatch scripts in {SCRIPTS_DIR}/")
    print(f"  Employed: {len(emp)}, Unemployed: {len(unemp)}")
    print(f"  Configs: {args.cfgs}")
    print(f"  Time: 8h, GPU: H200")
    print(f"  Output dir: outputs_v7/steering/per_user_v2/")

    # Save user list
    with open(os.path.join(os.path.dirname(__file__), 'user_list.txt'), 'w') as f:
        for uid in emp:
            f.write(f"{uid},1\n")
        for uid in unemp:
            f.write(f"{uid},0\n")
    print(f"  User list: steering/user_list.txt")


if __name__ == "__main__":
    main()
