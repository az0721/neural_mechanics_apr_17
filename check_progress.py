#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check extraction progress: how many users are done, partial results ready?

Usage:
    python check_progress.py              # summary
    python check_progress.py --details    # per-user breakdown
"""
import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PER_USER_DIR, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()

    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    n_emp = (users['is_employed'] == 1).sum()
    n_unemp = (users['is_employed'] == 0).sum()

    done_files = set(f.replace('.npz', '')
                     for f in os.listdir(PER_USER_DIR)
                     if f.endswith('.npz')) if os.path.exists(PER_USER_DIR) else set()

    done_emp = sum(1 for _, u in users.iterrows()
                   if u['cuebiq_id'] in done_files and u['is_employed'] == 1)
    done_unemp = sum(1 for _, u in users.iterrows()
                     if u['cuebiq_id'] in done_files and u['is_employed'] == 0)

    print(f"{'='*60}")
    print(f"Extraction Progress")
    print(f"{'='*60}")
    print(f"  Employed:   {done_emp:3d}/{n_emp} done")
    print(f"  Unemployed: {done_unemp:3d}/{n_unemp} done")
    print(f"  Total:      {len(done_files):3d}/{len(users)} done "
          f"({len(done_files)/len(users)*100:.0f}%)")

    if len(done_files) > 0:
        # Quick stats from completed files
        n_samples = 0
        n_oom = 0
        for f in list(done_files)[:10]:  # sample first 10
            path = os.path.join(PER_USER_DIR, f"{f}.npz")
            d = np.load(path, allow_pickle=True)
            n_samples += len(d['labels'])
            n_oom += sum(1 for a in d['answers'] if a.get('status') == 'oom')
        avg = n_samples / min(10, len(done_files))
        print(f"\n  Avg prompts/user: {avg:.0f}")
        print(f"  OOM (sampled): {n_oom}")
        print(f"  Est total samples: ~{avg * len(users):.0f}")

    if args.details:
        print(f"\nPending users:")
        for _, u in users.iterrows():
            uid = u['cuebiq_id']
            if uid not in done_files:
                emp = "emp" if u['is_employed'] else "unemp"
                print(f"  {uid[:16]}... ({emp})")


if __name__ == "__main__":
    main()
