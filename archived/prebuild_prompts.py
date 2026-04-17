#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-build all prompts and save as compressed JSON.
Run BEFORE submitting GPU extraction jobs.

Usage:
    python prebuild_prompts.py                          # all exps, all configs
    python prebuild_prompts.py --exp exp2 --config 1    # specific
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EXP_CONFIG, CONFIG_MATRIX, ACTIVE_EXPERIMENTS
from utils import load_data, generate_samples, save_prebuilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=ACTIVE_EXPERIMENTS,
                        choices=['exp1a', 'exp1b', 'exp2'])
    parser.add_argument('--config', default='all', help='Config ID or "all"')
    args = parser.parse_args()

    df = load_data()

    for exp in args.exp:
        config_ids = ([c['id'] for c in CONFIG_MATRIX[exp]]
                      if args.config == 'all' else [int(args.config)])

        for cfg_id in config_ids:
            print(f"\n{'='*60}")
            print(f"Building: {exp} cfg{cfg_id}")
            print(f"{'='*60}")

            prompts, labels, meta = generate_samples(df, exp, cfg_id)
            if len(prompts) == 0:
                print("  No samples, skipping")
                continue

            # Token stats
            lens = [len(p) for p in prompts]
            print(f"  Chars: min={min(lens):,}, max={max(lens):,}, "
                  f"mean={sum(lens)/len(lens):,.0f}")
            print(f"  Est tokens: ~{max(lens)//4:,} (max chars/4)")

            save_prebuilt(prompts, labels, meta, exp, cfg_id)

    print(f"\nAll done.")


if __name__ == "__main__":
    main()