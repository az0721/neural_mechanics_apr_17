#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-build all steering prompts on CPU and save to data/steering_prompts/.
Run BEFORE submitting GPU steering jobs.

Builds: 20 users × 2 cfgs × 4 questions = 160 prompt strings.
Each prompt = GPS body + question suffix (unwrapped, ~34K tokens worth of text).
The GPU runner will apply chat template + tokenize.

Output per user:
  data/steering_prompts/{uid[:32]}.json
  {
    "meta": {"user": ..., "is_employed": ..., "date": ...},
    "prompts": {
      "(5, 'behavioral')": "Your task is to analyze...",
      "(5, 'binary')": "Your task is to analyze...",
      ...
    }
  }

Usage:
    python steering/prebuild_steering_prompts.py
    python steering/prebuild_steering_prompts.py --n-per-class 10
"""
import sys, os, argparse, random, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (DATA_DIR, EXP_CONFIG, CONFIG_MATRIX, SAMPLING_CONFIG,
                    RANDOM_SEED)
from utils import (load_data, _targets_exp2, _filter_window,
                   _fmt_geo, _fmt_dow, _ts, _history_exp2)
from steering.prompts import STEERING_PROMPTS
import pandas as pd

PROMPT_DIR = os.path.join(DATA_DIR, 'steering_prompts')
DEFAULT_EXP = 'exp2'
DEFAULT_CFGS = [5, 6]


def build_gps_body(user_df, target_date, cfg_dict):
    """Build GPS history + context (shared across all questions for a cfg)."""
    exp_cfg = EXP_CONFIG[DEFAULT_EXP]
    gh, dh = cfg_dict['geo_hash'], cfg_dict['day_hash']

    hist = _history_exp2(user_df, target_date, exp_cfg['hist_window'], gh, dh)
    if hist is None:
        return None

    ctx = _filter_window(user_df[user_df['date'] == target_date], exp_cfg['time_window'])
    if len(ctx) == 0:
        return None

    ctx_lines = []
    for _, r in ctx.iterrows():
        ctx_lines.append(f"{_fmt_dow(r['dow'], dh)}, {_ts(r)}, {_fmt_geo(r['geo_id'], gh)}")

    return (
        f"Your task is to analyze an individual's mobility pattern "
        f"based on their location history.\n"
        f"Each record: {exp_cfg['prompt_fmt']}\n\n"
        f"=== Mobility History ===\n"
        + "\n".join(hist) + "\n\n"
        f"=== Current Day Context ===\n"
        + "\n".join(ctx_lines) + "\n\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-per-class', type=int, default=10)
    parser.add_argument('--cfgs', nargs='+', type=int, default=DEFAULT_CFGS)
    args = parser.parse_args()

    os.makedirs(PROMPT_DIR, exist_ok=True)

    t0 = time.time()
    print(f"{'='*60}")
    print(f"Pre-building steering prompts → {PROMPT_DIR}")
    print(f"{'='*60}")

    # Load data
    df = load_data()
    users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))

    # Select same 20 users as Phase 2a (same seed)
    random.seed(RANDOM_SEED + 999)
    emp = random.sample(users_df[users_df['is_employed'] == 1]['cuebiq_id'].tolist(),
                        args.n_per_class)
    unemp = random.sample(users_df[users_df['is_employed'] == 0]['cuebiq_id'].tolist(),
                          args.n_per_class)
    all_users = emp + unemp

    cfg_dicts = {c['id']: c for c in CONFIG_MATRIX[DEFAULT_EXP]}
    n_built = 0
    n_skipped = 0

    for uid in all_users:
        user_df = df[df['cuebiq_id'] == uid]
        is_emp = bool(users_df[users_df['cuebiq_id'] == uid].iloc[0]['is_employed'])
        exp_df = user_df.query(EXP_CONFIG[DEFAULT_EXP]['filter_query'])

        random.seed(RANDOM_SEED + hash(uid) % 10000)
        targets = _targets_exp2(exp_df)
        if targets is None:
            print(f"  {uid[:16]}: no targets, skipping")
            n_skipped += 1
            continue
        tdate = targets[0]

        prompts = {}
        for cfg_id in args.cfgs:
            # Use same seed as run_steering_per_user.py
            random.seed(RANDOM_SEED + hash(f"{uid}_{tdate}_12:00") % 10000)
            gps_body = build_gps_body(exp_df, tdate, cfg_dicts[cfg_id])
            if gps_body is None:
                print(f"  {uid[:16]}: no GPS body for cfg{cfg_id}")
                continue

            for qkey in STEERING_PROMPTS:
                full_text = gps_body + STEERING_PROMPTS[qkey]['question']
                prompts[str((cfg_id, qkey))] = full_text

        save_data = {
            'meta': {'user': uid, 'is_employed': is_emp, 'date': tdate,
                     'cfgs': args.cfgs},
            'prompts': prompts,
        }
        out_path = os.path.join(PROMPT_DIR, f"{uid[:32]}.json")
        with open(out_path, 'w') as f:
            json.dump(save_data, f)

        n_built += 1
        tag = 'EMP' if is_emp else 'UNE'
        print(f"  {uid[:16]} ({tag}): {len(prompts)} prompts, "
              f"~{len(list(prompts.values())[0])//4:,} tokens/prompt")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: {n_built} users, {n_skipped} skipped, {elapsed:.0f}s")
    print(f"Saved to: {PROMPT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
