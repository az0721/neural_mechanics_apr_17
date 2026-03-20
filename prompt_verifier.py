#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify prompt building for V7 (15-min intervals).
Checks token counts, line counts, format, and sample balance.

Usage:
    python prompt_verifier.py --user <uid>           # one user
    python prompt_verifier.py --summary              # all users stats
"""
import sys, os, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_DIR, EXP_CONFIG, CONFIG_MATRIX, RANDOM_SEED,
                    TIME_INTERVAL, config_label)
from utils import load_data, build_user_prompts, wrap_with_chat_template
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', help='Specific user ID to verify')
    parser.add_argument('--summary', action='store_true',
                        help='Summary stats across all users')
    parser.add_argument('--n-users', type=int, default=10,
                        help='Number of users to sample for summary')
    args = parser.parse_args()

    df = load_data()
    import pandas as pd
    users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    tokenizer = AutoTokenizer.from_pretrained(
        'google/gemma-3-12b-it', local_files_only=True)

    if args.user:
        verify_user(df, users_df, tokenizer, args.user)
    elif args.summary:
        verify_summary(df, users_df, tokenizer, args.n_users)
    else:
        parser.print_help()


def verify_user(df, users_df, tokenizer, uid):
    user_df = df[df['cuebiq_id'] == uid]
    user_info = users_df[users_df['cuebiq_id'] == uid].iloc[0]
    is_emp = bool(user_info['is_employed'])

    print(f"{'='*60}")
    print(f"User: {uid[:16]}... | Employed: {is_emp}")
    print(f"Time interval: {TIME_INTERVAL} min")
    print(f"{'='*60}")

    prompts_info = build_user_prompts(user_df, uid, is_emp)
    print(f"Total prompts: {len(prompts_info)}")

    for i, pinfo in enumerate(prompts_info):
        full = wrap_with_chat_template(tokenizer, pinfo['prompt'], pinfo['use_sys'])
        toks = tokenizer(full, return_tensors='pt', truncation=True, max_length=128000)
        n_tok = toks['input_ids'].shape[1]
        n_lines = pinfo['prompt'].count('\n')
        exp_cfg = f"{pinfo['exp_name']}_cfg{pinfo['config_id']}"
        meta = pinfo['meta']
        cot = "CoT" if pinfo['use_sys'] else "no"

        if i < 5 or i == len(prompts_info) - 1:
            print(f"  [{i:2d}] {exp_cfg:20s} | date={meta['date']} "
                  f"t={meta['pred_time']} | {n_lines:,} lines | "
                  f"{n_tok:,} tok | {cot}")

    # Token stats
    all_tok = []
    for pinfo in prompts_info:
        full = wrap_with_chat_template(tokenizer, pinfo['prompt'], pinfo['use_sys'])
        toks = tokenizer(full, return_tensors='pt', truncation=True, max_length=128000)
        all_tok.append(toks['input_ids'].shape[1])

    all_tok = np.array(all_tok)
    print(f"\nToken stats: min={all_tok.min():,} max={all_tok.max():,} "
          f"mean={all_tok.mean():,.0f}")
    print(f"Headroom (128K): {128000 - all_tok.max():,} tokens")

    # Expected: 15-min = ~38K max
    if all_tok.max() > 100000:
        print(f"WARNING: max tokens {all_tok.max():,} > 100K!")
    else:
        print("PASS: all tokens < 100K")


def verify_summary(df, users_df, tokenizer, n_sample):
    random.seed(RANDOM_SEED)
    all_uids = users_df['cuebiq_id'].tolist()
    sample_uids = random.sample(all_uids, min(n_sample, len(all_uids)))

    print(f"Sampling {len(sample_uids)} users for summary...")
    total_prompts = {'emp': 0, 'unemp': 0}
    all_tokens = []

    for uid in sample_uids:
        user_df = df[df['cuebiq_id'] == uid]
        user_info = users_df[users_df['cuebiq_id'] == uid].iloc[0]
        is_emp = bool(user_info['is_employed'])

        prompts = build_user_prompts(user_df, uid, is_emp)
        key = 'emp' if is_emp else 'unemp'
        total_prompts[key] += len(prompts)

        for p in prompts[:3]:  # Sample a few for token counts
            full = wrap_with_chat_template(tokenizer, p['prompt'], p['use_sys'])
            toks = tokenizer(full, return_tensors='pt', truncation=True, max_length=128000)
            all_tokens.append(toks['input_ids'].shape[1])

    all_tokens = np.array(all_tokens)
    n_emp = sum(1 for uid in sample_uids
                if bool(users_df[users_df['cuebiq_id'] == uid].iloc[0]['is_employed']))
    n_unemp = len(sample_uids) - n_emp

    print(f"\n{'='*60}")
    print(f"Summary ({len(sample_uids)} users: {n_emp} emp + {n_unemp} unemp)")
    print(f"Time interval: {TIME_INTERVAL} min")
    print(f"{'='*60}")
    print(f"Prompts/emp user:   ~{total_prompts['emp'] // max(n_emp,1)}")
    print(f"Prompts/unemp user: ~{total_prompts['unemp'] // max(n_unemp,1)}")
    print(f"Est total (240 users): "
          f"~{total_prompts['emp'] // max(n_emp,1) * 120 + total_prompts['unemp'] // max(n_unemp,1) * 120}")
    print(f"\nToken stats (sampled):")
    print(f"  min={all_tokens.min():,}  max={all_tokens.max():,}  "
          f"mean={all_tokens.mean():,.0f}")
    print(f"  Headroom: {128000 - all_tokens.max():,} tokens")

    if all_tokens.max() > 100000:
        print(f"\nWARNING: max {all_tokens.max():,} tokens — may OOM on some GPUs")
    else:
        print(f"\nPASS: all sampled tokens < 100K (15-min working as expected)")


if __name__ == "__main__":
    main()
