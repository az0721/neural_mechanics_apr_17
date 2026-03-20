#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-user extraction: hidden states + answers for all configs.
One user = one job. No checkpoints needed (~35 min per employed user).

Usage:
    python run_extract_per_user.py --user <cuebiq_id> [--model 12b]

Output:
    outputs_v7/per_user/{user_id}.npz
    Contains hidden_states, labels, meta, answers for ALL applicable configs.
"""
import sys, os, argparse, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, user_output_path, PER_USER_DIR,
                    MODEL_REGISTRY, config_label)
from utils import (load_data, load_model, build_user_prompts, process_sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='cuebiq_id')
    parser.add_argument('--model', default='12b', choices=['12b'])
    args = parser.parse_args()

    uid = args.user
    out_path = user_output_path(uid)

    # Skip if already done
    if os.path.exists(out_path):
        print(f"Already done: {out_path}")
        return

    t0 = time.time()
    print(f"{'='*60}")
    print(f"User: {uid}")
    print(f"Model: {MODEL_REGISTRY[args.model]['tag']}")
    print(f"{'='*60}")

    # --- Load data and build prompts (CPU, ~30s) ---
    df = load_data()
    user_df = df[df['cuebiq_id'] == uid]
    if len(user_df) == 0:
        print(f"ERROR: User {uid} not found in data")
        return

    is_employed = bool(user_df['is_employed'].iloc[0])
    print(f"  Employed: {is_employed}")

    prompts_info = build_user_prompts(user_df, uid, is_employed)
    n_prompts = len(prompts_info)
    if n_prompts == 0:
        print("  No prompts generated, skipping")
        return

    # Summarize what we'll run
    exp_cfg_counts = {}
    for p in prompts_info:
        key = f"{p['exp_name']}_cfg{p['config_id']}"
        exp_cfg_counts[key] = exp_cfg_counts.get(key, 0) + 1
    print(f"  Prompts: {n_prompts} total")
    for k, v in sorted(exp_cfg_counts.items()):
        print(f"    {k}: {v}")

    # --- Load model (GPU, ~2 min) ---
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size

    # --- Pre-allocate ---
    hidden_states = np.zeros((n_prompts, n_layers, hidden_dim), dtype=np.float32)
    labels = np.zeros(n_prompts, dtype=np.int32)
    meta_list = []
    answers_list = []

    # --- Process each prompt ---
    print(f"\n  Processing {n_prompts} prompts...")
    for i, pinfo in enumerate(prompts_info):
        t_start = time.time()

        try:
            hidden, answer = process_sample(
                model, tokenizer, pinfo['prompt'], pinfo['use_sys'], device)
            hidden_states[i] = hidden
            answer['status'] = 'ok'
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i}/{n_prompts}] OOM — skipping")
            torch.cuda.empty_cache()
            answer = {'status': 'oom', 'top_tokens': [], 'top_probs': [],
                      'generated_text': '', 'parsed_answer': '', 'n_tokens': 0}

        labels[i] = pinfo['label']
        meta_list.append(pinfo['meta'])
        answers_list.append(answer)

        elapsed = time.time() - t_start
        if (i + 1) % 5 == 0 or i == 0:
            pct = (i + 1) / n_prompts * 100
            eta = elapsed * (n_prompts - i - 1) / 60
            exp_cfg = f"{pinfo['exp_name']}_cfg{pinfo['config_id']}"
            print(f"  [{i+1:3d}/{n_prompts}] {pct:5.1f}% | {elapsed:.0f}s | "
                  f"ETA {eta:.0f}min | {exp_cfg}")

    # --- Save ---
    os.makedirs(PER_USER_DIR, exist_ok=True)
    np.savez(out_path,
             hidden_states=hidden_states,
             labels=labels,
             meta=np.array(meta_list, dtype=object),
             answers=np.array(answers_list, dtype=object))

    total_time = (time.time() - t0) / 60
    n_oom = sum(1 for a in answers_list if a.get('status') == 'oom')
    print(f"\n{'='*60}")
    print(f"Done: {uid} | {n_prompts} prompts | {total_time:.1f} min | "
          f"OOM: {n_oom}")
    print(f"Saved: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import torch
    main()
