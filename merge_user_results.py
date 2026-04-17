#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge per-user .npz files into per-config .npz + answers JSON.
Supports both Gemma 3 12B and Gemma 4 31B.

Usage (Gemma 3, legacy):
    python merge_user_results.py --model 12b --iter v7
    python merge_user_results.py --model 12b --iter v8

Usage (Gemma 4):
    python merge_user_results.py --model gemma4_31b --iter v7
    python merge_user_results.py --model gemma4_31b --iter v8
    python merge_user_results.py --model gemma4_31b --iter v7 --exp exp2
"""
import sys, os, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EXP_CONFIG, CONFIG_MATRIX, MODEL_REGISTRY,
    get_iter_per_user_dir, get_iter_model_dirs, config_label,
)
from utils import parse_answer


def load_all_users(per_user_dir, model_key='12b'):
    """Load all per-user .npz files from the given directory."""
    files = sorted(f for f in os.listdir(per_user_dir) if f.endswith('.npz'))
    print(f"Found {len(files)} per-user files in {per_user_dir}")

    all_hidden, all_labels, all_meta, all_answers = [], [], [], []
    for f in files:
        d = np.load(os.path.join(per_user_dir, f), allow_pickle=True)
        all_hidden.append(d['hidden_states'])
        all_labels.append(d['labels'])
        all_meta.extend(d['meta'].tolist())
        all_answers.extend(d['answers'].tolist())

    hidden = np.concatenate(all_hidden, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total samples: {len(labels):,}")
    print(f"Hidden shape: {hidden.shape}")
    return hidden, labels, all_meta, all_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--iter', default='v7', choices=['v7', 'v8'],
                        help='Which iteration to merge')
    parser.add_argument('--require-all', action='store_true',
                        help='Only run if all 240 users are done')
    args = parser.parse_args()

    per_user_dir = get_iter_per_user_dir(args.model, args.iter)
    dirs = get_iter_model_dirs(args.model, args.iter)

    files = [f for f in os.listdir(per_user_dir) if f.endswith('.npz')]
    print(f"{'='*60}")
    print(f"Merge: model={args.model} iter={args.iter}")
    print(f"Per-user dir: {per_user_dir}")
    print(f"Per-user files: {len(files)}/240")
    print(f"Output: {dirs['hidden']}")
    print(f"{'='*60}")

    if args.require_all and len(files) < 240:
        print("Not all users done. Use without --require-all for partial merge.")
        return

    hidden, labels, meta, answers = load_all_users(per_user_dir, args.model)

    for exp in args.exp:
        configs = CONFIG_MATRIX[exp]
        print(f"\n{'='*60}")
        print(f"Merging: {exp}")
        print(f"{'='*60}")

        for cfg in configs:
            cfg_id = cfg['id']
            clabel = config_label(exp, cfg_id)

            mask = [(m['exp_name'] == exp and m['config_id'] == cfg_id)
                    for m in meta]
            idx = np.where(mask)[0]

            if len(idx) == 0:
                print(f"  cfg{cfg_id} ({clabel}): no samples")
                continue

            h = hidden[idx]
            y = labels[idx]
            m = [meta[i] for i in idx]
            a = [answers[i] for i in idx]

            n_oom = sum(1 for ai in a if ai.get('status') == 'oom')
            users = set(mi['user'] for mi in m)

            tag = f"{exp}_cfg{cfg_id}"
            npz_path = os.path.join(dirs['hidden'], f"{tag}.npz")
            np.savez(npz_path,
                     hidden_states=h, labels=y,
                     meta=np.array(m, dtype=object))

            combined = []
            for i, (ai, mi) in enumerate(zip(a, m)):
                ai['parsed_answer'] = parse_answer(
                    ai.get('generated_text', ''), args.model)
                combined.append({**mi, 'label': int(y[i]), **ai})
            json_path = os.path.join(dirs['answers'], f"{tag}_answers.json")
            with open(json_path, 'w') as f:
                json.dump(combined, f, indent=2)

            print(f"  cfg{cfg_id} ({clabel}): {len(idx)} samples, "
                  f"{len(users)} users, {n_oom} OOM")
            print(f"    → {npz_path}")
            print(f"    → {json_path}")

    print(f"\n{'='*60}")
    print(f"Done. Outputs in: {dirs['hidden']}")
    print(f"Answers in:       {dirs['answers']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()