#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge per-user .npz files into per-config .npz files.
This creates the format that prob_results.py and geometry scripts expect.

Also saves per-config answers JSON for prediction_results.py.

Usage:
    python merge_user_results.py                     # all available
    python merge_user_results.py --exp exp2          # exp2 only
    python merge_user_results.py --require-all       # only if all users done
"""
import sys, os, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (PER_USER_DIR, EXP_CONFIG, CONFIG_MATRIX,
                    get_model_dirs, config_label)
from utils import parse_answer 


def load_all_users():
    """Load all per-user .npz files, return combined data."""
    files = sorted(f for f in os.listdir(PER_USER_DIR) if f.endswith('.npz'))
    print(f"Found {len(files)} per-user files")

    all_hidden, all_labels, all_meta, all_answers = [], [], [], []
    for f in files:
        d = np.load(os.path.join(PER_USER_DIR, f), allow_pickle=True)
        all_hidden.append(d['hidden_states'])
        all_labels.append(d['labels'])
        all_meta.extend(d['meta'].tolist())
        all_answers.extend(d['answers'].tolist())

    hidden = np.concatenate(all_hidden, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total samples: {len(labels):,}")
    return hidden, labels, all_meta, all_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--model', default='12b')
    parser.add_argument('--require-all', action='store_true',
                        help='Only run if all 240 users are done')
    args = parser.parse_args()

    # Check completeness
    files = [f for f in os.listdir(PER_USER_DIR) if f.endswith('.npz')]
    print(f"Per-user files: {len(files)}/240")
    if args.require_all and len(files) < 240:
        print("Not all users done. Use without --require-all for partial merge.")
        return

    # Load everything
    hidden, labels, meta, answers = load_all_users()
    dirs = get_model_dirs(args.model)

    for exp in args.exp:
        configs = CONFIG_MATRIX[exp]
        print(f"\n{'='*60}")
        print(f"Merging: {exp}")
        print(f"{'='*60}")

        for cfg in configs:
            cfg_id = cfg['id']
            clabel = config_label(exp, cfg_id)

            # Filter to this (exp, config)
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

            # Count stats
            n_oom = sum(1 for ai in a if ai.get('status') == 'oom')
            users = set(mi['user'] for mi in m)

            # Save hidden states .npz
            tag = f"{exp}_cfg{cfg_id}"
            npz_path = os.path.join(dirs['hidden'], f"{tag}.npz")
            np.savez(npz_path,
                     hidden_states=h, labels=y,
                     meta=np.array(m, dtype=object))

            # Save answers JSON (re-parse with latest parse_answer)
            combined = []
            for i, (ai, mi) in enumerate(zip(a, m)):
                ai['parsed_answer'] = parse_answer(ai.get('generated_text', ''))
                combined.append({**mi, 'label': int(y[i]), **ai})
            json_path = os.path.join(dirs['answers'], f"{tag}_answers.json")
            with open(json_path, 'w') as f:
                json.dump(combined, f, indent=2)

            print(f"  cfg{cfg_id} ({clabel}): {len(idx)} samples, "
                  f"{len(users)} users, {n_oom} OOM")
            print(f"    → {npz_path}")
            print(f"    → {json_path}")

    print(f"\nDone. Analysis scripts can now read from {dirs['hidden']}/")


if __name__ == "__main__":
    main()
