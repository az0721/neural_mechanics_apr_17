#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stayed vs Moved Analysis — Matteo's label check for Exp2.

Checks if the model learned "employment" or just "stayed home vs moved away".

For each sample:
  morning_mode_geo = most common geo_id in context window (5am-8:30am)
  prediction_geo   = ground truth geo_id at prediction time
  new_label = 0 if same (stayed), 1 if different (moved)

Then compare PCA with new_label vs original employed/unemployed label.

Usage:
    python stayed_vs_moved_analysis.py --model 12b --config 1
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, OUTPUT_DIR, DATA_DIR,
                    MODEL_REGISTRY, get_model_dirs, config_label)


def load_hidden(model_key, config_id):
    dirs = get_model_dirs(model_key)
    path = os.path.join(dirs['hidden'], f"exp2_cfg{config_id}.npz")
    if not os.path.exists(path):
        return None, None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels'], d['meta']


def compute_stayed_moved_labels(meta, df):
    """For each sample, check if prediction geo == morning mode geo."""
    new_labels = []
    # Pre-group
    user_date_groups = {}
    for _, row in df.iterrows():
        key = (row['cuebiq_id'], str(row['date']))
        if key not in user_date_groups:
            user_date_groups[key] = []
        user_date_groups[key].append(row)

    for m in meta:
        uid, date, pt, gt_geo = m['user'], m['date'], m['pred_time'], m.get('gt_geo_id')

        # Get morning slots (5am-8:30am) for this user-date
        key = (uid, date)
        if key not in user_date_groups:
            new_labels.append(-1)  # unknown
            continue

        rows = user_date_groups[key]
        morning_geos = []
        for r in rows:
            minutes = int(r['hour']) * 60 + int(r['min5'])
            if 300 <= minutes <= 510:  # 5:00-8:30
                morning_geos.append(str(r['geo_id']))

        if not morning_geos or gt_geo is None:
            new_labels.append(-1)
            continue

        morning_mode = Counter(morning_geos).most_common(1)[0][0]
        new_labels.append(0 if morning_mode == str(gt_geo) else 1)  # 0=stayed, 1=moved

    return np.array(new_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='12b', choices=['1b', '4b', '12b'])
    parser.add_argument('--config', required=True, type=int)
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer (default: 75% depth)')
    args = parser.parse_args()

    mk = args.model
    cfg_id = args.config
    tag = MODEL_REGISTRY[mk]['tag']
    clabel = config_label('exp2', cfg_id)

    print(f"{'='*60}")
    print(f"Stayed vs Moved Analysis: exp2 cfg{cfg_id} ({clabel}) | {tag}")
    print(f"{'='*60}")

    X, y_emp, meta = load_hidden(mk, cfg_id)
    if X is None:
        print("Hidden states not found!")
        return

    n_layers = X.shape[1]
    layer = args.layer if args.layer is not None else int(n_layers * 0.75)

    # Load trajectory data for morning geo computation
    print("Loading trajectory data...")
    df = pd.read_csv(f"{DATA_DIR}/trajectories_processed.csv", low_memory=False)
    df['date'] = df['date'].astype(str)

    print("Computing stayed/moved labels...")
    y_sm = compute_stayed_moved_labels(meta, df)

    valid = y_sm >= 0
    X_valid = X[valid, layer, :]
    y_emp_valid = y_emp[valid]
    y_sm_valid = y_sm[valid]

    print(f"  Valid samples: {valid.sum()}/{len(valid)}")
    print(f"  Employed/Unemployed: {(y_emp_valid==1).sum()}/{(y_emp_valid==0).sum()}")
    print(f"  Moved/Stayed:        {(y_sm_valid==1).sum()}/{(y_sm_valid==0).sum()}")

    # Agreement
    agree = (y_emp_valid == y_sm_valid).mean()
    print(f"  Label agreement (emp==moved): {agree:.1%}")

    # PCA comparison
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X_valid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Exp2 cfg{cfg_id} ({clabel}) — {tag} — Layer {layer}\n"
                 f"Employed/Unemployed vs Stayed/Moved",
                 fontsize=14, fontweight='bold')

    # Left: original label
    for c, color, label in [(0, 'coral', 'Unemployed'), (1, 'steelblue', 'Employed')]:
        mask = y_emp_valid == c
        ax1.scatter(X2d[mask, 0], X2d[mask, 1], c=color, label=label, alpha=0.3, s=8)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('Original: Employed vs Unemployed')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Right: stayed/moved label
    for c, color, label in [(0, 'coral', 'Stayed (same geo)'), (1, 'steelblue', 'Moved (diff geo)')]:
        mask = y_sm_valid == c
        ax2.scatter(X2d[mask, 0], X2d[mask, 1], c=color, label=label, alpha=0.3, s=8)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title(f'Relabeled: Stayed vs Moved (agreement={agree:.0%})')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    dirs = get_model_dirs(mk)
    out = os.path.join(dirs['results'], f"stayed_vs_moved_cfg{cfg_id}_L{layer}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
