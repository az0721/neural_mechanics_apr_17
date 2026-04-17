#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute RAW (unnormalized) steering vectors for NL steering experiments.

Two methods:
  Old:  V = mean(H_emp) - mean(H_unemp)           → raw, use with coefficients
  New:  V_emp = mean(H_emp) - mean(H_all)          → raw, apply directly (no coeff)
        V_unemp = mean(H_unemp) - mean(H_all)      → raw, apply directly (no coeff)

Vectors are NOT normalized — this is intentional.

Usage:
    python steering/compute_vectors_nl.py --model gemma4_31b --iter v7 --exp exp2 --cfgs 2 4 6 8
    python steering/compute_vectors_nl.py --model gemma4_31b --all
    python steering/compute_vectors_nl.py --model 12b --all
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (config_label, MODEL_REGISTRY, COT_ONLY_IDS,
                    get_iter_model_dirs, get_iter_output_dir)


def compute_vectors(X, y):
    """Compute both old and new method vectors (RAW, unnormalized)."""
    n_layers, hidden_dim = X.shape[1], X.shape[2]

    v_old = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    v_emp_new = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    v_unemp_new = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    norms_old = np.zeros(n_layers, dtype=np.float32)
    norms_emp = np.zeros(n_layers, dtype=np.float32)
    norms_unemp = np.zeros(n_layers, dtype=np.float32)

    for l in range(n_layers):
        h_emp = X[y == 1, l, :].mean(0)
        h_unemp = X[y == 0, l, :].mean(0)
        h_all = X[:, l, :].mean(0)

        v = h_emp - h_unemp
        v_old[l] = v
        norms_old[l] = np.linalg.norm(v)

        ve = h_emp - h_all
        vu = h_unemp - h_all
        v_emp_new[l] = ve
        v_unemp_new[l] = vu
        norms_emp[l] = np.linalg.norm(ve)
        norms_unemp[l] = np.linalg.norm(vu)

    return {
        'v_old': v_old, 'norms_old': norms_old,
        'v_emp_new': v_emp_new, 'v_unemp_new': v_unemp_new,
        'norms_emp': norms_emp, 'norms_unemp': norms_unemp,
    }


def process(iter_name, exp, cfg_id, model_key='12b'):
    dirs = get_iter_model_dirs(model_key, iter_name)
    path = os.path.join(dirs['hidden'], f"{exp}_cfg{cfg_id}.npz")
    if not os.path.exists(path):
        print(f"  {iter_name}/{exp}_cfg{cfg_id}: NOT FOUND ({path})")
        return False

    # Output dir: model-aware
    out_base = get_iter_output_dir(model_key, iter_name)
    out_dir = os.path.join(out_base, 'steering', 'vectors_nl')
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(path, allow_pickle=True)
    X, y = data['hidden_states'], data['labels']
    clabel = config_label(exp, cfg_id)

    print(f"  {iter_name}/{exp}_cfg{cfg_id} ({clabel}): "
          f"N={X.shape[0]}, layers={X.shape[1]}, "
          f"emp={int((y==1).sum())}, unemp={int((y==0).sum())}")

    vecs = compute_vectors(X, y)

    out_path = os.path.join(out_dir, f"{exp}_cfg{cfg_id}_nl_vectors.npz")
    np.savez(out_path, **vecs)

    # Print sample norms at key layers
    for l in [0, 12, 25, 33, 45, 55]:
        if l < X.shape[1]:
            print(f"    L{l}: ||V_old||={vecs['norms_old'][l]:.1f}  "
                  f"||V_emp||={vecs['norms_emp'][l]:.1f}  "
                  f"||V_unemp||={vecs['norms_unemp'][l]:.1f}")

    print(f"    Saved: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=None, help='v7 or v8')
    parser.add_argument('--exp', default=None)
    parser.add_argument('--cfgs', nargs='+', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    if args.all:
        # Use COT_ONLY_IDS for the model — covers both 12b and gemma4
        combos = []
        for it in ['v7', 'v8']:
            for exp in ['exp2', 'exp1a']:
                cfgs = COT_ONLY_IDS[exp]
                combos.append((it, exp, cfgs))
    else:
        if not args.iter or not args.exp or not args.cfgs:
            print("Usage: --iter v7 --exp exp2 --cfgs 2 4 6 8  OR  --all")
            return
        combos = [(args.iter, args.exp, args.cfgs)]

    print(f"{'='*60}")
    print(f"Computing NL steering vectors (RAW, unnormalized)")
    print(f"  Model: {args.model} ({MODEL_REGISTRY[args.model]['tag']})")
    print(f"{'='*60}")

    for iter_name, exp, cfgs in combos:
        for cfg_id in cfgs:
            process(iter_name, exp, cfg_id, args.model)

    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == "__main__":
    main()