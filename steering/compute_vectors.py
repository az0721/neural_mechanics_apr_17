#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute steering vectors — Method A (direct difference) + Method B (baseline-relative).

Method A (old, from Iter 7):
    V = mean(H_emp) - mean(H_unemp)
    V = V / ||V||

Method B (new, Matteo Mar 25):
    H_baseline = mean(H_all_240_users)
    V_emp   = (mean(H_emp) - H_baseline) / ||...||
    V_unemp = (mean(H_unemp) - H_baseline) / ||...||

Both methods produce normalized unit vectors per layer.
Method A produces 1 vector; Method B produces 2 vectors.

Output:
    outputs_v8/steering/vectors/
        {exp}_cfg{id}_vectors_A.npz   → vectors (n_layers, hidden_dim), norms
        {exp}_cfg{id}_vectors_B.npz   → v_emp, v_unemp, norms_emp, norms_unemp

Usage:
    python steering/compute_vectors_v2.py                    # all cfgs
    python steering/compute_vectors_v2.py --cfgs 5 6         # specific
    python steering/compute_vectors_v2.py --method both      # default
    python steering/compute_vectors_v2.py --method A         # only Method A
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_iter_output_dir, get_model_dirs, CONFIG_MATRIX, config_label, OUTPUT_DIR, get_iter_model_dirs, MODEL_REGISTRY

VECTORS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'vectors')
os.makedirs(VECTORS_DIR, exist_ok=True)


def compute_method_a(X, y):
    """V = mean(emp) - mean(unemp), normalized."""
    n_layers = X.shape[1]
    vectors = np.zeros((n_layers, X.shape[2]), dtype=np.float32)
    norms = np.zeros(n_layers, dtype=np.float32)

    for layer in range(n_layers):
        v = X[y == 1, layer, :].mean(0) - X[y == 0, layer, :].mean(0)
        norm = np.linalg.norm(v)
        norms[layer] = norm
        vectors[layer] = v / max(norm, 1e-10)

    return vectors, norms


def compute_method_b(X, y):
    """V_emp = mean(emp) - baseline, V_unemp = mean(unemp) - baseline, both normalized."""
    n_layers = X.shape[1]
    v_emp = np.zeros((n_layers, X.shape[2]), dtype=np.float32)
    v_unemp = np.zeros((n_layers, X.shape[2]), dtype=np.float32)
    norms_emp = np.zeros(n_layers, dtype=np.float32)
    norms_unemp = np.zeros(n_layers, dtype=np.float32)

    for layer in range(n_layers):
        h_all = X[:, layer, :].mean(0)             # baseline = mean of ALL users
        h_emp = X[y == 1, layer, :].mean(0)
        h_unemp = X[y == 0, layer, :].mean(0)

        ve = h_emp - h_all
        vu = h_unemp - h_all

        ne = np.linalg.norm(ve)
        nu = np.linalg.norm(vu)
        norms_emp[layer] = ne
        norms_unemp[layer] = nu

        v_emp[layer] = ve / max(ne, 1e-10)
        v_unemp[layer] = vu / max(nu, 1e-10)

    return v_emp, v_unemp, norms_emp, norms_unemp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='exp2')
    parser.add_argument('--cfgs', nargs='+', type=int, default=[5, 6])
    parser.add_argument('--model', default='12b')
    parser.add_argument('--method', default='both', choices=['A', 'B', 'both'])
    args = parser.parse_args()

    dirs = get_model_dirs(args.model)

    for cfg_id in args.cfgs:
        clabel = config_label(args.exp, cfg_id)
        path = os.path.join(dirs['hidden'], f"{args.exp}_cfg{cfg_id}.npz")
        if not os.path.exists(path):
            print(f"  cfg{cfg_id}: {path} not found, skipping")
            continue

        data = np.load(path, allow_pickle=True)
        X, y = data['hidden_states'], data['labels']
        n_emp = (y == 1).sum()
        n_unemp = (y == 0).sum()

        print(f"\n{'='*60}")
        print(f"cfg{cfg_id} ({clabel}): {X.shape[0]} samples, {X.shape[1]} layers")
        print(f"  Employed: {n_emp}, Unemployed: {n_unemp}")

        # ── Method A: direct difference ──
        if args.method in ('A', 'both'):
            vectors_a, norms_a = compute_method_a(X, y)
            out_a = os.path.join(VECTORS_DIR, f"{args.exp}_cfg{cfg_id}_vectors_A.npz")
            np.savez(out_a, vectors=vectors_a, norms=norms_a)
            print(f"  Method A saved: {out_a}")
            print(f"    Norms (raw before normalization): "
                  f"L0={norms_a[0]:.1f}, L25={norms_a[min(25,len(norms_a)-1)]:.1f}, "
                  f"L33={norms_a[min(33,len(norms_a)-1)]:.1f}, "
                  f"L48={norms_a[-1]:.1f}")
            print(f"    Vectors normalized: "
                  f"||V[25]||={np.linalg.norm(vectors_a[min(25,len(norms_a)-1)]):.4f}")

        # ── Method B: baseline-relative ──
        if args.method in ('B', 'both'):
            v_emp, v_unemp, ne, nu = compute_method_b(X, y)
            out_b = os.path.join(VECTORS_DIR, f"{args.exp}_cfg{cfg_id}_vectors_B.npz")
            np.savez(out_b, v_emp=v_emp, v_unemp=v_unemp,
                     norms_emp=ne, norms_unemp=nu)
            print(f"  Method B saved: {out_b}")
            print(f"    Emp norms: L25={ne[min(25,len(ne)-1)]:.1f}, "
                  f"L33={ne[min(33,len(ne)-1)]:.1f}")
            print(f"    Unemp norms: L25={nu[min(25,len(nu)-1)]:.1f}, "
                  f"L33={nu[min(33,len(nu)-1)]:.1f}")
            print(f"    Vectors normalized: "
                  f"||V_emp[25]||={np.linalg.norm(v_emp[min(25,len(ne)-1)]):.4f}")

            # ── Compare A and B ──
            if args.method == 'both':
                print(f"\n  Method A vs B comparison:")
                for l in [1, 10, 20, 25, 33, 40]:
                    if l >= X.shape[1]: continue
                    cos_a_emp = np.dot(vectors_a[l], v_emp[l])
                    cos_a_unemp = np.dot(vectors_a[l], v_unemp[l])
                    cos_emp_unemp = np.dot(v_emp[l], v_unemp[l])
                    print(f"    L{l}: cos(A, B_emp)={cos_a_emp:+.4f}  "
                          f"cos(A, B_unemp)={cos_a_unemp:+.4f}  "
                          f"cos(B_emp, B_unemp)={cos_emp_unemp:+.4f}")

    print(f"\n{'='*60}")
    print(f"Done! Vectors saved to: {VECTORS_DIR}")


if __name__ == "__main__":
    main()