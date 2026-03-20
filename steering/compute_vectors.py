#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute normalized steering vectors from merged hidden states.
One vector per (config, layer). Saved as .npz.

Usage:
    python steering/compute_vectors.py
    python steering/compute_vectors.py --cfgs 5 6 --exp exp2
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_model_dirs, CONFIG_MATRIX, config_label, OUTPUT_DIR

VECTORS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'vectors')
os.makedirs(VECTORS_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='exp2')
    parser.add_argument('--cfgs', nargs='+', type=int, default=[5, 6])
    parser.add_argument('--model', default='12b')
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
        n_layers = X.shape[1]

        print(f"\n{'='*60}")
        print(f"cfg{cfg_id} ({clabel}): {X.shape[0]} samples, {n_layers} layers")
        print(f"  Employed: {(y==1).sum()}, Unemployed: {(y==0).sum()}")

        vectors = np.zeros((n_layers, X.shape[2]), dtype=np.float32)
        norms = np.zeros(n_layers, dtype=np.float32)

        for layer in range(n_layers):
            v = X[y == 1, layer, :].mean(0) - X[y == 0, layer, :].mean(0)
            norm = np.linalg.norm(v)
            norms[layer] = norm
            vectors[layer] = v  # normalized to unit length

        out_path = os.path.join(VECTORS_DIR, f"{args.exp}_cfg{cfg_id}_vectors.npz")
        np.savez(out_path, vectors=vectors, norms=norms)
        print(f"  Saved: {out_path}")
        print(f"  Norm range: L0={norms[0]:.1f}, L23={norms[23]:.1f}, "
              f"L31={norms[31]:.1f}, L48={norms[48]:.1f}")


if __name__ == "__main__":
    main()
