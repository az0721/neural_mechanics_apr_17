#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry analysis: PCA scatter, Direction Accuracy, Normalized Separation.
All configs on one figure per experiment.

Usage:
    python geometry_v7.py --exp exp2
    python geometry_v7.py --exp exp1a exp2
"""
import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXP_CONFIG, CONFIG_MATRIX, get_model_dirs, config_label

CONFIG_STYLES = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--', 7: '-.', 8: ':'}
CONFIG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
                 5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}
LAYERS_PCA = [0, 12, 24, 36, 48]


def direction_accuracy(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    d = c1 - c0
    projs = X @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    acc0 = (preds[y == 0] == 0).mean()
    acc1 = (preds[y == 1] == 1).mean()
    return (acc0 + acc1) / 2


def normalized_separation(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    return np.linalg.norm(c1 - c0) / np.sqrt(X.shape[1])


def make_pca_page(all_data, exp_name, exp_cfg, tag, dirs):
    """PCA scatter: rows=configs, cols=layers."""
    configs = [d for d in all_data if d is not None]
    n_cfg = len(configs)
    if n_cfg == 0:
        return
    n_layers = len(LAYERS_PCA)
    class_names = {0: 'Class 0', 1: 'Class 1'}
    if 'weekday' in exp_cfg['label_col'].lower():
        class_names = {0: 'Weekend', 1: 'Weekday'}
    elif 'employed' in exp_cfg['label_col'].lower():
        class_names = {0: 'Unemployed', 1: 'Employed'}

    fig, axes = plt.subplots(n_cfg, n_layers, figsize=(4 * n_layers, 3.5 * n_cfg))
    if n_cfg == 1:
        axes = axes.reshape(1, -1)

    for row, d in enumerate(configs):
        X, y, cfg_id, clabel = d['X'], d['y'], d['cfg_id'], d['clabel']
        for col, layer in enumerate(LAYERS_PCA):
            ax = axes[row, col]
            Xl = X[:, layer, :]
            pca = PCA(n_components=2)
            pc = pca.fit_transform(Xl)
            ev = pca.explained_variance_ratio_

            for cls in [0, 1]:
                mask = y == cls
                ax.scatter(pc[mask, 0], pc[mask, 1], s=8, alpha=0.4,
                           label=class_names[cls] if row == 0 and col == n_layers - 1 else None)
            ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=7)
            ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"cfg{cfg_id}\n({clabel})\n\nPC2 ({ev[1]:.1%})", fontsize=7)

    if configs:
        axes[0, -1].legend(fontsize=7, loc='upper right')
    fig.suptitle(f"{exp_cfg['name']} — {tag} (Iter 7, 15-min)\n"
                 f"PCA Scatter (rows=configs, cols=layers)", fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(dirs['results'], f"geometry_{exp_name}_pca.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  PCA saved: {out}")
    plt.close()


def make_da_sep_page(all_data, exp_name, exp_cfg, tag, dirs):
    """Direction Accuracy + Normalized Separation, all configs on one figure."""
    configs = [d for d in all_data if d is not None]
    if not configs:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    baseline = 1.0 / exp_cfg['n_classes']

    for d in configs:
        X, y = d['X'], d['y']
        cfg_id, clabel = d['cfg_id'], d['clabel']
        n_layers = X.shape[1]
        das, seps = [], []
        for layer in range(n_layers):
            das.append(direction_accuracy(X[:, layer, :], y))
            seps.append(normalized_separation(X[:, layer, :], y))

        das, seps = np.array(das), np.array(seps)
        best_l = das.argmax()
        ls = CONFIG_STYLES.get(cfg_id, '-')
        color = CONFIG_COLORS.get(cfg_id, '#333')

        ax1.plot(range(n_layers), das, ls, color=color, lw=2,
                 label=f"cfg{cfg_id} ({clabel}) {das.max():.1%}@L{best_l}")
        ax1.scatter(best_l, das.max(), s=40, color=color, zorder=5)
        ax2.plot(range(n_layers), seps, ls, color=color, lw=2,
                 label=f"cfg{cfg_id} ({clabel})")

    ax1.axhline(baseline, color='red', ls='--', alpha=0.4, label=f'Baseline {baseline:.0%}')
    ax1.set_ylabel('Balanced Accuracy')
    ax1.set_title('Direction Accuracy by Layer')
    ax1.legend(fontsize=6, loc='lower right')
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0, 1.0)

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('L2 / sqrt(hidden_dim)')
    ax2.set_title('Normalized Separation by Layer')
    ax2.legend(fontsize=6, loc='upper left')
    ax2.grid(True, alpha=0.15)

    fig.suptitle(f"{exp_cfg['name']} — {tag} (Iter 7, 15-min)\n"
                 f"Direction Accuracy & Normalized Separation (all configs)",
                 fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(dirs['results'], f"geometry_{exp_name}_da_sep.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  DA+Sep saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--model', default='12b')
    args = parser.parse_args()

    dirs = get_model_dirs(args.model)
    tag = f"gemma3_{args.model}b_it"

    for exp in args.exp:
        exp_cfg = EXP_CONFIG[exp]
        configs = CONFIG_MATRIX[exp]

        print(f"\n{'='*60}")
        print(f"Geometry: {exp} | {tag}")
        print(f"{'='*60}")

        all_data = []
        for cfg in configs:
            cfg_id = cfg['id']
            clabel = config_label(exp, cfg_id)
            path = os.path.join(dirs['hidden'], f"{exp}_cfg{cfg_id}.npz")
            if not os.path.exists(path):
                print(f"  cfg{cfg_id}: not found")
                all_data.append(None)
                continue

            data = np.load(path, allow_pickle=True)
            X, y = data['hidden_states'], data['labels']
            print(f"  cfg{cfg_id} ({clabel}): {X.shape[0]} samples")
            all_data.append({'X': X, 'y': y, 'cfg_id': cfg_id, 'clabel': clabel})

        make_pca_page(all_data, exp, exp_cfg, tag, dirs)
        make_da_sep_page(all_data, exp, exp_cfg, tag, dirs)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
