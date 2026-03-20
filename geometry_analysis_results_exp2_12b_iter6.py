#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry analysis PDF — Iter 6, Exp2, 12b only.
Page 1-2: PCA scatter (8 cfgs split into 2 pages × 4 rows × 5 layers)
Page 3: Direction accuracy + separation (8 cfgs overlaid)

Usage:
    python geometry_analysis_results_exp2_12b_iter6.py
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, OUTPUT_DIR, MODEL_REGISTRY, get_model_dirs, config_label)

CLASS_LABELS = {0: 'Unemployed', 1: 'Employed'}
CFG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
              5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}
CFG_STYLES = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--', 7: '-.', 8: ':'}


def load_hidden(config_id):
    dirs = get_model_dirs('12b')
    path = os.path.join(dirs['hidden'], f"exp2_cfg{config_id}.npz")
    if not os.path.exists(path):
        return None, None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels'], d['meta']


def get_pca_layers(n_layers):
    return np.linspace(0, n_layers - 1, 5, dtype=int).tolist()


def direction_accuracy(X_layer, y):
    c0, c1 = X_layer[y == 0], X_layer[y == 1]
    d = c1.mean(axis=0) - c0.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-10:
        return 0.5
    scores = X_layer @ (d / norm)
    preds = (scores > np.median(scores)).astype(int)
    return balanced_accuracy_score(y, preds)


def separation_l2(X_layer, y, hidden_dim):
    c0 = X_layer[y == 0].mean(axis=0)
    c1 = X_layer[y == 1].mean(axis=0)
    return np.linalg.norm(c1 - c0) / np.sqrt(hidden_dim)


def make_pca_page(pdf, cfgs, page_label):
    tag = MODEL_REGISTRY['12b']['tag']
    fig, axes = plt.subplots(4, 5, figsize=(28, 20))
    fig.suptitle(f"Exp2: Employed vs Unemployed — {tag} (Iter 6, bfloat16)\n"
                 f"PCA Scatter — {page_label}",
                 fontsize=20, fontweight='bold', y=0.99)

    for row, cfg_id in enumerate(cfgs):
        clabel = config_label('exp2', cfg_id)
        X, y, _ = load_hidden(cfg_id)

        if X is None:
            for col in range(5):
                axes[row, col].text(0.5, 0.5, 'Not found', ha='center', va='center')
                if col == 0:
                    axes[row, col].set_ylabel(f'cfg{cfg_id}\n({clabel})', fontsize=11, fontweight='bold')
            continue

        pca_layers = get_pca_layers(X.shape[1])
        for col, layer in enumerate(pca_layers):
            ax = axes[row, col]
            pca = PCA(n_components=2)
            X2d = pca.fit_transform(X[:, layer, :])
            pc1, pc2 = pca.explained_variance_ratio_[:2]

            for c, color in [(0, '#e74c3c'), (1, '#3498db')]:
                mask = y == c
                ax.scatter(X2d[mask, 0], X2d[mask, 1], c=color,
                           label=CLASS_LABELS[c], alpha=0.35, s=8, edgecolors='none')

            ax.set_xlabel(f'PC1 ({pc1:.1%})', fontsize=9)
            ax.set_ylabel(f'PC2 ({pc2:.1%})', fontsize=9)
            ax.set_title(f'Layer {layer}', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)

            if col == 0:
                ax.annotate(f'cfg{cfg_id}\n({clabel})', xy=(-0.35, 0.5),
                            xycoords='axes fraction', fontsize=9, fontweight='bold',
                            ha='center', va='center', rotation=90)
            if row == 0 and col == 4:
                ax.legend(fontsize=8, markerscale=2, loc='upper right')

    plt.tight_layout(rect=[0.04, 0, 1, 0.96], h_pad=3, w_pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"  PCA page: {page_label}")


def make_da_sep_page(pdf):
    tag = MODEL_REGISTRY['12b']['tag']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Exp2: Employed vs Unemployed — {tag} (Iter 6, bfloat16)\n"
                 f"Direction Accuracy & Normalized Separation (all 8 configs)",
                 fontsize=16, fontweight='bold', y=0.98)

    for cfg_id in range(1, 9):
        clabel = config_label('exp2', cfg_id)
        X, y, _ = load_hidden(cfg_id)
        if X is None:
            continue

        n_layers = X.shape[1]
        hidden_dim = X.shape[2]
        color = CFG_COLORS[cfg_id]
        ls = CFG_STYLES[cfg_id]
        lw = 2.5 if cfg_id <= 4 else 1.8

        da = [direction_accuracy(X[:, l, :], y) for l in range(n_layers)]
        sep = [separation_l2(X[:, l, :], y, hidden_dim) for l in range(n_layers)]

        best_da = max(da)
        best_l = da.index(best_da)
        ax1.plot(da, ls, color=color, lw=lw,
                 label=f"cfg{cfg_id} ({clabel}) — {best_da:.1%}@L{best_l}")
        ax1.plot(best_l, best_da, 'o', color=color, ms=6, zorder=5)
        ax2.plot(sep, ls, color=color, lw=lw, label=f"cfg{cfg_id} ({clabel})")

    ax1.axhline(0.5, color='red', ls='--', alpha=0.3, label='Baseline 50%')
    ax1.set_ylabel('Balanced Accuracy', fontsize=13)
    ax1.set_title('Direction Accuracy by Layer', fontsize=13)
    ax1.legend(fontsize=7, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0, 1.0)

    ax2.set_xlabel('Layer', fontsize=13)
    ax2.set_ylabel('L2 / sqrt(hidden_dim)', fontsize=13)
    ax2.set_title('Normalized Separation by Layer', fontsize=13)
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print("  DA+Sep page done")


def main():
    out_path = os.path.join(OUTPUT_DIR, 'geometry_exp2_iter6.pdf')
    print(f"Generating: {out_path}")
    with PdfPages(out_path) as pdf:
        make_pca_page(pdf, [1, 2, 3, 4], "Configs 1-4 (geo=orig)")
        make_pca_page(pdf, [5, 6, 7, 8], "Configs 5-8 (geo=hash)")
        make_da_sep_page(pdf)
    print(f"Saved: {out_path} (3 pages)")


if __name__ == "__main__":
    main()