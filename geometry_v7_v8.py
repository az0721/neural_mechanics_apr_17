#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Holistic Geometry Analysis — V7 vs V8 Cross-Iteration Comparison.

10 analysis types per experiment:
  1.  PCA scatter (V7 vs V8, key layers)
  2.  t-SNE visualization (key layers)
  3.  Direction Accuracy curves (all configs, V7 vs V8)
  4.  Normalized Separation curves
  5.  Fisher Discriminant Ratio curves
  6.  Projection Distribution Histograms (concept direction)
  7.  Cross-layer direction cosine (within-iter stability)
  8.  Cross-config direction cosine (concept direction consistency)
  9.  Generalization test (train/test split DA)
  10. Explained variance ratio (concept direction vs total)
  +   Summary table

Output: cross_iteration_comparison/geometry_holistic_{exp}.pdf

Usage:
    python geometry_v7_v8.py                   # both exps
    python geometry_v7_v8.py --exp exp2        # one exp
"""
import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXP_CONFIG, CONFIG_MATRIX, config_label, MODEL_REGISTRY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'cross_iteration_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

ITERS = {'v7': os.path.join(BASE_DIR, 'outputs_v7', 'hidden_states',
                             MODEL_REGISTRY['12b']['tag']),
         'v8': os.path.join(BASE_DIR, 'outputs_v8', 'hidden_states',
                             MODEL_REGISTRY['12b']['tag'])}

LAYERS_VIS = [0, 12, 24, 33, 48]    # for PCA / t-SNE
CFG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
              5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}
ITER_STYLE = {'v7': '-', 'v8': '--'}
ITER_COLOR = {'v7': '#2196F3', 'v8': '#FF5722'}
CLASS_NAMES = {
    'exp1a': {0: 'Weekend', 1: 'Weekday'},
    'exp2':  {0: 'Unemployed', 1: 'Employed'},
}

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_hidden(base_dir, exp, cfg_id):
    path = os.path.join(base_dir, f"{exp}_cfg{cfg_id}.npz")
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels']


def load_all(exp):
    """Load all configs for both iters. Returns {(iter, cfg_id): (X, y)}."""
    data = {}
    for it, base in ITERS.items():
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            X, y = load_hidden(base, exp, cid)
            if X is not None:
                data[(it, cid)] = (X, y)
                print(f"  {it}/cfg{cid}: {X.shape}")
    return data


# ══════════════════════════════════════════════════════════════════════
# Metric Functions
# ══════════════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def direction_accuracy(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    d = c1 - c0
    projs = X @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    return ((preds[y == 0] == 0).mean() + (preds[y == 1] == 1).mean()) / 2


def normalized_separation(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    return np.linalg.norm(c1 - c0) / np.sqrt(X.shape[1])


def fisher_ratio(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    between = np.linalg.norm(c1 - c0) ** 2
    var0 = np.var(X[y == 0], axis=0).sum()
    var1 = np.var(X[y == 1], axis=0).sum()
    within = var0 + var1
    return between / max(within, 1e-10)


def concept_direction(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def explained_variance_ratio(X, y):
    d = concept_direction(X, y)
    d = d / max(np.linalg.norm(d), 1e-10)
    projs = X @ d
    var_concept = np.var(projs)
    var_total = np.var(X, axis=0).sum()
    return var_concept / max(var_total, 1e-10)


def direction_accuracy_split(X, y, train_frac=0.7):
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * train_frac)
    tr_idx, te_idx = idx[:split], idx[split:]
    d = X[tr_idx][y[tr_idx] == 1].mean(0) - X[tr_idx][y[tr_idx] == 0].mean(0)
    projs = X[te_idx] @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    y_te = y[te_idx]
    return ((preds[y_te == 0] == 0).mean() + (preds[y_te == 1] == 1).mean()) / 2


# ══════════════════════════════════════════════════════════════════════
# Helper: pick best config per iter for detailed pages
# ══════════════════════════════════════════════════════════════════════

def pick_best_cfg(data, exp):
    best = {}
    for it in ['v7', 'v8']:
        best_da, best_cid = 0, None
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            da = max(direction_accuracy(X[:, l, :], y) for l in range(X.shape[1]))
            if da > best_da:
                best_da, best_cid = da, cid
        best[it] = best_cid
    return best


# ══════════════════════════════════════════════════════════════════════
# Page 1: PCA Scatter (V7 vs V8, best config, key layers)
# ══════════════════════════════════════════════════════════════════════

def pg_pca(pdf, data, exp, best_cfgs):
    cn = CLASS_NAMES[exp]
    layers = [l for l in LAYERS_VIS]

    fig, axes = plt.subplots(2, len(layers), figsize=(4 * len(layers), 8),
                              squeeze=False)
    for row, it in enumerate(['v7', 'v8']):
        cid = best_cfgs[it]
        if cid is None or (it, cid) not in data:
            continue
        X, y = data[(it, cid)]
        n_layers = X.shape[1]
        for col, l in enumerate(layers):
            ax = axes[row, col]
            if l >= n_layers:
                ax.set_visible(False); continue
            Xl = X[:, l, :]
            pca = PCA(n_components=2)
            proj = pca.fit_transform(Xl)
            for c in [0, 1]:
                mask = y == c
                ax.scatter(proj[mask, 0], proj[mask, 1], s=3, alpha=0.4,
                           label=cn[c] if col == 0 else None)
            ax.set_title(f'L{l}', fontsize=9, fontweight='bold')
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=7)
            if col == 0:
                ax.set_ylabel(f'{it.upper()} cfg{cid}\nPC2 ({ev[1]:.1%})', fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0 and row == 0:
                ax.legend(fontsize=6, markerscale=3)

    fig.suptitle(f'{EXP_CONFIG[exp]["name"]} — PCA Scatter (V7 vs V8, best config)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 2: t-SNE (V7 vs V8, best config, 3 key layers)
# ══════════════════════════════════════════════════════════════════════

def pg_tsne(pdf, data, exp, best_cfgs):
    cn = CLASS_NAMES[exp]
    tsne_layers = [12, 33, 48]
    n_sub = 300  # subsample for speed

    fig, axes = plt.subplots(2, len(tsne_layers),
                              figsize=(5 * len(tsne_layers), 9), squeeze=False)
    for row, it in enumerate(['v7', 'v8']):
        cid = best_cfgs[it]
        if cid is None or (it, cid) not in data:
            continue
        X, y = data[(it, cid)]
        n = X.shape[0]
        idx = np.random.choice(n, min(n_sub, n), replace=False)

        for col, l in enumerate(tsne_layers):
            ax = axes[row, col]
            if l >= X.shape[1]:
                ax.set_visible(False); continue
            Xl = X[idx, l, :]
            yl = y[idx]
            tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                        n_iter=1000)
            proj = tsne.fit_transform(Xl)
            for c in [0, 1]:
                mask = yl == c
                ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.5,
                           label=cn[c] if col == 0 else None)
            ax.set_title(f'L{l}', fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{it.upper()} cfg{cid}', fontsize=9)
            ax.tick_params(labelsize=6)
            if col == 0 and row == 0:
                ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(f'{EXP_CONFIG[exp]["name"]} — t-SNE (V7 vs V8, n={n_sub})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 3: Direction Accuracy Curves (all configs, V7 vs V8)
# ══════════════════════════════════════════════════════════════════════

def pg_da(pdf, data, exp):
    fig, ax = plt.subplots(figsize=(14, 7))
    for it in ['v7', 'v8']:
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            das = [direction_accuracy(X[:, l, :], y) for l in range(X.shape[1])]
            ax.plot(das, ls=ITER_STYLE[it], color=CFG_COLORS[cid], lw=1.5,
                    alpha=0.9, label=f'{it}/cfg{cid}')
    ax.set(xlabel='Layer', ylabel='Direction Accuracy', ylim=(0.45, 1.0))
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Direction Accuracy\n'
                 f'(solid=V7, dashed=V8)', fontweight='bold')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 4: Normalized Separation Curves
# ══════════════════════════════════════════════════════════════════════

def pg_sep(pdf, data, exp):
    fig, ax = plt.subplots(figsize=(14, 7))
    for it in ['v7', 'v8']:
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            seps = [normalized_separation(X[:, l, :], y) for l in range(X.shape[1])]
            ax.plot(seps, ls=ITER_STYLE[it], color=CFG_COLORS[cid], lw=1.5,
                    alpha=0.9, label=f'{it}/cfg{cid}')
    ax.set(xlabel='Layer', ylabel='Normalized Separation')
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Normalized Separation\n'
                 f'(solid=V7, dashed=V8)', fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 5: Fisher Discriminant Ratio
# ══════════════════════════════════════════════════════════════════════

def pg_fisher(pdf, data, exp):
    fig, ax = plt.subplots(figsize=(14, 7))
    for it in ['v7', 'v8']:
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            frs = [fisher_ratio(X[:, l, :], y) for l in range(X.shape[1])]
            ax.plot(frs, ls=ITER_STYLE[it], color=CFG_COLORS[cid], lw=1.5,
                    alpha=0.9, label=f'{it}/cfg{cid}')
    ax.set(xlabel='Layer', ylabel='Fisher Ratio (between / within)')
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Fisher Discriminant Ratio\n'
                 f'(solid=V7, dashed=V8, higher=better separation)',
                 fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 6: Projection Distribution Histograms
# ══════════════════════════════════════════════════════════════════════

def pg_projection_hist(pdf, data, exp, best_cfgs):
    cn = CLASS_NAMES[exp]
    hist_layers = [12, 25, 33, 45]

    fig, axes = plt.subplots(2, len(hist_layers),
                              figsize=(4.5 * len(hist_layers), 8), squeeze=False)
    for row, it in enumerate(['v7', 'v8']):
        cid = best_cfgs[it]
        if cid is None or (it, cid) not in data:
            continue
        X, y = data[(it, cid)]

        for col, l in enumerate(hist_layers):
            ax = axes[row, col]
            if l >= X.shape[1]:
                ax.set_visible(False); continue
            Xl = X[:, l, :]
            d = concept_direction(Xl, y)
            d_norm = d / max(np.linalg.norm(d), 1e-10)
            projs = Xl @ d_norm

            for c, color in [(0, '#FF5722'), (1, '#2196F3')]:
                ax.hist(projs[y == c], bins=40, alpha=0.5, color=color,
                        density=True, label=cn[c] if col == 0 else None)

            da = direction_accuracy(Xl, y)
            ax.set_title(f'L{l} (DA={da:.1%})', fontsize=9, fontweight='bold')
            ax.set_xlabel('Projection onto concept direction', fontsize=7)
            if col == 0:
                ax.set_ylabel(f'{it.upper()} cfg{cid}\nDensity', fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0 and row == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f'{EXP_CONFIG[exp]["name"]} — Projection Distribution\n'
                 f'(each sample projected onto concept direction vector)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 7: Cross-Layer Direction Cosine (within-iter stability)
# ══════════════════════════════════════════════════════════════════════

def pg_cross_layer_cosine(pdf, data, exp, best_cfgs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)

    for col, it in enumerate(['v7', 'v8']):
        ax = axes[0, col]
        cid = best_cfgs[it]
        if cid is None or (it, cid) not in data:
            ax.set_visible(False); continue
        X, y = data[(it, cid)]
        n_layers = X.shape[1]

        mat = np.zeros((n_layers, n_layers))
        dirs = [concept_direction(X[:, l, :], y) for l in range(n_layers)]
        for i in range(n_layers):
            for j in range(n_layers):
                mat[i, j] = cosine_sim(dirs[i], dirs[j])

        im = ax.imshow(mat, aspect='equal', origin='lower', cmap='RdYlGn',
                        vmin=-0.5, vmax=1.0)
        ax.set(xlabel='Layer', ylabel='Layer')
        ax.set_title(f'{it.upper()} cfg{cid} — Direction Cosine (layer×layer)',
                     fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(f'{EXP_CONFIG[exp]["name"]} — Cross-Layer Direction Stability\n'
                 f'(bright=same direction, dark=different direction)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 8: Cross-Config Direction Cosine (at key layers)
# ══════════════════════════════════════════════════════════════════════

def pg_cross_config_cosine(pdf, data, exp):
    key_layers = [12, 25, 33, 45]
    cfgs = [c['id'] for c in CONFIG_MATRIX[exp]]

    for it in ['v7', 'v8']:
        avail = [cid for cid in cfgs if (it, cid) in data]
        if len(avail) < 2:
            continue
        n_avail = len(avail)
        fig, axes = plt.subplots(1, len(key_layers),
                                  figsize=(5 * len(key_layers), 5), squeeze=False)
        for col, l in enumerate(key_layers):
            ax = axes[0, col]
            mat = np.zeros((n_avail, n_avail))
            dirs = {}
            for cid in avail:
                X, y = data[(it, cid)]
                if l < X.shape[1]:
                    dirs[cid] = concept_direction(X[:, l, :], y)

            for i, ci in enumerate(avail):
                for j, cj in enumerate(avail):
                    if ci in dirs and cj in dirs:
                        mat[i, j] = cosine_sim(dirs[ci], dirs[cj])

            im = ax.imshow(mat, aspect='equal', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(range(n_avail))
            ax.set_xticklabels([f'c{c}' for c in avail], fontsize=7)
            ax.set_yticks(range(n_avail))
            ax.set_yticklabels([f'c{c}' for c in avail], fontsize=7)
            ax.set_title(f'L{l}', fontsize=10, fontweight='bold')
            # Annotate values
            for i in range(n_avail):
                for j in range(n_avail):
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                            fontsize=5, color='black' if mat[i,j] > 0.5 else 'white')
            plt.colorbar(im, ax=ax, shrink=0.7)

        fig.suptitle(f'{EXP_CONFIG[exp]["name"]} — Cross-Config Direction Cosine '
                     f'({it.upper()})\n'
                     f'(do different configs find the same concept direction?)',
                     fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 9: Generalization Test (train/test split DA)
# ══════════════════════════════════════════════════════════════════════

def pg_generalization(pdf, data, exp):
    n_trials = 10  # average over random splits

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)
    for col, it in enumerate(['v7', 'v8']):
        ax = axes[0, col]
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            n_layers = X.shape[1]

            da_full = [direction_accuracy(X[:, l, :], y) for l in range(n_layers)]
            da_test = []
            for l in range(n_layers):
                scores = [direction_accuracy_split(X[:, l, :], y)
                          for _ in range(n_trials)]
                da_test.append(np.mean(scores))

            ax.plot(da_full, ls='-', color=CFG_COLORS[cid], lw=1.5, alpha=0.7,
                    label=f'c{cid} full')
            ax.plot(da_test, ls='--', color=CFG_COLORS[cid], lw=1.5, alpha=0.7,
                    label=f'c{cid} test')

        ax.set(xlabel='Layer', ylabel='Direction Accuracy', ylim=(0.45, 1.0))
        ax.set_title(f'{it.upper()} — Full (solid) vs Held-out 30% (dashed)',
                     fontsize=10, fontweight='bold')
        ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
        ax.legend(fontsize=5, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.1)

    fig.suptitle(f'{EXP_CONFIG[exp]["name"]} — Generalization Test\n'
                 f'(concept direction trained on 70%, tested on 30%, '
                 f'avg {n_trials} splits)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 10: Explained Variance Ratio
# ══════════════════════════════════════════════════════════════════════

def pg_explained_var(pdf, data, exp):
    fig, ax = plt.subplots(figsize=(14, 7))
    for it in ['v7', 'v8']:
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            evrs = [explained_variance_ratio(X[:, l, :], y)
                    for l in range(X.shape[1])]
            ax.plot(evrs, ls=ITER_STYLE[it], color=CFG_COLORS[cid], lw=1.5,
                    alpha=0.9, label=f'{it}/cfg{cid}')

    ax.set(xlabel='Layer', ylabel='Variance Ratio')
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Explained Variance Ratio\n'
                 f'(fraction of total variance along concept direction; '
                 f'solid=V7, dashed=V8)',
                 fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════════════

def pg_summary(pdf, data, exp):
    fig, ax = plt.subplots(figsize=(20, 14)); ax.axis('off')
    lines = [f'{EXP_CONFIG[exp]["name"]} — Geometry Summary\n',
             '=' * 115 + '\n\n']

    hdr = (f'{"Iter":>4s} {"Cfg":>4s} {"Label":>20s} '
           f'{"DA_max":>7s} {"@L":>3s} {"Sep_max":>8s} {"@L":>3s} '
           f'{"Fisher":>8s} {"@L":>3s} {"EVR":>8s} {"@L":>3s} '
           f'{"DA_gen":>7s} {"@L":>3s}\n')
    lines.append(hdr)
    lines.append('-' * 115 + '\n')

    for it in ['v7', 'v8']:
        for cfg in CONFIG_MATRIX[exp]:
            cid = cfg['id']
            if (it, cid) not in data:
                continue
            X, y = data[(it, cid)]
            n_layers = X.shape[1]
            cl = config_label(exp, cid)

            das = [direction_accuracy(X[:, l, :], y) for l in range(n_layers)]
            seps = [normalized_separation(X[:, l, :], y) for l in range(n_layers)]
            frs = [fisher_ratio(X[:, l, :], y) for l in range(n_layers)]
            evrs = [explained_variance_ratio(X[:, l, :], y) for l in range(n_layers)]
            da_gens = [np.mean([direction_accuracy_split(X[:, l, :], y)
                                for _ in range(5)]) for l in range(n_layers)]

            lines.append(
                f'{it:>4s} {cid:>4d} {cl:>20s} '
                f'{max(das):>7.1%} {np.argmax(das):>3d} '
                f'{max(seps):>8.5f} {np.argmax(seps):>3d} '
                f'{max(frs):>8.5f} {np.argmax(frs):>3d} '
                f'{max(evrs):>8.6f} {np.argmax(evrs):>3d} '
                f'{max(da_gens):>7.1%} {np.argmax(da_gens):>3d}\n')
        lines.append('\n')

    lines.append('-' * 115 + '\n')
    lines.append('DA_max: best direction accuracy (all data). '
                 'Sep_max: max normalized centroid separation.\n')
    lines.append('Fisher: max Fisher discriminant ratio (between/within). '
                 'EVR: max explained variance ratio.\n')
    lines.append('DA_gen: best generalization DA (70/30 split, avg 5 trials). '
                 '@L: layer of maximum.\n')

    ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
            fontsize=6.5, va='top', fontfamily='monospace')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model key (default: 12b)')
    parser.add_argument('--iter', default='v7',
                        choices=['v7', 'v8'],
                        help='Iteration (default: v7)')
    args = parser.parse_args()

    for exp in args.exp:
        print(f"\n{'='*60}")
        print(f"Loading: {exp}")
        print(f"{'='*60}")

        data = load_all(exp)
        if not data:
            print(f"  No data found for {exp}, skipping."); continue

        best_cfgs = pick_best_cfg(data, exp)
        print(f"  Best configs: V7=cfg{best_cfgs['v7']}, V8=cfg{best_cfgs['v8']}")

        out_path = os.path.join(OUT_DIR, f'geometry_holistic_{exp}.pdf')
        print(f"  Output: {out_path}")

        with PdfPages(out_path) as pdf:
            print(f"  [1/11] PCA scatter...")
            pg_pca(pdf, data, exp, best_cfgs)

            print(f"  [2/11] t-SNE...")
            pg_tsne(pdf, data, exp, best_cfgs)

            print(f"  [3/11] Direction Accuracy curves...")
            pg_da(pdf, data, exp)

            print(f"  [4/11] Normalized Separation curves...")
            pg_sep(pdf, data, exp)

            print(f"  [5/11] Fisher Discriminant Ratio...")
            pg_fisher(pdf, data, exp)

            print(f"  [6/11] Projection distributions...")
            pg_projection_hist(pdf, data, exp, best_cfgs)

            print(f"  [7/11] Cross-layer direction cosine...")
            pg_cross_layer_cosine(pdf, data, exp, best_cfgs)

            print(f"  [8/11] Cross-config direction cosine...")
            pg_cross_config_cosine(pdf, data, exp)

            print(f"  [9/11] Generalization test...")
            pg_generalization(pdf, data, exp)

            print(f"  [10/11] Explained variance ratio...")
            pg_explained_var(pdf, data, exp)

            print(f"  [11/11] Summary table...")
            pg_summary(pdf, data, exp)

        print(f"  Done → {out_path}")

    print(f"\n{'='*60}\nAll done!\n{'='*60}")


if __name__ == "__main__":
    main()