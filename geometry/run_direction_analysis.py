#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry Script 3: Direction Analysis.

  1. Projection Distribution Histograms (all 49 layers, per config)
  2. Cross-layer direction cosine heatmap (49×49, per config)
  3. Cross-config direction cosine (cfg×cfg at key layers)
  4. Fisher Discriminant Ratio (all configs, all layers)
  5. Explained Variance Ratio (all configs, all layers)
  6. Generalization Test (full DA vs 70/30 split, all layers)
  7. Summary Table

Output: geometry/outputs/direction_analysis_{exp}_{iter}.pdf

Usage:
    python geometry/run_direction_analysis.py --exp exp2 --iter v7
    python geometry/run_direction_analysis.py --exp exp1a --iter v8
"""
import sys, os, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import EXP_CONFIG, CONFIG_MATRIX, config_label, MODEL_REGISTRY

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'geometry', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

CFG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
              5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}
CLASS_NAMES = {
    'exp1a': {0: 'Weekend', 1: 'Weekday'},
    'exp2':  {0: 'Unemployed', 1: 'Employed'},
}
ROWS, COLS = 4, 5
N_GEN_TRIALS = 10

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════
# Loading + Metrics
# ══════════════════════════════════════════════════════════════════════

def load_hidden(iter_name, exp, cfg_id, model_key='12b'):
    from config import get_iter_model_dirs
    dirs = get_iter_model_dirs(model_key, iter_name)
    path = os.path.join(dirs['hidden'], f'{exp}_cfg{cfg_id}.npz')
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels']


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def concept_dir(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def direction_accuracy(X, y):
    d = concept_dir(X, y)
    projs = X @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    return ((preds[y == 0] == 0).mean() + (preds[y == 1] == 1).mean()) / 2


def direction_accuracy_split(X, y):
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]
    d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    projs = X[te] @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    yt = y[te]
    return ((preds[yt == 0] == 0).mean() + (preds[yt == 1] == 1).mean()) / 2


def normalized_separation(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    return np.linalg.norm(c1 - c0) / np.sqrt(X.shape[1])


def fisher_ratio(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    between = np.linalg.norm(c1 - c0) ** 2
    within = np.var(X[y == 0], axis=0).sum() + np.var(X[y == 1], axis=0).sum()
    return between / max(within, 1e-10)


def explained_var_ratio(X, y):
    d = concept_dir(X, y)
    dn = d / max(np.linalg.norm(d), 1e-10)
    projs = X @ dn
    return np.var(projs) / max(np.var(X, axis=0).sum(), 1e-10)


# ══════════════════════════════════════════════════════════════════════
# 1. Projection Distribution Histograms
# ══════════════════════════════════════════════════════════════════════

def pg_proj_hist(pdf, X, y, exp, iter_name, cfg_id):
    cn = CLASS_NAMES[exp]
    cl = config_label(exp, cfg_id)
    n_layers = X.shape[1]
    per_page = ROWS * COLS

    for page_start in range(0, n_layers, per_page):
        page_layers = list(range(page_start, min(page_start + per_page, n_layers)))
        n_plots = len(page_layers)
        nrows = min(ROWS, (n_plots + COLS - 1) // COLS)

        fig, axes = plt.subplots(nrows, COLS, figsize=(3.5 * COLS, 3 * nrows),
                                  squeeze=False)
        for idx, l in enumerate(page_layers):
            ax = axes[idx // COLS, idx % COLS]
            Xl = X[:, l, :]
            d = concept_dir(Xl, y)
            dn = d / max(np.linalg.norm(d), 1e-10)
            projs = Xl @ dn

            for c, color in [(0, '#FF5722'), (1, '#2196F3')]:
                ax.hist(projs[y == c], bins=30, alpha=0.45, color=color,
                        density=True, label=cn[c] if idx == 0 else None)

            da = direction_accuracy(Xl, y)
            ax.set_title(f'L{l} ({da:.0%})', fontsize=7, fontweight='bold')
            ax.tick_params(labelsize=4)

        for idx in range(n_plots, nrows * COLS):
            axes[idx // COLS, idx % COLS].set_visible(False)

        if page_start == 0:
            axes[0, 0].legend(fontsize=6, loc='best')

        p = page_start // per_page + 1
        fig.suptitle(f'Projection Distribution — {iter_name.upper()}/{exp}/cfg{cfg_id} '
                     f'({cl}) — Page {p}',
                     fontsize=10, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=120); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 2. Cross-Layer Direction Cosine Heatmap
# ══════════════════════════════════════════════════════════════════════

def pg_cross_layer(pdf, X, y, exp, iter_name, cfg_id):
    cl = config_label(exp, cfg_id)
    n_layers = X.shape[1]
    dirs = [concept_dir(X[:, l, :], y) for l in range(n_layers)]

    mat = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            mat[i, j] = cosine_sim(dirs[i], dirs[j])

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, aspect='equal', origin='lower', cmap='RdYlGn',
                    vmin=-0.5, vmax=1.0)
    ax.set(xlabel='Layer', ylabel='Layer')
    ax.set_xticks(range(0, n_layers, 6))
    ax.set_yticks(range(0, n_layers, 6))
    ax.set_title(f'Cross-Layer Direction Cosine\n'
                 f'{iter_name.upper()}/{exp}/cfg{cfg_id} ({cl})',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 3. Cross-Config Direction Cosine
# ══════════════════════════════════════════════════════════════════════

def pg_cross_config(pdf, all_data, exp, iter_name):
    cfgs = sorted(all_data.keys())
    n_cfg = len(cfgs)
    if n_cfg < 2:
        return

    key_layers = [0, 6, 12, 20, 25, 28, 33, 40, 48]
    n_layers_model = next(iter(all_data.values()))[0].shape[1]
    key_layers = [l for l in key_layers if l < n_layers_model]

    ncols = min(len(key_layers), 5)
    nrows = (len(key_layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                              squeeze=False)

    for idx, l in enumerate(key_layers):
        ax = axes[idx // ncols, idx % ncols]
        dirs = {}
        for cid in cfgs:
            X, y = all_data[cid]
            if l < X.shape[1]:
                dirs[cid] = concept_dir(X[:, l, :], y)

        mat = np.zeros((n_cfg, n_cfg))
        for i, ci in enumerate(cfgs):
            for j, cj in enumerate(cfgs):
                if ci in dirs and cj in dirs:
                    mat[i, j] = cosine_sim(dirs[ci], dirs[cj])

        im = ax.imshow(mat, aspect='equal', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(n_cfg))
        ax.set_xticklabels([f'c{c}' for c in cfgs], fontsize=5, rotation=45)
        ax.set_yticks(range(n_cfg))
        ax.set_yticklabels([f'c{c}' for c in cfgs], fontsize=5)
        ax.set_title(f'L{l}', fontsize=8, fontweight='bold')
        for i in range(n_cfg):
            for j in range(n_cfg):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=4, color='k' if mat[i, j] > 0.5 else 'w')

    for idx in range(len(key_layers), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'Cross-Config Direction Cosine — {iter_name.upper()}/{exp}\n'
                 f'(do different configs find the same concept direction?)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4. Fisher Discriminant Ratio
# ══════════════════════════════════════════════════════════════════════

def pg_fisher(pdf, all_data, exp, iter_name):
    fig, ax = plt.subplots(figsize=(14, 7))
    for cid in sorted(all_data):
        X, y = all_data[cid]
        frs = [fisher_ratio(X[:, l, :], y) for l in range(X.shape[1])]
        cl = config_label(exp, cid)
        ax.plot(frs, color=CFG_COLORS[cid], lw=1.5, label=f'cfg{cid} ({cl})')
    ax.set(xlabel='Layer', ylabel='Fisher Ratio (between / within)')
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Fisher Discriminant Ratio '
                 f'({iter_name.upper()})',
                 fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. Explained Variance Ratio
# ══════════════════════════════════════════════════════════════════════

def pg_evr(pdf, all_data, exp, iter_name):
    fig, ax = plt.subplots(figsize=(14, 7))
    for cid in sorted(all_data):
        X, y = all_data[cid]
        evrs = [explained_var_ratio(X[:, l, :], y) for l in range(X.shape[1])]
        cl = config_label(exp, cid)
        ax.plot(evrs, color=CFG_COLORS[cid], lw=1.5, label=f'cfg{cid} ({cl})')
    ax.set(xlabel='Layer', ylabel='Explained Variance Ratio')
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Concept Direction Explained Variance '
                 f'({iter_name.upper()})\n'
                 f'(fraction of total activation variance along concept direction)',
                 fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6. Generalization Test
# ══════════════════════════════════════════════════════════════════════

def pg_generalization(pdf, all_data, exp, iter_name):
    fig, ax = plt.subplots(figsize=(14, 7))
    for cid in sorted(all_data):
        X, y = all_data[cid]
        n_layers = X.shape[1]
        cl = config_label(exp, cid)

        da_full = [direction_accuracy(X[:, l, :], y) for l in range(n_layers)]
        da_test = []
        for l in range(n_layers):
            scores = [direction_accuracy_split(X[:, l, :], y)
                      for _ in range(N_GEN_TRIALS)]
            da_test.append(np.mean(scores))

        ax.plot(da_full, ls='-', color=CFG_COLORS[cid], lw=1.5, alpha=0.8,
                label=f'cfg{cid} full')
        ax.plot(da_test, ls='--', color=CFG_COLORS[cid], lw=1.5, alpha=0.6,
                label=f'cfg{cid} test (30%)')

    ax.set(xlabel='Layer', ylabel='Direction Accuracy', ylim=(0.45, 1.0))
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_title(f'{EXP_CONFIG[exp]["name"]} — Generalization Test '
                 f'({iter_name.upper()})\n'
                 f'(solid=full data, dashed=held-out 30%, avg {N_GEN_TRIALS} splits)',
                 fontweight='bold')
    ax.legend(fontsize=5, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.1)
    plt.tight_layout(); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 7. Summary Table
# ══════════════════════════════════════════════════════════════════════

def pg_summary(pdf, all_data, exp, iter_name):
    fig, ax = plt.subplots(figsize=(20, 12)); ax.axis('off')
    lines = [f'{EXP_CONFIG[exp]["name"]} — Geometry Summary '
             f'({iter_name.upper()})\n',
             '=' * 110 + '\n\n']

    lines.append(
        f'{"Cfg":>4s} {"Label":>22s} '
        f'{"DA":>6s} {"@L":>3s} '
        f'{"Sep":>8s} {"@L":>3s} '
        f'{"Fisher":>8s} {"@L":>3s} '
        f'{"EVR":>8s} {"@L":>3s} '
        f'{"DA_gen":>7s} {"@L":>3s}\n')
    lines.append('-' * 110 + '\n')

    for cid in sorted(all_data):
        X, y = all_data[cid]
        n_layers = X.shape[1]
        cl = config_label(exp, cid)

        das = [direction_accuracy(X[:, l, :], y) for l in range(n_layers)]
        seps = [normalized_separation(X[:, l, :], y) for l in range(n_layers)]
        frs = [fisher_ratio(X[:, l, :], y) for l in range(n_layers)]
        evrs = [explained_var_ratio(X[:, l, :], y) for l in range(n_layers)]
        da_gens = [np.mean([direction_accuracy_split(X[:, l, :], y)
                            for _ in range(5)])
                   for l in range(n_layers)]

        lines.append(
            f'{cid:>4d} {cl:>22s} '
            f'{max(das):>6.1%} {np.argmax(das):>3d} '
            f'{max(seps):>8.5f} {np.argmax(seps):>3d} '
            f'{max(frs):>8.5f} {np.argmax(frs):>3d} '
            f'{max(evrs):>8.6f} {np.argmax(evrs):>3d} '
            f'{max(da_gens):>7.1%} {np.argmax(da_gens):>3d}\n')

    lines.append('\n' + '-' * 110 + '\n')
    lines.append('DA: best direction accuracy (all data). '
                 'Sep: max normalized centroid separation.\n')
    lines.append('Fisher: max between/within variance ratio. '
                 'EVR: max explained variance ratio.\n')
    lines.append('DA_gen: best generalization DA (70/30 split, avg 5 trials).\n')

    ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace')
    fig.suptitle(f'Geometry Summary — {iter_name.upper()}/{exp}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, choices=['exp1a', 'exp2'])
    parser.add_argument('--iter', required=True, choices=['v7', 'v8'])
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model key (default: 12b)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Direction Analysis: {args.iter}/{args.exp}")
    print(f"{'='*60}")

    all_data = {}
    for cfg in CONFIG_MATRIX[args.exp]:
        cid = cfg['id']
        X, y = load_hidden(args.iter, args.exp, cid, args.model)
        if X is not None:
            all_data[cid] = (X, y)
            print(f"  cfg{cid}: {X.shape}")

    if not all_data:
        print("  No data found."); return

    out_path = os.path.join(OUT_DIR,
                            f'direction_analysis_{args.exp}_{args.iter}.pdf')
    print(f"  Output: {out_path}")

    t0 = time.time()
    with PdfPages(out_path) as pdf:
        # 1. Projection histograms (per config)
        for cid in sorted(all_data):
            X, y = all_data[cid]
            print(f"  [1] Projection hist cfg{cid}...")
            pg_proj_hist(pdf, X, y, args.exp, args.iter, cid)

        # 2. Cross-layer cosine (per config)
        for cid in sorted(all_data):
            X, y = all_data[cid]
            print(f"  [2] Cross-layer cosine cfg{cid}...")
            pg_cross_layer(pdf, X, y, args.exp, args.iter, cid)

        # 3. Cross-config cosine
        print(f"  [3] Cross-config cosine...")
        pg_cross_config(pdf, all_data, args.exp, args.iter)

        # 4. Fisher
        print(f"  [4] Fisher ratio...")
        pg_fisher(pdf, all_data, args.exp, args.iter)

        # 5. EVR
        print(f"  [5] Explained variance ratio...")
        pg_evr(pdf, all_data, args.exp, args.iter)

        # 6. Generalization
        print(f"  [6] Generalization test...")
        pg_generalization(pdf, all_data, args.exp, args.iter)

        # 7. Summary
        print(f"  [7] Summary table...")
        pg_summary(pdf, all_data, args.exp, args.iter)

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
    print(f"  Done → {out_path}")


if __name__ == "__main__":
    main()
