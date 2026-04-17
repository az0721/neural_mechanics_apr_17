# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Geometry analysis: PCA scatter, Direction Accuracy, Normalized Separation.
# All configs on one figure per experiment.

# Usage:
#     python geometry_v7.py --exp exp2
#     python geometry_v7.py --exp exp1a exp2
# """
# import sys, os, argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from config import EXP_CONFIG, CONFIG_MATRIX, get_model_dirs, config_label

# CONFIG_STYLES = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--', 7: '-.', 8: ':'}
# CONFIG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
#                  5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}
# LAYERS_PCA = [0, 12, 24, 36, 48]


# def direction_accuracy(X, y):
#     c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
#     d = c1 - c0
#     projs = X @ d
#     thr = np.median(projs)
#     preds = (projs > thr).astype(int)
#     acc0 = (preds[y == 0] == 0).mean()
#     acc1 = (preds[y == 1] == 1).mean()
#     return (acc0 + acc1) / 2


# def normalized_separation(X, y):
#     c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
#     return np.linalg.norm(c1 - c0) / np.sqrt(X.shape[1])


# def make_pca_page(all_data, exp_name, exp_cfg, tag, dirs):
#     """PCA scatter: rows=configs, cols=layers."""
#     configs = [d for d in all_data if d is not None]
#     n_cfg = len(configs)
#     if n_cfg == 0:
#         return
#     n_layers = len(LAYERS_PCA)
#     class_names = {0: 'Class 0', 1: 'Class 1'}
#     if 'weekday' in exp_cfg['label_col'].lower():
#         class_names = {0: 'Weekend', 1: 'Weekday'}
#     elif 'employed' in exp_cfg['label_col'].lower():
#         class_names = {0: 'Unemployed', 1: 'Employed'}

#     fig, axes = plt.subplots(n_cfg, n_layers, figsize=(4 * n_layers, 3.5 * n_cfg))
#     if n_cfg == 1:
#         axes = axes.reshape(1, -1)

#     for row, d in enumerate(configs):
#         X, y, cfg_id, clabel = d['X'], d['y'], d['cfg_id'], d['clabel']
#         for col, layer in enumerate(LAYERS_PCA):
#             ax = axes[row, col]
#             Xl = X[:, layer, :]
#             pca = PCA(n_components=2)
#             pc = pca.fit_transform(Xl)
#             ev = pca.explained_variance_ratio_

#             for cls in [0, 1]:
#                 mask = y == cls
#                 ax.scatter(pc[mask, 0], pc[mask, 1], s=8, alpha=0.4,
#                            label=class_names[cls] if row == 0 and col == n_layers - 1 else None)
#             ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=7)
#             ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=7)
#             ax.tick_params(labelsize=6)
#             if row == 0:
#                 ax.set_title(f"Layer {layer}", fontsize=9)
#             if col == 0:
#                 ax.set_ylabel(f"cfg{cfg_id}\n({clabel})\n\nPC2 ({ev[1]:.1%})", fontsize=7)

#     if configs:
#         axes[0, -1].legend(fontsize=7, loc='upper right')
#     fig.suptitle(f"{exp_cfg['name']} — {tag} (Iter 7, 15-min)\n"
#                  f"PCA Scatter (rows=configs, cols=layers)", fontweight='bold', fontsize=11)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     out = os.path.join(dirs['results'], f"geometry_{exp_name}_pca.png")
#     plt.savefig(out, dpi=200, bbox_inches='tight')
#     print(f"  PCA saved: {out}")
#     plt.close()


# def make_da_sep_page(all_data, exp_name, exp_cfg, tag, dirs):
#     """Direction Accuracy + Normalized Separation, all configs on one figure."""
#     configs = [d for d in all_data if d is not None]
#     if not configs:
#         return

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     baseline = 1.0 / exp_cfg['n_classes']

#     for d in configs:
#         X, y = d['X'], d['y']
#         cfg_id, clabel = d['cfg_id'], d['clabel']
#         n_layers = X.shape[1]
#         das, seps = [], []
#         for layer in range(n_layers):
#             das.append(direction_accuracy(X[:, layer, :], y))
#             seps.append(normalized_separation(X[:, layer, :], y))

#         das, seps = np.array(das), np.array(seps)
#         best_l = das.argmax()
#         ls = CONFIG_STYLES.get(cfg_id, '-')
#         color = CONFIG_COLORS.get(cfg_id, '#333')

#         ax1.plot(range(n_layers), das, ls, color=color, lw=2,
#                  label=f"cfg{cfg_id} ({clabel}) {das.max():.1%}@L{best_l}")
#         ax1.scatter(best_l, das.max(), s=40, color=color, zorder=5)
#         ax2.plot(range(n_layers), seps, ls, color=color, lw=2,
#                  label=f"cfg{cfg_id} ({clabel})")

#     ax1.axhline(baseline, color='red', ls='--', alpha=0.4, label=f'Baseline {baseline:.0%}')
#     ax1.set_ylabel('Balanced Accuracy')
#     ax1.set_title('Direction Accuracy by Layer')
#     ax1.legend(fontsize=6, loc='lower right')
#     ax1.grid(True, alpha=0.15)
#     ax1.set_ylim(0, 1.0)

#     ax2.set_xlabel('Layer')
#     ax2.set_ylabel('L2 / sqrt(hidden_dim)')
#     ax2.set_title('Normalized Separation by Layer')
#     ax2.legend(fontsize=6, loc='upper left')
#     ax2.grid(True, alpha=0.15)

#     fig.suptitle(f"{exp_cfg['name']} — {tag} (Iter 7, 15-min)\n"
#                  f"Direction Accuracy & Normalized Separation (all configs)",
#                  fontweight='bold', fontsize=11)
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     out = os.path.join(dirs['results'], f"geometry_{exp_name}_da_sep.png")
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     print(f"  DA+Sep saved: {out}")
#     plt.close()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
#     parser.add_argument('--model', default='12b')
#     args = parser.parse_args()

#     dirs = get_model_dirs(args.model)
#     tag = f"gemma3_{args.model}b_it"

#     for exp in args.exp:
#         exp_cfg = EXP_CONFIG[exp]
#         configs = CONFIG_MATRIX[exp]

#         print(f"\n{'='*60}")
#         print(f"Geometry: {exp} | {tag}")
#         print(f"{'='*60}")

#         all_data = []
#         for cfg in configs:
#             cfg_id = cfg['id']
#             clabel = config_label(exp, cfg_id)
#             path = os.path.join(dirs['hidden'], f"{exp}_cfg{cfg_id}.npz")
#             if not os.path.exists(path):
#                 print(f"  cfg{cfg_id}: not found")
#                 all_data.append(None)
#                 continue

#             data = np.load(path, allow_pickle=True)
#             X, y = data['hidden_states'], data['labels']
#             print(f"  cfg{cfg_id} ({clabel}): {X.shape[0]} samples")
#             all_data.append({'X': X, 'y': y, 'cfg_id': cfg_id, 'clabel': clabel})

#         make_pca_page(all_data, exp, exp_cfg, tag, dirs)
#         make_da_sep_page(all_data, exp, exp_cfg, tag, dirs)

#     print(f"\nDone.")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry Analysis — PCA, Direction Accuracy, Normalized Separation.

Output: geometry_{exp}.pdf per experiment (separate PDFs).

Pages per experiment:
  1. PCA scatter grid (rows=configs, cols=key layers)
  2. Direction Accuracy curves (all configs, all layers)
  3. Normalized Separation curves (all configs, all layers)
  4. Combined DA + Sep per config subplots
  5. Summary table

Usage:
    python geometry_v7.py --exp exp1a         # one experiment
    python geometry_v7.py --exp exp2          # one experiment
    python geometry_v7.py --exp exp1a exp2    # both (separate PDFs)
"""
import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXP_CONFIG, CONFIG_MATRIX, get_model_dirs, config_label

CFG_STYLES = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--', 7: '-.', 8: ':'}
CFG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
              5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}

# Key layers for PCA scatter
LAYERS_PCA = [0, 12, 24, 33, 48]


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def direction_accuracy(X, y):
    """Balanced accuracy of median-split projection onto class centroid difference."""
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    d = c1 - c0
    projs = X @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    acc0 = (preds[y == 0] == 0).mean()
    acc1 = (preds[y == 1] == 1).mean()
    return (acc0 + acc1) / 2


def normalized_separation(X, y):
    """L2 distance between class centroids, normalized by sqrt(hidden_dim)."""
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    return np.linalg.norm(c1 - c0) / np.sqrt(X.shape[1])


def compute_all_layers(X, y):
    """Compute DA and Sep for all layers."""
    n_layers = X.shape[1]
    das = np.array([direction_accuracy(X[:, l, :], y) for l in range(n_layers)])
    seps = np.array([normalized_separation(X[:, l, :], y) for l in range(n_layers)])
    return das, seps


# ══════════════════════════════════════════════════════════════════════
# Page 1: PCA Scatter Grid
# ══════════════════════════════════════════════════════════════════════

def page_pca(pdf, configs, exp_cfg, tag):
    n_cfg = len(configs)
    if n_cfg == 0:
        return
    n_layers = len(LAYERS_PCA)
    cn = {0: 'Class 0', 1: 'Class 1'}
    if 'weekday' in exp_cfg['label_col'].lower():
        cn = {0: 'Weekend', 1: 'Weekday'}
    elif 'employed' in exp_cfg['label_col'].lower():
        cn = {0: 'Unemployed', 1: 'Employed'}

    fig, axes = plt.subplots(n_cfg, n_layers, figsize=(4 * n_layers, 3.5 * n_cfg),
                              squeeze=False)
    colors_cls = {0: '#2196F3', 1: '#FF5722'}

    for row, d in enumerate(configs):
        X, y = d['X'], d['y']
        cfg_id, clabel = d['cfg_id'], d['clabel']
        for col, layer in enumerate(LAYERS_PCA):
            ax = axes[row, col]
            # Clamp layer to valid range
            l = min(layer, X.shape[1] - 1)
            Xl = X[:, l, :]
            pca = PCA(n_components=2)
            pc = pca.fit_transform(Xl)
            ev = pca.explained_variance_ratio_

            for cls in [0, 1]:
                mask = y == cls
                ax.scatter(pc[mask, 0], pc[mask, 1], s=6, alpha=0.35,
                           color=colors_cls[cls],
                           label=cn[cls] if row == 0 and col == n_layers - 1 else None)
            ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"Layer {l}", fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"cfg{cfg_id}\n({clabel})\n\nPC2 ({ev[1]:.1%})", fontsize=7)
            else:
                ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=7)

    axes[0, -1].legend(fontsize=7, loc='upper right')
    fig.suptitle(f"{exp_cfg['name']} — PCA Scatter\n"
                 f"{tag} | Layers: {LAYERS_PCA}", fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 2: Direction Accuracy Curves
# ══════════════════════════════════════════════════════════════════════

def page_da(pdf, configs, exp_cfg, tag):
    if not configs:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    baseline = 1.0 / exp_cfg['n_classes']

    for d in configs:
        das = d['das']
        cfg_id, clabel = d['cfg_id'], d['clabel']
        best_l = das.argmax()
        ax.plot(range(len(das)), das, CFG_STYLES.get(cfg_id, '-'),
                color=CFG_COLORS.get(cfg_id, '#333'), lw=2,
                label=f"cfg{cfg_id} ({clabel}) — {das.max():.1%} @ L{best_l}")
        ax.scatter(best_l, das.max(), s=50, color=CFG_COLORS.get(cfg_id, '#333'), zorder=5)

    ax.axhline(baseline, color='red', ls='--', alpha=0.4, label=f'Chance {baseline:.0%}')
    ax.set(xlabel='Layer', ylabel='Balanced Accuracy')
    ax.set_title(f'{exp_cfg["name"]} — Direction Accuracy by Layer\n{tag}',
                 fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 3: Normalized Separation Curves
# ══════════════════════════════════════════════════════════════════════

def page_sep(pdf, configs, exp_cfg, tag):
    if not configs:
        return
    fig, ax = plt.subplots(figsize=(14, 7))

    for d in configs:
        seps = d['seps']
        cfg_id, clabel = d['cfg_id'], d['clabel']
        best_l = seps.argmax()
        ax.plot(range(len(seps)), seps, CFG_STYLES.get(cfg_id, '-'),
                color=CFG_COLORS.get(cfg_id, '#333'), lw=2,
                label=f"cfg{cfg_id} ({clabel}) — {seps.max():.4f} @ L{best_l}")
        ax.scatter(best_l, seps.max(), s=50, color=CFG_COLORS.get(cfg_id, '#333'), zorder=5)

    ax.set(xlabel='Layer', ylabel='L2 / sqrt(hidden_dim)')
    ax.set_title(f'{exp_cfg["name"]} — Normalized Separation by Layer\n{tag}',
                 fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 4: Combined DA + Sep per config subplots
# ══════════════════════════════════════════════════════════════════════

def page_combined(pdf, configs, exp_cfg, tag):
    if not configs:
        return
    n = len(configs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    baseline = 1.0 / exp_cfg['n_classes']

    for idx, d in enumerate(configs):
        ax = axes[idx // ncols, idx % ncols]
        das, seps = d['das'], d['seps']
        cfg_id, clabel = d['cfg_id'], d['clabel']
        layers = range(len(das))

        ax.plot(layers, das, 'b-', lw=2, label=f'DA {das.max():.1%}@L{das.argmax()}')
        ax.set_ylabel('Direction Accuracy', color='blue', fontsize=8)
        ax.axhline(baseline, color='blue', ls='--', alpha=0.2)
        ax.set_ylim(0.4, 1.0)
        ax.tick_params(axis='y', labelcolor='blue', labelsize=7)

        ax2 = ax.twinx()
        ax2.plot(layers, seps, 'r--', lw=2, label=f'Sep {seps.max():.4f}@L{seps.argmax()}')
        ax2.set_ylabel('Norm. Separation', color='red', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=7)

        ax.set_title(f'cfg{cfg_id} ({clabel})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=8)
        ax.grid(True, alpha=0.15)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='lower right')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{exp_cfg["name"]} — DA + Separation Per Config\n{tag}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 5: Summary Table
# ══════════════════════════════════════════════════════════════════════

def page_summary(pdf, configs, exp_cfg, tag):
    fig, ax = plt.subplots(figsize=(16, 8)); ax.axis('off')
    h = (f'{"cfg":>5s} {"label":>22s} {"DA_L":>5s} {"DA%":>7s} '
         f'{"Sep_L":>5s} {"Sep":>8s} {"N":>6s}\n')
    sep = '─' * 70 + '\n'
    lines = [f'{exp_cfg["name"]} — Geometry Summary | {tag}\n\n', h, sep]

    for d in configs:
        das, seps = d['das'], d['seps']
        dl, sl = das.argmax(), seps.argmax()
        lines.append(f'{d["cfg_id"]:>5d} {d["clabel"]:>22s} '
                     f'L{dl:>3d} {das[dl]:>7.1%} L{sl:>3d} '
                     f'{seps[sl]:>8.5f} {d["X"].shape[0]:>6d}\n')

    lines.append(sep)
    lines.append('\nDA = Direction Accuracy (balanced acc of median-split projection)\n')
    lines.append('Sep = Normalized Separation (L2(centroid diff) / sqrt(hidden_dim))\n')
    lines.append(f'PCA layers: {LAYERS_PCA}\n')

    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace')
    fig.suptitle(f'{exp_cfg["name"]} — Summary', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--model', default='12b')
    args = parser.parse_args()

    dirs = get_model_dirs(args.model)
    tag = f"gemma3_{args.model}b_it"

    for exp in args.exp:
        exp_cfg = EXP_CONFIG[exp]
        cfglist = CONFIG_MATRIX[exp]
        pdf_path = os.path.join(dirs['results'], f'geometry_{exp}.pdf')

        print(f"\n{'='*60}")
        print(f"Geometry: {exp} ({exp_cfg['name']})")
        print(f"PDF: {pdf_path}")
        print(f"{'='*60}")

        configs = []
        for cfg in cfglist:
            cfg_id = cfg['id']
            clabel = config_label(exp, cfg_id)
            path = os.path.join(dirs['hidden'], f"{exp}_cfg{cfg_id}.npz")
            if not os.path.exists(path):
                print(f"  cfg{cfg_id}: not found"); continue

            data = np.load(path, allow_pickle=True)
            X, y = data['hidden_states'], data['labels']
            das, seps = compute_all_layers(X, y)
            print(f"  cfg{cfg_id} ({clabel}): N={X.shape[0]}, "
                  f"DA={das.max():.1%}@L{das.argmax()}, "
                  f"Sep={seps.max():.4f}@L{seps.argmax()}")
            configs.append({'X': X, 'y': y, 'cfg_id': cfg_id,
                            'clabel': clabel, 'das': das, 'seps': seps})

        if not configs:
            print("  No data found"); continue

        with PdfPages(pdf_path) as pdf:
            print(f"  P1: PCA scatter")
            page_pca(pdf, configs, exp_cfg, tag)

            print(f"  P2: Direction Accuracy curves")
            page_da(pdf, configs, exp_cfg, tag)

            print(f"  P3: Normalized Separation curves")
            page_sep(pdf, configs, exp_cfg, tag)

            print(f"  P4: Combined DA+Sep per config")
            page_combined(pdf, configs, exp_cfg, tag)

            print(f"  P5: Summary table")
            page_summary(pdf, configs, exp_cfg, tag)

        print(f"\n  Done! → {pdf_path}")

    print(f"\n{'='*60}\nAll done.\n{'='*60}")


if __name__ == "__main__":
    main()
