#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Iteration Comparison: V7 (no hint) vs V8 (concept hint).

Compares hidden states between Iter 7 and Iter 8 to understand
how the concept hint affects internal representations.

Output: cross_iteration_comparison/v7_v8_hidden_state_comparison.pdf

Pages:
  Per experiment:
    1. Overall cosine similarity (V7 mean vs V8 mean) per layer per config
    2. Per-category cosine sim: emp_v7 vs emp_v8, unemp_v7 vs unemp_v8
    3. Concept direction cosine sim: V_direction_v7 vs V_direction_v8 per layer
    4. Separation comparison: V7 vs V8 normalized separation per layer
    5. Probing-relevant: centroid shift magnitude per layer
    6. Heatmap: layer × config cosine similarity matrix
    7. Summary table

Usage:
    python compare_v7_v8.py                             # 12b, both exps
    python compare_v7_v8.py --model gemma4_31b          # Gemma 4 31B
    python compare_v7_v8.py --model gemma4_31b --exp exp2
"""
import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, CONFIG_MATRIX, config_label,
                    MODEL_REGISTRY, get_iter_model_dirs)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
              5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}
CFG_STYLES = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--', 7: '-.', 8: ':'}


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def load_hidden(base_dir, exp, cfg_id):
    path = os.path.join(base_dir, f"{exp}_cfg{cfg_id}.npz")
    if not os.path.exists(path):
        return None, None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels'], d['meta']


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalized_separation(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    return np.linalg.norm(c1 - c0) / np.sqrt(X.shape[1])


def direction_accuracy(X, y):
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    d = c1 - c0
    projs = X @ d
    thr = np.median(projs)
    preds = (projs > thr).astype(int)
    return ((preds[y == 0] == 0).mean() + (preds[y == 1] == 1).mean()) / 2


# ══════════════════════════════════════════════════════════════════════
# Page 1: Overall Cosine Similarity (mean_v7 vs mean_v8 per layer)
# ══════════════════════════════════════════════════════════════════════

def page_overall_cosine(pdf, pairs, exp_cfg):
    fig, ax = plt.subplots(figsize=(14, 7))
    for cfg_id, p in sorted(pairs.items()):
        X7, X8 = p['X7'], p['X8']
        n_layers = X7.shape[1]
        sims = [cosine_sim(X7[:, l, :].mean(0), X8[:, l, :].mean(0))
                for l in range(n_layers)]
        ax.plot(range(n_layers), sims, CFG_STYLES.get(cfg_id, '-'),
                color=CFG_COLORS.get(cfg_id, '#333'), lw=2,
                label=f"cfg{cfg_id} ({p['clabel']})")
    ax.set(xlabel='Layer', ylabel='Cosine Similarity')
    ax.set_title(f'{exp_cfg["name"]} — Overall Mean Activation Similarity\n'
                 f'cos(mean_v7, mean_v8) per layer — high = hint didn\'t change much',
                 fontweight='bold')
    ax.set_ylim(0.9, 1.001)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.15)
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 2: Per-Category Cosine (emp_v7 vs emp_v8, unemp_v7 vs unemp_v8)
# ══════════════════════════════════════════════════════════════════════

def page_per_category_cosine(pdf, pairs, exp_cfg):
    cat_names = {0: 'Class0', 1: 'Class1'}
    if 'employed' in exp_cfg['label_col']:
        cat_names = {0: 'Unemployed', 1: 'Employed'}
    elif 'weekday' in exp_cfg['label_col']:
        cat_names = {0: 'Weekend', 1: 'Weekday'}

    cfgs = sorted(pairs.keys())
    n = len(cfgs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, cfg_id in enumerate(cfgs):
        p = pairs[cfg_id]
        ax = axes[idx // ncols, idx % ncols]
        X7, y7, X8, y8 = p['X7'], p['y7'], p['X8'], p['y8']
        n_layers = X7.shape[1]

        for cat, color, ls in [(0, '#FF5722', '--'), (1, '#2196F3', '-')]:
            sims = [cosine_sim(X7[y7 == cat, l, :].mean(0),
                               X8[y8 == cat, l, :].mean(0))
                    for l in range(n_layers)]
            ax.plot(range(n_layers), sims, ls, color=color, lw=2,
                    label=f'{cat_names[cat]}')

        ax.set_title(f'cfg{cfg_id} ({p["clabel"]})', fontsize=9, fontweight='bold')
        ax.set_ylim(0.85, 1.001)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0: ax.set_ylabel('Cosine Sim', fontsize=8)
        if idx // ncols == nrows - 1: ax.set_xlabel('Layer', fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{exp_cfg["name"]} — Per-Category Cosine Similarity\n'
                 f'cos(category_mean_v7, category_mean_v8) — does hint affect categories differently?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 3: Concept Direction Cosine (V7 direction vs V8 direction)
# ══════════════════════════════════════════════════════════════════════

def page_direction_cosine(pdf, pairs, exp_cfg):
    fig, ax = plt.subplots(figsize=(14, 7))
    for cfg_id, p in sorted(pairs.items()):
        X7, y7, X8, y8 = p['X7'], p['y7'], p['X8'], p['y8']
        n_layers = X7.shape[1]
        sims = []
        for l in range(n_layers):
            d7 = X7[y7 == 1, l, :].mean(0) - X7[y7 == 0, l, :].mean(0)
            d8 = X8[y8 == 1, l, :].mean(0) - X8[y8 == 0, l, :].mean(0)
            sims.append(cosine_sim(d7, d8))
        ax.plot(range(n_layers), sims, CFG_STYLES.get(cfg_id, '-'),
                color=CFG_COLORS.get(cfg_id, '#333'), lw=2,
                label=f"cfg{cfg_id} ({p['clabel']})")
    ax.set(xlabel='Layer', ylabel='Cosine Similarity')
    ax.set_title(f'{exp_cfg["name"]} — Concept Direction Similarity\n'
                 f'cos(V7_direction, V8_direction) — high = same concept axis found',
                 fontweight='bold')
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.15)
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 4: Separation Comparison (V7 vs V8 per layer)
# ══════════════════════════════════════════════════════════════════════

def page_separation_comparison(pdf, pairs, exp_cfg):
    cfgs = sorted(pairs.keys())
    n = len(cfgs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, cfg_id in enumerate(cfgs):
        p = pairs[cfg_id]
        ax = axes[idx // ncols, idx % ncols]
        X7, y7, X8, y8 = p['X7'], p['y7'], p['X8'], p['y8']
        n_layers = X7.shape[1]

        sep7 = [normalized_separation(X7[:, l, :], y7) for l in range(n_layers)]
        sep8 = [normalized_separation(X8[:, l, :], y8) for l in range(n_layers)]

        ax.plot(range(n_layers), sep7, 'b-', lw=2, label=f'V7 (no hint)')
        ax.plot(range(n_layers), sep8, 'r--', lw=2, label=f'V8 (hint)')
        ax.fill_between(range(n_layers), sep7, sep8, alpha=0.08,
                        color='green' if np.mean(sep8) >= np.mean(sep7) else 'red')
        ax.set_title(f'cfg{cfg_id} ({p["clabel"]})', fontsize=9, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0: ax.set_ylabel('Separation', fontsize=8)
        if idx // ncols == nrows - 1: ax.set_xlabel('Layer', fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{exp_cfg["name"]} — Normalized Separation: V7 vs V8\n'
                 f'Green fill = V8 higher (hint helps), Red = V8 lower',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 5: Direction Accuracy Comparison (V7 vs V8)
# ══════════════════════════════════════════════════════════════════════

def page_da_comparison(pdf, pairs, exp_cfg):
    cfgs = sorted(pairs.keys())
    n = len(cfgs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              squeeze=False)
    baseline = 1.0 / exp_cfg['n_classes']

    for idx, cfg_id in enumerate(cfgs):
        p = pairs[cfg_id]
        ax = axes[idx // ncols, idx % ncols]
        X7, y7, X8, y8 = p['X7'], p['y7'], p['X8'], p['y8']
        n_layers = X7.shape[1]

        da7 = [direction_accuracy(X7[:, l, :], y7) for l in range(n_layers)]
        da8 = [direction_accuracy(X8[:, l, :], y8) for l in range(n_layers)]

        ax.plot(range(n_layers), da7, 'b-', lw=2,
                label=f'V7 {max(da7):.1%}@L{np.argmax(da7)}')
        ax.plot(range(n_layers), da8, 'r--', lw=2,
                label=f'V8 {max(da8):.1%}@L{np.argmax(da8)}')
        ax.axhline(baseline, color='gray', ls=':', alpha=0.3)
        ax.set_title(f'cfg{cfg_id} ({p["clabel"]})', fontsize=9, fontweight='bold')
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0: ax.set_ylabel('Dir. Accuracy', fontsize=8)
        if idx // ncols == nrows - 1: ax.set_xlabel('Layer', fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{exp_cfg["name"]} — Direction Accuracy: V7 vs V8\n'
                 f'Does the concept hint improve geometric separability?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 6: Centroid Shift Magnitude (how much did hint move each category?)
# ══════════════════════════════════════════════════════════════════════

def page_centroid_shift(pdf, pairs, exp_cfg):
    cat_names = {0: 'Unemployed', 1: 'Employed'} if 'employed' in exp_cfg['label_col'] \
        else {0: 'Weekend', 1: 'Weekday'}

    fig, ax = plt.subplots(figsize=(14, 7))
    for cfg_id, p in sorted(pairs.items()):
        X7, y7, X8, y8 = p['X7'], p['y7'], p['X8'], p['y8']
        n_layers = X7.shape[1]
        shifts_0, shifts_1 = [], []
        for l in range(n_layers):
            c7_0 = X7[y7 == 0, l, :].mean(0)
            c8_0 = X8[y8 == 0, l, :].mean(0)
            c7_1 = X7[y7 == 1, l, :].mean(0)
            c8_1 = X8[y8 == 1, l, :].mean(0)
            shifts_0.append(np.linalg.norm(c8_0 - c7_0))
            shifts_1.append(np.linalg.norm(c8_1 - c7_1))
        ax.plot(range(n_layers), shifts_0, '--',
                color=CFG_COLORS.get(cfg_id, '#333'), lw=1, alpha=0.5)
        ax.plot(range(n_layers), shifts_1, '-',
                color=CFG_COLORS.get(cfg_id, '#333'), lw=2,
                label=f'cfg{cfg_id}')

    # Dummy for legend
    ax.plot([], [], 'k-', lw=2, label=f'{cat_names[1]} (solid)')
    ax.plot([], [], 'k--', lw=1, label=f'{cat_names[0]} (dashed)')

    ax.set(xlabel='Layer', ylabel='L2 Distance')
    ax.set_title(f'{exp_cfg["name"]} — Centroid Shift: ||mean_v8 - mean_v7||\n'
                 f'How much did the concept hint move each category\'s centroid?',
                 fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.15)
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 7: Heatmap — Layer × Config cosine similarity
# ══════════════════════════════════════════════════════════════════════

def page_heatmap(pdf, pairs, exp_cfg):
    cfgs = sorted(pairs.keys())
    n_layers = next(iter(pairs.values()))['X7'].shape[1]

    # Three heatmaps: overall, direction, separation_delta
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # 1. Overall cosine
    mat_overall = np.zeros((len(cfgs), n_layers))
    for i, cfg_id in enumerate(cfgs):
        p = pairs[cfg_id]
        for l in range(n_layers):
            mat_overall[i, l] = cosine_sim(
                p['X7'][:, l, :].mean(0), p['X8'][:, l, :].mean(0))

    im = axes[0].imshow(mat_overall, aspect='auto', cmap='RdYlGn',
                         vmin=0.95, vmax=1.0)
    axes[0].set_yticks(range(len(cfgs)))
    axes[0].set_yticklabels([f'cfg{c}' for c in cfgs], fontsize=8)
    axes[0].set_xlabel('Layer')
    axes[0].set_title('Overall Cosine Sim\n(mean_v7 vs mean_v8)', fontweight='bold')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # 2. Direction cosine
    mat_dir = np.zeros((len(cfgs), n_layers))
    for i, cfg_id in enumerate(cfgs):
        p = pairs[cfg_id]
        for l in range(n_layers):
            d7 = p['X7'][p['y7'] == 1, l, :].mean(0) - p['X7'][p['y7'] == 0, l, :].mean(0)
            d8 = p['X8'][p['y8'] == 1, l, :].mean(0) - p['X8'][p['y8'] == 0, l, :].mean(0)
            mat_dir[i, l] = cosine_sim(d7, d8)

    im = axes[1].imshow(mat_dir, aspect='auto', cmap='RdYlGn',
                         vmin=-0.2, vmax=1.0)
    axes[1].set_yticks(range(len(cfgs)))
    axes[1].set_yticklabels([f'cfg{c}' for c in cfgs], fontsize=8)
    axes[1].set_xlabel('Layer')
    axes[1].set_title('Direction Cosine Sim\n(concept_axis_v7 vs v8)', fontweight='bold')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # 3. Separation delta (V8 - V7)
    mat_sep = np.zeros((len(cfgs), n_layers))
    for i, cfg_id in enumerate(cfgs):
        p = pairs[cfg_id]
        for l in range(n_layers):
            s7 = normalized_separation(p['X7'][:, l, :], p['y7'])
            s8 = normalized_separation(p['X8'][:, l, :], p['y8'])
            mat_sep[i, l] = s8 - s7

    vmax = max(abs(mat_sep.min()), abs(mat_sep.max()), 0.01)
    im = axes[2].imshow(mat_sep, aspect='auto', cmap='RdBu',
                         vmin=-vmax, vmax=vmax)
    axes[2].set_yticks(range(len(cfgs)))
    axes[2].set_yticklabels([f'cfg{c}' for c in cfgs], fontsize=8)
    axes[2].set_xlabel('Layer')
    axes[2].set_title('Δ Separation (V8 − V7)\n(blue=V8 better, red=V7 better)',
                       fontweight='bold')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    fig.suptitle(f'{exp_cfg["name"]} — Layer × Config Heatmaps',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 8: Summary Table
# ══════════════════════════════════════════════════════════════════════

def page_summary(pdf, pairs, exp_cfg):
    fig, ax = plt.subplots(figsize=(18, 10)); ax.axis('off')
    lines = [f'{exp_cfg["name"]} — V7 vs V8 Summary\n\n']
    lines.append(f'{"cfg":>5s} {"label":>22s} '
                 f'{"cos_mean":>9s} {"cos_dir":>9s} '
                 f'{"sep_v7":>8s} {"sep_v8":>8s} {"Δsep":>7s} '
                 f'{"DA_v7":>7s} {"DA_v8":>7s} {"ΔDA":>6s}\n')
    lines.append('─' * 100 + '\n')

    for cfg_id in sorted(pairs.keys()):
        p = pairs[cfg_id]
        X7, y7, X8, y8 = p['X7'], p['y7'], p['X8'], p['y8']

        # Best layer metrics
        sep7 = [normalized_separation(X7[:, l, :], y7) for l in range(X7.shape[1])]
        sep8 = [normalized_separation(X8[:, l, :], y8) for l in range(X8.shape[1])]
        da7 = [direction_accuracy(X7[:, l, :], y7) for l in range(X7.shape[1])]
        da8 = [direction_accuracy(X8[:, l, :], y8) for l in range(X8.shape[1])]

        bl7, bl8 = np.argmax(da7), np.argmax(da8)

        # Overall cosine at best layer
        cos_mean = cosine_sim(X7[:, bl7, :].mean(0), X8[:, bl7, :].mean(0))
        d7 = X7[y7 == 1, bl7, :].mean(0) - X7[y7 == 0, bl7, :].mean(0)
        d8 = X8[y8 == 1, bl7, :].mean(0) - X8[y8 == 0, bl7, :].mean(0)
        cos_dir = cosine_sim(d7, d8)

        lines.append(
            f'{cfg_id:>5d} {p["clabel"]:>22s} '
            f'{cos_mean:>9.4f} {cos_dir:>9.4f} '
            f'{max(sep7):>8.5f} {max(sep8):>8.5f} '
            f'{max(sep8)-max(sep7):>+7.5f} '
            f'{max(da7):>7.1%} {max(da8):>7.1%} '
            f'{max(da8)-max(da7):>+6.1%}\n')

    lines.append('─' * 100 + '\n')
    lines.append('\ncos_mean = cosine(overall_mean_v7, overall_mean_v8) at V7 best DA layer\n')
    lines.append('cos_dir = cosine(concept_direction_v7, concept_direction_v8) at V7 best DA layer\n')
    lines.append('sep = max normalized separation across all layers\n')
    lines.append('DA = max direction accuracy across all layers\n')
    lines.append('Δ = V8 value minus V7 value (positive = hint helped)\n')

    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace')
    fig.suptitle(f'{exp_cfg["name"]} — Numerical Summary',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model key (default: 12b)')
    args = parser.parse_args()

    # ── Model-aware path resolution ──
    v7_dirs = get_iter_model_dirs(args.model, 'v7')
    v8_dirs = get_iter_model_dirs(args.model, 'v8')
    v7_hidden = v7_dirs['hidden']
    v8_hidden = v8_dirs['hidden']

    if args.model == '12b':
        out_dir = os.path.join(BASE_DIR, 'cross_iteration_comparison')
    else:
        model_base = MODEL_REGISTRY[args.model]['hf_name']
        out_dir = os.path.join(model_base, 'cross_iteration_comparison')
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, 'v7_v8_hidden_state_comparison.pdf')
    print(f"{'='*60}")
    print(f"V7 vs V8 Hidden State Comparison")
    print(f"  Model: {args.model} ({MODEL_REGISTRY[args.model]['tag']})")
    print(f"  V7: {v7_hidden}")
    print(f"  V8: {v8_hidden}")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*60}")

    with PdfPages(pdf_path) as pdf:
        for exp in args.exp:
            exp_cfg = EXP_CONFIG[exp]
            cfgs = CONFIG_MATRIX[exp]
            print(f"\n  {exp} ({exp_cfg['name']}): {len(cfgs)} configs")

            pairs = {}
            for cfg in cfgs:
                cfg_id = cfg['id']
                clabel = config_label(exp, cfg_id)

                X7, y7, m7 = load_hidden(v7_hidden, exp, cfg_id)
                X8, y8, m8 = load_hidden(v8_hidden, exp, cfg_id)

                if X7 is None:
                    print(f"    cfg{cfg_id}: V7 not found, skip"); continue
                if X8 is None:
                    print(f"    cfg{cfg_id}: V8 not found, skip"); continue

                print(f"    cfg{cfg_id} ({clabel}): V7={X7.shape[0]}, V8={X8.shape[0]}")
                pairs[cfg_id] = {
                    'X7': X7, 'y7': y7, 'X8': X8, 'y8': y8, 'clabel': clabel
                }

            if not pairs:
                print(f"    No paired data found for {exp}")
                continue

            print(f"    Generating pages...")
            page_overall_cosine(pdf, pairs, exp_cfg)
            page_per_category_cosine(pdf, pairs, exp_cfg)
            page_direction_cosine(pdf, pairs, exp_cfg)
            page_separation_comparison(pdf, pairs, exp_cfg)
            page_da_comparison(pdf, pairs, exp_cfg)
            page_centroid_shift(pdf, pairs, exp_cfg)
            page_heatmap(pdf, pairs, exp_cfg)
            page_summary(pdf, pairs, exp_cfg)

    print(f"\n{'='*60}")
    print(f"Done! → {pdf_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()