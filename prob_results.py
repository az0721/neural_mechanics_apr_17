#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Probe Analysis v3 — Iteration 7.

Changes from v2:
  - Val actually selects best C via grid search [0.001, 0.01, 0.1, 1.0, 10.0]
  - Separate PDF per experiment (probe_results_exp1a.pdf / probe_results_exp2.pdf)
  - Cache all probe results to .npz per config → skip recomputation on rerun
  - --force flag to ignore cache and recompute
  - Summary table includes best_C column

Pages (per experiment):
  1. Layer-wise linear probing curves (all configs)
  2. Random label baseline + selectivity (Hewitt & Liang 2019)
  3. MLP vs Linear probe comparison at best-5 layers
  4. Train/Val/Test split with C-tuning + classification report
  5. Selectivity test: cross-concept probe transfer (exp2 only)
  6. Probe weight analysis + cosine similarity to steering vectors
  7. Summary table

Usage:
    python prob_results_v3.py --exp exp1a              # one experiment
    python prob_results_v3.py --exp exp2               # one experiment
    python prob_results_v3.py --exp exp1a exp2         # both (separate PDFs)
    python prob_results_v3.py --exp exp2 --cfg 5 6     # specific configs
    python prob_results_v3.py --exp exp2 --force       # ignore cache
"""
import sys, os, argparse, warnings, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (StratifiedGroupKFold, cross_val_score,
                                     GroupShuffleSplit)
from sklearn.metrics import (balanced_accuracy_score, classification_report)
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXP_CONFIG, CONFIG_MATRIX, get_model_dirs, config_label, OUTPUT_DIR


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════
C_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0]

CFG_STYLES = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--', 7: '-.', 8: ':'}
CFG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
              5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f'}


# ══════════════════════════════════════════════════════════════════════
# Probe Constructors
# ══════════════════════════════════════════════════════════════════════

def make_linear(C=1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, solver='saga', C=C,
                           class_weight='balanced', n_jobs=-1))

def make_mlp():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256,), activation='relu',
                      max_iter=500, early_stopping=True,
                      validation_fraction=0.15, random_state=42))


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def make_groups(meta, exp_name):
    if exp_name == 'exp2':
        return np.array([m['user'] for m in meta])
    return np.array([f"{m['user']}__{m['date']}" for m in meta])


def load_cfg_data(dirs, exp, cfg_id):
    path = os.path.join(dirs['hidden'], f"{exp}_cfg{cfg_id}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data['hidden_states'], data['labels'], data['meta']


def cache_path(dirs, exp, cfg_id):
    d = os.path.join(dirs['results'], 'probe_cache')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{exp}_cfg{cfg_id}_probe.npz")


def save_cache(path, results):
    """Save probe results to .npz for later reuse."""
    save_dict = {
        'means': results['means'],
        'stds': results['stds'],
        'best_layer': np.array(results['best_layer']),
        'clabel': np.array(results['clabel']),
    }
    if 'rand_curves' in results:
        save_dict['rand_curves'] = results['rand_curves']
        save_dict['rand_best_mean'] = np.array(results['rand_best_mean'])
        save_dict['rand_best_std'] = np.array(results['rand_best_std'])
    if 'mlp_layers' in results:
        mlp_l = sorted(results['mlp_layers'].keys())
        save_dict['mlp_layer_ids'] = np.array(mlp_l)
        save_dict['mlp_means'] = np.array([results['mlp_layers'][l][0] for l in mlp_l])
        save_dict['mlp_stds'] = np.array([results['mlp_layers'][l][1] for l in mlp_l])
    if 'tvt' in results and results['tvt'] is not None:
        tvt = results['tvt']
        save_dict['tvt_train_acc'] = np.array(tvt['train_acc'])
        save_dict['tvt_val_acc'] = np.array(tvt['val_acc'])
        save_dict['tvt_test_acc'] = np.array(tvt['test_acc'])
        save_dict['tvt_best_C'] = np.array(tvt['best_C'])
        save_dict['tvt_n_train'] = np.array(tvt['n_train'])
        save_dict['tvt_n_val'] = np.array(tvt['n_val'])
        save_dict['tvt_n_test'] = np.array(tvt['n_test'])
        save_dict['tvt_report_str'] = np.array(tvt['report_str'])
        if tvt.get('W') is not None:
            save_dict['tvt_W'] = tvt['W']
            save_dict['tvt_b'] = np.array(tvt['b'])
    np.savez_compressed(path, **save_dict)


def load_cache(path):
    """Load cached probe results."""
    data = np.load(path, allow_pickle=True)
    r = {
        'means': data['means'],
        'stds': data['stds'],
        'best_layer': int(data['best_layer']),
        'clabel': str(data['clabel']),
    }
    if 'rand_curves' in data:
        r['rand_curves'] = data['rand_curves']
        r['rand_best_mean'] = float(data['rand_best_mean'])
        r['rand_best_std'] = float(data['rand_best_std'])
    if 'mlp_layer_ids' in data:
        mlp_l = data['mlp_layer_ids'].tolist()
        mlp_m = data['mlp_means'].tolist()
        mlp_s = data['mlp_stds'].tolist()
        r['mlp_layers'] = {l: (m, s) for l, m, s in zip(mlp_l, mlp_m, mlp_s)}
    if 'tvt_train_acc' in data:
        r['tvt'] = {
            'train_acc': float(data['tvt_train_acc']),
            'val_acc': float(data['tvt_val_acc']),
            'test_acc': float(data['tvt_test_acc']),
            'best_C': float(data['tvt_best_C']),
            'n_train': int(data['tvt_n_train']),
            'n_val': int(data['tvt_n_val']),
            'n_test': int(data['tvt_n_test']),
            'report_str': str(data['tvt_report_str']),
            'W': data.get('tvt_W', None),
            'b': float(data['tvt_b']) if 'tvt_b' in data else None,
        }
    return r


# ══════════════════════════════════════════════════════════════════════
# Probe Computation
# ══════════════════════════════════════════════════════════════════════

def probe_all_layers(X, y, groups, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    means, stds = [], []
    for layer in tqdm(range(X.shape[1]), desc="  Linear probe", leave=False):
        scores = cross_val_score(
            make_linear(), X[:, layer, :], y, cv=cv, groups=groups,
            scoring='balanced_accuracy', n_jobs=-1)
        means.append(scores.mean())
        stds.append(scores.std())
    return np.array(means), np.array(stds)


def run_random_baseline(X, y, groups, n_splits=5, n_repeats=5):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_curves = []
    for rep in range(n_repeats):
        y_shuf = np.random.RandomState(rep).permutation(y)
        means = []
        for layer in tqdm(range(X.shape[1]),
                          desc=f"  Random rep {rep+1}/{n_repeats}", leave=False):
            scores = cross_val_score(
                make_linear(), X[:, layer, :], y_shuf, cv=cv, groups=groups,
                scoring='balanced_accuracy', n_jobs=-1)
            means.append(scores.mean())
        all_curves.append(means)
    return np.array(all_curves)


def run_mlp_at_layers(X, y, groups, layers, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    results = {}
    for layer in tqdm(layers, desc="  MLP probe", leave=False):
        scores = cross_val_score(
            make_mlp(), X[:, layer, :], y, cv=cv, groups=groups,
            scoring='balanced_accuracy', n_jobs=-1)
        results[layer] = (scores.mean(), scores.std())
    return results


def run_train_val_test(X_layer, y, groups):
    """70/15/15 split with C-tuning on validation set."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(X_layer, y, groups))

    temp_groups = groups[temp_idx]
    temp_y = y[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(gss2.split(X_layer[temp_idx], temp_y, temp_groups))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    # ── C-tuning on validation set ──
    best_C, best_val_acc = 1.0, 0
    c_results = {}
    for C in C_CANDIDATES:
        clf = make_linear(C=C)
        clf.fit(X_layer[train_idx], y[train_idx])
        va = balanced_accuracy_score(y[val_idx], clf.predict(X_layer[val_idx]))
        ta = balanced_accuracy_score(y[train_idx], clf.predict(X_layer[train_idx]))
        c_results[C] = {'train': ta, 'val': va}
        if va > best_val_acc:
            best_val_acc = va
            best_C = C

    # ── Final model with best C ──
    clf = make_linear(C=best_C)
    clf.fit(X_layer[train_idx], y[train_idx])

    train_acc = balanced_accuracy_score(y[train_idx], clf.predict(X_layer[train_idx]))
    val_acc = balanced_accuracy_score(y[val_idx], clf.predict(X_layer[val_idx]))
    test_acc = balanced_accuracy_score(y[test_idx], clf.predict(X_layer[test_idx]))

    report_str = classification_report(y[test_idx],
                                       clf.predict(X_layer[test_idx]), digits=3)

    W = clf.named_steps['logisticregression'].coef_[0]
    b = clf.named_steps['logisticregression'].intercept_[0]

    return {
        'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
        'best_C': best_C, 'c_results': c_results,
        'report_str': report_str,
        'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
        'W': W, 'b': b,
    }


def compute_all(X, y, groups, exp, cfg_id, clabel, dirs, args):
    """Run all probe analyses for one config. Returns result dict."""
    r = {'clabel': clabel}

    # 1. Layer-wise
    print(f"    [1/4] Linear probe (49 layers × 5 folds)...")
    means, stds = probe_all_layers(X, y, groups)
    r['means'] = means
    r['stds'] = stds
    r['best_layer'] = int(means.argmax())
    print(f"      Best: {means.max():.1%} @ L{r['best_layer']}")

    # 2. Random baseline
    print(f"    [2/4] Random baseline ({args.n_rand} repeats × 49 layers)...")
    rand_curves = run_random_baseline(X, y, groups, n_repeats=args.n_rand)
    r['rand_curves'] = rand_curves
    r['rand_best_mean'] = float(rand_curves.max(axis=1).mean())
    r['rand_best_std'] = float(rand_curves.max(axis=1).std())
    print(f"      Random best: {r['rand_best_mean']:.1%} ± {r['rand_best_std']:.1%}")

    # 3. MLP at top layers
    top_layers = np.argsort(means)[-args.n_mlp_layers:][::-1]
    top_layers = sorted(top_layers.tolist())
    print(f"    [3/4] MLP @ layers {top_layers}...")
    r['mlp_layers'] = run_mlp_at_layers(X, y, groups, top_layers)
    for l, (m, s) in r['mlp_layers'].items():
        gap = m - means[l]
        print(f"      L{l}: Linear={means[l]:.1%} MLP={m:.1%} "
              f"({'+'if gap>0 else ''}{gap:.1%})")

    # 4. Train/Val/Test with C-tuning
    bl = r['best_layer']
    print(f"    [4/4] T/V/T with C-tuning @ L{bl}...")
    tvt = run_train_val_test(X[:, bl, :], y, groups)
    r['tvt'] = tvt
    print(f"      Best C={tvt['best_C']}")
    print(f"      Train={tvt['train_acc']:.1%} Val={tvt['val_acc']:.1%} "
          f"Test={tvt['test_acc']:.1%}")
    for C, cr in tvt['c_results'].items():
        marker = ' ✓' if C == tvt['best_C'] else ''
        print(f"        C={C:<6} train={cr['train']:.1%} val={cr['val']:.1%}{marker}")

    return r


# ══════════════════════════════════════════════════════════════════════
# Page 1: Layer-wise Probing Curves
# ══════════════════════════════════════════════════════════════════════

def page_layerwise(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    baseline = 1.0 / exp_cfg['n_classes']

    for cfg_id, r in sorted(all_results.items()):
        means = r['means']
        best_l = int(means.argmax())
        best_a = means.max()
        ax.plot(range(len(means)), means,
                linestyle=CFG_STYLES.get(cfg_id, '-'),
                color=CFG_COLORS.get(cfg_id, '#333'), lw=2,
                label=f"cfg{cfg_id} ({r['clabel']}) — {best_a:.1%} @ L{best_l}")
        ax.scatter(best_l, best_a, s=50,
                   color=CFG_COLORS.get(cfg_id, '#333'), zorder=5)

    ax.axhline(baseline, color='red', ls='--', alpha=0.4,
               label=f'Chance {baseline:.0%}')
    ax.set(xlabel='Layer', ylabel='Balanced Accuracy')
    ax.set_title(f"Page 1: {exp_cfg['name']} — Layer-wise Linear Probe\n"
                 f"{tag} (Iter 7, 15-min)", fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Page 2: Random Label Baseline + Selectivity
# ══════════════════════════════════════════════════════════════════════

def page_random_baseline(pdf, all_results, exp_cfg, tag):
    cfgs = sorted(all_results.keys())
    n = len(cfgs)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    real_accs = [all_results[c]['means'].max() for c in cfgs]
    rand_accs = [all_results[c].get('rand_best_mean', 0.5) for c in cfgs]
    selectivities = [r - b for r, b in zip(real_accs, rand_accs)]

    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, real_accs, w, label='Real', color='#2196F3')
    ax.bar(x, rand_accs, w, label='Random Labels', color='#FF9800')
    ax.bar(x + w, selectivities, w, label='Selectivity', color='#4CAF50')
    ax.set_xticks(x)
    ax.set_xticklabels([f"cfg{c}" for c in cfgs], fontsize=8)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.3)
    ax.set_ylabel('Balanced Accuracy / Selectivity')
    ax.set_title('Best-Layer Accuracy vs Random Baseline')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis='y')
    ax.set_ylim(0, 1.0)

    ax2 = axes[1]
    best_cfg = cfgs[np.argmax(real_accs)]
    r = all_results[best_cfg]
    ax2.plot(r['means'], 'b-', lw=2, label=f'cfg{best_cfg} real')
    if 'rand_curves' in r:
        rm = r['rand_curves'].mean(axis=0)
        rs = r['rand_curves'].std(axis=0)
        ax2.plot(rm, 'r--', lw=1.5, label='Random mean')
        ax2.fill_between(range(len(rm)), rm - rs, rm + rs, color='red', alpha=0.1)
    ax2.axhline(0.5, ls='--', color='gray', alpha=0.3)
    ax2.set(xlabel='Layer', ylabel='Balanced Accuracy')
    ax2.set_title(f'cfg{best_cfg}: Real vs Random Labels')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15)
    ax2.set_ylim(0.3, 1.0)

    fig.suptitle(f"Page 2: {exp_cfg['name']} — Random Label Baseline "
                 f"(Hewitt & Liang 2019)\n"
                 f"Selectivity = Real − Random  |  {tag}",
                 fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Page 3: MLP vs Linear
# ══════════════════════════════════════════════════════════════════════

def page_mlp_comparison(pdf, all_results, exp_cfg, tag):
    cfgs = [c for c in sorted(all_results.keys()) if 'mlp_layers' in all_results[c]]
    if not cfgs:
        return

    n = len(cfgs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), squeeze=False)

    for i, cfg_id in enumerate(cfgs):
        ax = axes[0, i]
        r = all_results[cfg_id]
        layers = sorted(r['mlp_layers'].keys())
        lin = [r['means'][l] for l in layers]
        mlp = [r['mlp_layers'][l][0] for l in layers]

        x = np.arange(len(layers))
        w = 0.35
        ax.bar(x - w/2, lin, w, label='Linear', color='#2196F3')
        ax.bar(x + w/2, mlp, w, label='MLP(256)', color='#FF5722')
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'cfg{cfg_id} ({r["clabel"]})', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, ls='--', color='gray', alpha=0.3)

        for j, (lv, mv) in enumerate(zip(lin, mlp)):
            gap = mv - lv
            ax.text(j, max(lv, mv) + 0.02,
                    f"{'+'if gap>0 else ''}{gap:.1%}",
                    ha='center', fontsize=7, color='green' if gap > 0 else 'red')

    fig.suptitle(f"Page 3: {exp_cfg['name']} — Linear vs MLP Probe\n"
                 f"MLP >> Linear → nonlinearly encoded  |  {tag}",
                 fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Page 4: Train/Val/Test with C-tuning
# ══════════════════════════════════════════════════════════════════════

def page_train_val_test(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.axis('off')

    lines = [f"Page 4: {exp_cfg['name']} — Train/Val/Test (70/15/15) with C-tuning\n"
             f"{tag}  |  Val selects best C from {C_CANDIDATES}\n"
             f"{'='*80}\n"]

    for cfg_id in sorted(all_results.keys()):
        r = all_results[cfg_id]
        tvt = r.get('tvt')
        if tvt is None:
            continue

        lines.append(
            f"\ncfg{cfg_id} ({r['clabel']}) @ L{r['best_layer']}  "
            f"| Best C = {tvt['best_C']}\n"
            f"  Train: {tvt['n_train']} samples → {tvt['train_acc']:.1%}\n"
            f"  Val:   {tvt['n_val']} samples → {tvt['val_acc']:.1%}\n"
            f"  Test:  {tvt['n_test']} samples → {tvt['test_acc']:.1%}\n"
        )

        # Show C selection table
        if 'c_results' in tvt:
            lines.append(f"\n  C Selection:\n")
            lines.append(f"  {'C':>8s}  {'Train':>8s}  {'Val':>8s}\n")
            for C in C_CANDIDATES:
                cr = tvt['c_results'].get(C, {})
                marker = ' ✓' if C == tvt['best_C'] else ''
                lines.append(
                    f"  {C:>8.3f}  {cr.get('train',0):.1%}  "
                    f"{cr.get('val',0):.1%}{marker}\n")

        lines.append(f"\n  Classification Report (Test):\n")
        for line in tvt['report_str'].split('\n'):
            lines.append(f"  {line}\n")
        lines.append(f"  {'─'*60}\n")

    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Page 5: Selectivity Test
# ══════════════════════════════════════════════════════════════════════

def run_selectivity(all_results, dirs, exp):
    """Run cross-concept selectivity (exp2 only)."""
    if exp != 'exp2':
        return []

    selectivity_data = []
    for cfg_id, r in all_results.items():
        if 'X' not in r:
            continue
        meta = r['meta']
        exp1a_path = os.path.join(dirs['hidden'], "exp1a_cfg1.npz")
        if not os.path.exists(exp1a_path):
            continue

        exp1a_data = np.load(exp1a_path, allow_pickle=True)
        user_wd = {}
        for m1, y1 in zip(exp1a_data['meta'], exp1a_data['labels']):
            user_wd.setdefault(m1['user'], []).append(y1)

        control_y, valid_mask = [], []
        for m in meta:
            if m['user'] in user_wd:
                control_y.append(int(np.mean(user_wd[m['user']]) > 0.5))
                valid_mask.append(True)
            else:
                control_y.append(0)
                valid_mask.append(False)

        if sum(valid_mask) < 100:
            continue

        valid_mask = np.array(valid_mask)
        control_y = np.array(control_y)
        bl = r['best_layer']
        X_bl = r['X'][:, bl, :]
        cv = StratifiedGroupKFold(n_splits=5)

        target_scores = cross_val_score(
            make_linear(), X_bl, r['y'], cv=cv, groups=r['groups'],
            scoring='balanced_accuracy', n_jobs=-1)

        n_classes_ctrl = len(set(control_y[valid_mask]))
        if n_classes_ctrl < 2:
            continue
        control_scores = cross_val_score(
            make_linear(), X_bl[valid_mask], control_y[valid_mask],
            cv=StratifiedGroupKFold(n_splits=min(5, n_classes_ctrl)),
            groups=r['groups'][valid_mask],
            scoring='balanced_accuracy', n_jobs=-1)

        selectivity_data.append({
            'name': f'cfg{cfg_id}\nEmployment→Weekday',
            'target_acc': target_scores.mean(),
            'control_acc': control_scores.mean(),
        })
        print(f"    Selectivity cfg{cfg_id}: target={target_scores.mean():.1%} "
              f"control={control_scores.mean():.1%}")

    return selectivity_data


def page_selectivity(pdf, selectivity_data, exp_cfg, tag):
    if not selectivity_data:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = [s['name'] for s in selectivity_data]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, [s['target_acc'] for s in selectivity_data], w,
           label='Target concept', color='#2196F3')
    ax.bar(x + w/2, [s['control_acc'] for s in selectivity_data], w,
           label='Control concept', color='#FF9800')
    ax.axhline(0.5, ls='--', color='gray', alpha=0.3, label='Chance')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Balanced Accuracy'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis='y'); ax.set_ylim(0, 1.0)
    fig.suptitle(f"Page 5: Selectivity — Train on A → test on B → should be ~50%\n"
                 f"{tag}", fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig, dpi=200); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Page 6: Weight Analysis + Cosine Similarity
# ══════════════════════════════════════════════════════════════════════

def page_weight_analysis(pdf, all_results, exp, exp_cfg, tag):
    cfgs = [c for c in sorted(all_results.keys())
            if all_results[c].get('tvt') and all_results[c]['tvt'].get('W') is not None]
    if not cfgs:
        return

    n = len(cfgs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    vectors_dir = os.path.join(OUTPUT_DIR, 'steering', 'vectors')

    for row, cfg_id in enumerate(cfgs):
        r = all_results[cfg_id]
        W = r['tvt']['W']
        bl = r['best_layer']

        # Left: top 20 weight dims
        ax_w = axes[row, 0]
        top_idx = np.argsort(np.abs(W))[-20:][::-1]
        top_vals = W[top_idx]
        colors = ['#2196F3' if v > 0 else '#FF5722' for v in top_vals]
        ax_w.barh(range(20), top_vals, color=colors)
        ax_w.set_yticks(range(20))
        ax_w.set_yticklabels([f"dim {i}" for i in top_idx], fontsize=7)
        ax_w.set_xlabel('Probe Weight')
        ax_w.set_title(f'cfg{cfg_id}: Top 20 Dims @ L{bl} (C={r["tvt"]["best_C"]})',
                       fontsize=9)
        ax_w.grid(True, alpha=0.15, axis='x')
        ax_w.invert_yaxis()

        # Right: cosine similarity
        ax_cos = axes[row, 1]
        vec_path = os.path.join(vectors_dir, f"{exp}_cfg{cfg_id}_vectors.npz")
        if os.path.exists(vec_path):
            sv_data = np.load(vec_path)
            sv = sv_data['vectors'][bl]
            W_n = W / max(np.linalg.norm(W), 1e-10)
            sv_n = sv / max(np.linalg.norm(sv), 1e-10)
            cos_sim = float(np.dot(W_n, sv_n))
            r['cosine_sim'] = cos_sim

            cos_per_l = []
            for l in range(sv_data['vectors'].shape[0]):
                sv_l = sv_data['vectors'][l]
                sv_l_n = sv_l / max(np.linalg.norm(sv_l), 1e-10)
                cos_per_l.append(np.dot(W_n, sv_l_n))
            ax_cos.plot(cos_per_l, 'g-', lw=1.5)
            ax_cos.axvline(bl, color='red', ls='--', alpha=0.5,
                           label=f'Best L{bl}')
            ax_cos.scatter([bl], [cos_sim], s=80, c='red', zorder=5)
            ax_cos.set(xlabel='Layer', ylabel='Cosine Similarity')
            ax_cos.set_title(f'cfg{cfg_id}: cos(W, sv) @ L{bl} = {cos_sim:.4f}',
                             fontsize=9)
            ax_cos.legend(fontsize=8)
            ax_cos.grid(True, alpha=0.15); ax_cos.set_ylim(-1, 1)
        else:
            ax_cos.text(0.5, 0.5, f'No steering vectors\nfor {exp} cfg{cfg_id}',
                        ha='center', va='center', transform=ax_cos.transAxes)

    fig.suptitle(f"Page 6: {exp_cfg['name']} — Probe Weight Analysis\n"
                 f"Left: top dims  |  Right: cos(probe_W, steering_vector)\n{tag}",
                 fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=200); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Page 7: Summary Table
# ══════════════════════════════════════════════════════════════════════

def page_summary(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.axis('off')

    header = (f"{'cfg':>5} {'label':>22} {'bestL':>6} {'bestC':>6} "
              f"{'linear':>8} {'rand':>8} {'select':>8} {'MLP':>8} "
              f"{'tvt_trn':>8} {'tvt_val':>8} {'tvt_tst':>8} {'cos':>8}\n")
    sep = '─' * 115 + '\n'
    rows = [f"Page 7: {exp_cfg['name']} — Summary  |  {tag}\n"
            f"C selected from {C_CANDIDATES} via validation set\n\n",
            header, sep]

    for cfg_id in sorted(all_results.keys()):
        r = all_results[cfg_id]
        bl = r['best_layer']
        lin = r['means'].max()
        rand = r.get('rand_best_mean', float('nan'))
        sel = lin - rand if not np.isnan(rand) else float('nan')
        mlp_best = (max(r['mlp_layers'].values(), key=lambda x: x[0])[0]
                    if r.get('mlp_layers') else float('nan'))
        tvt = r.get('tvt', {})
        bc = tvt.get('best_C', float('nan'))
        ta = tvt.get('train_acc', float('nan'))
        va = tvt.get('val_acc', float('nan'))
        te = tvt.get('test_acc', float('nan'))
        cos = r.get('cosine_sim', float('nan'))

        rows.append(
            f"{cfg_id:>5} {r['clabel']:>22} {bl:>6} {bc:>6.3f} "
            f"{lin:>8.1%} {rand:>8.1%} {sel:>8.1%} {mlp_best:>8.1%} "
            f"{ta:>8.1%} {va:>8.1%} {te:>8.1%} {cos:>8.3f}\n")

    rows.append(sep)
    rows.append(
        "\nlinear  = best balanced_accuracy from StratifiedGroupKFold(5)\n"
        "rand    = random label baseline (mean of best-layer across 5 repeats)\n"
        "select  = linear − rand  (Hewitt & Liang 2019)\n"
        "MLP     = best MLP(256) at same top-5 layers\n"
        "bestC   = C selected by validation set from grid search\n"
        "tvt_trn/val/tst = Train/Val/Test accuracy at best layer with best C\n"
        "cos     = cosine(probe_W, steering_vector) at best layer\n")

    ax.text(0.02, 0.98, ''.join(rows), transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--cfg', nargs='+', type=int, default=None)
    parser.add_argument('--model', default='12b')
    parser.add_argument('--n-rand', type=int, default=5)
    parser.add_argument('--n-mlp-layers', type=int, default=5)
    parser.add_argument('--force', action='store_true',
                        help='Ignore cache, recompute everything')
    args = parser.parse_args()

    dirs = get_model_dirs(args.model)
    tag = f"gemma3_{args.model}b_it"

    for exp in args.exp:
        exp_cfg = EXP_CONFIG[exp]
        configs = CONFIG_MATRIX[exp]
        if args.cfg:
            configs = [c for c in configs if c['id'] in args.cfg]

        pdf_path = os.path.join(dirs['results'], f'probe_results_{exp}.pdf')

        print(f"\n{'='*70}")
        print(f"Probe Analysis v3: {exp} ({exp_cfg['name']})")
        print(f"Output: {pdf_path}")
        print(f"Cache:  {os.path.join(dirs['results'], 'probe_cache/')}")
        print(f"{'='*70}")

        all_results = {}

        # ── Compute or load cache for each config ──
        for cfg in configs:
            cfg_id = cfg['id']
            clabel = config_label(exp, cfg_id)
            cp = cache_path(dirs, exp, cfg_id)

            if os.path.exists(cp) and not args.force:
                print(f"\n  cfg{cfg_id} ({clabel}): loading from cache")
                r = load_cache(cp)
                r['clabel'] = clabel
                # Need to reload X for selectivity test (exp2 only)
                if exp == 'exp2':
                    data = load_cfg_data(dirs, exp, cfg_id)
                    if data:
                        r['X'] = data[0]
                        r['y'] = data[1]
                        r['meta'] = data[2]
                        r['groups'] = make_groups(data[2], exp)
                all_results[cfg_id] = r
                print(f"    Best: {r['means'].max():.1%} @ L{r['best_layer']}")
                continue

            # Load raw data
            data = load_cfg_data(dirs, exp, cfg_id)
            if data is None:
                print(f"\n  cfg{cfg_id}: data not found, skipping")
                continue

            X, y, meta = data
            groups = make_groups(meta, exp)
            print(f"\n  cfg{cfg_id} ({clabel}): N={len(y)}, "
                  f"users={len(set(m['user'] for m in meta))}")

            r = compute_all(X, y, groups, exp, cfg_id, clabel, dirs, args)
            r['X'] = X
            r['y'] = y
            r['meta'] = meta
            r['groups'] = groups

            # Save cache
            save_cache(cp, r)
            print(f"    Cached → {cp}")

            all_results[cfg_id] = r

        if not all_results:
            print("  No data, skipping")
            continue

        # ── Generate PDF ──
        print(f"\n  Generating PDF: {pdf_path}")

        with PdfPages(pdf_path) as pdf:
            print("    Page 1: Layer-wise curves")
            page_layerwise(pdf, all_results, exp_cfg, tag)

            print("    Page 2: Random baseline")
            page_random_baseline(pdf, all_results, exp_cfg, tag)

            print("    Page 3: MLP comparison")
            page_mlp_comparison(pdf, all_results, exp_cfg, tag)

            print("    Page 4: T/V/T with C-tuning")
            page_train_val_test(pdf, all_results, exp_cfg, tag)

            # Page 5: selectivity (exp2 only)
            print("    Page 5: Selectivity")
            sel_data = run_selectivity(all_results, dirs, exp)
            if sel_data:
                page_selectivity(pdf, sel_data, exp_cfg, tag)

            print("    Page 6: Weight analysis")
            page_weight_analysis(pdf, all_results, exp, exp_cfg, tag)

            # Free X before summary
            for cfg_id in all_results:
                for key in ['X', 'y', 'meta', 'groups']:
                    if key in all_results[cfg_id]:
                        del all_results[cfg_id][key]

            print("    Page 7: Summary")
            page_summary(pdf, all_results, exp_cfg, tag)

        print(f"\n  Done! → {pdf_path}")

    print(f"\n{'='*70}")
    print(f"All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()