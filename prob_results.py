#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Probe Analysis v4.

Updates from v3:
  - MLP runs on ALL layers (was top-5 only → Matteo: must compare full curves)
  - MLP fix: max_iter=500, early_stopping=False (was 500 + early_stopping=True → undertrained)
  - Group by user for BOTH exps (was user__date for exp1a → potential leakage)
  - New page: MLP layer-wise curves (all configs, same format as linear)
  - New page: Combined Linear + MLP per config (overlay on same plot)
  - Cache includes full mlp_means/mlp_stds arrays (49 layers)
  - Outputs to current OUTPUT_DIR (v8)

Pages (per experiment):
  1. Linear probe layer-wise curves (all configs)
  2. MLP probe layer-wise curves (all configs)          ← NEW
  3. Combined Linear + MLP per config                    ← NEW
  4. Random label baseline + selectivity (Hewitt & Liang 2019)
  5. Train/Val/Test split with C-tuning
  6. Selectivity test: cross-concept (exp2 only)
  7. Probe weight analysis + cosine similarity
  8. Summary table (includes MLP best layer + best acc)

Usage:
    python prob_results.py --exp exp1a
    python prob_results.py --exp exp2
    python prob_results.py --exp exp2 --force     # ignore cache
"""
import sys, os, argparse, warnings
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
from config import (EXP_CONFIG, CONFIG_MATRIX, MODEL_REGISTRY, OUTPUT_DIR,
                    get_model_dirs, get_iter_model_dirs, config_label)

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
    """MLP probe — fixed: more iterations, no early stopping."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256,), activation='relu',
                      max_iter=500, early_stopping=False,
                      random_state=42))

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def make_groups(meta, exp_name):
    """Group by user for ALL experiments (prevents identity leakage)."""
    return np.array([m['user'] for m in meta])


def load_cfg_data(dirs, exp, cfg_id):
    path = os.path.join(dirs['hidden'], f"{exp}_cfg{cfg_id}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data['hidden_states'], data['labels'], data['meta']


def cache_path(dirs, exp, cfg_id):
    d = os.path.join(dirs['results'], 'probe_cache_v4')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{exp}_cfg{cfg_id}_probe.npz")


def save_cache(path, r):
    sd = {
        'means': r['means'], 'stds': r['stds'],
        'mlp_means': r['mlp_means'], 'mlp_stds': r['mlp_stds'],
        'best_layer': np.array(r['best_layer']),
        'mlp_best_layer': np.array(r['mlp_best_layer']),
        'clabel': np.array(r['clabel']),
    }
    if 'rand_curves' in r:
        sd['rand_curves'] = r['rand_curves']
        sd['rand_best_mean'] = np.array(r['rand_best_mean'])
        sd['rand_best_std'] = np.array(r['rand_best_std'])
    if 'tvt' in r and r['tvt']:
        tvt = r['tvt']
        for k in ['train_acc', 'val_acc', 'test_acc', 'best_C',
                   'n_train', 'n_val', 'n_test']:
            sd[f'tvt_{k}'] = np.array(tvt[k])
        sd['tvt_report_str'] = np.array(tvt['report_str'])
        if tvt.get('W') is not None:
            sd['tvt_W'] = tvt['W']
            sd['tvt_b'] = np.array(tvt['b'])
    np.savez_compressed(path, **sd)


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    r = {
        'means': d['means'], 'stds': d['stds'],
        'mlp_means': d['mlp_means'], 'mlp_stds': d['mlp_stds'],
        'best_layer': int(d['best_layer']),
        'mlp_best_layer': int(d['mlp_best_layer']),
        'clabel': str(d['clabel']),
    }
    if 'rand_curves' in d:
        r['rand_curves'] = d['rand_curves']
        r['rand_best_mean'] = float(d['rand_best_mean'])
        r['rand_best_std'] = float(d['rand_best_std'])
    if 'tvt_train_acc' in d:
        r['tvt'] = {
            'train_acc': float(d['tvt_train_acc']),
            'val_acc': float(d['tvt_val_acc']),
            'test_acc': float(d['tvt_test_acc']),
            'best_C': float(d['tvt_best_C']),
            'n_train': int(d['tvt_n_train']),
            'n_val': int(d['tvt_n_val']),
            'n_test': int(d['tvt_n_test']),
            'report_str': str(d['tvt_report_str']),
            'W': d.get('tvt_W', None),
            'b': float(d['tvt_b']) if 'tvt_b' in d else None,
        }
    return r

# ══════════════════════════════════════════════════════════════════════
# Probe Computation
# ══════════════════════════════════════════════════════════════════════

def probe_all_layers(X, y, groups, probe_fn, desc, n_splits=5):
    """Per-layer probe with GroupKFold CV. Works for both linear and MLP."""
    cv = StratifiedGroupKFold(n_splits=n_splits)
    means, stds = [], []
    for layer in tqdm(range(X.shape[1]), desc=desc, leave=False):
        scores = cross_val_score(
            probe_fn(), X[:, layer, :], y, cv=cv, groups=groups,
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
                          desc=f"  Random {rep+1}/{n_repeats}", leave=False):
            scores = cross_val_score(
                make_linear(), X[:, layer, :], y_shuf, cv=cv, groups=groups,
                scoring='balanced_accuracy', n_jobs=-1)
            means.append(scores.mean())
        all_curves.append(means)
    return np.array(all_curves)


def run_train_val_test(X_layer, y, groups):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(X_layer, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(gss2.split(X_layer[temp_idx], y[temp_idx],
                                         groups[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    best_C, best_val = 1.0, 0
    c_results = {}
    for C in C_CANDIDATES:
        clf = make_linear(C=C)
        clf.fit(X_layer[train_idx], y[train_idx])
        va = balanced_accuracy_score(y[val_idx], clf.predict(X_layer[val_idx]))
        ta = balanced_accuracy_score(y[train_idx], clf.predict(X_layer[train_idx]))
        c_results[C] = {'train': ta, 'val': va}
        if va > best_val:
            best_val = va
            best_C = C

    clf = make_linear(C=best_C)
    clf.fit(X_layer[train_idx], y[train_idx])
    return {
        'train_acc': balanced_accuracy_score(y[train_idx], clf.predict(X_layer[train_idx])),
        'val_acc': balanced_accuracy_score(y[val_idx], clf.predict(X_layer[val_idx])),
        'test_acc': balanced_accuracy_score(y[test_idx], clf.predict(X_layer[test_idx])),
        'best_C': best_C, 'c_results': c_results,
        'report_str': classification_report(y[test_idx],
                                            clf.predict(X_layer[test_idx]), digits=3),
        'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
        'W': clf.named_steps['logisticregression'].coef_[0],
        'b': clf.named_steps['logisticregression'].intercept_[0],
    }


def compute_all(X, y, groups, clabel, args):
    r = {'clabel': clabel}

    # 1. Linear probe — all 49 layers
    print(f"    [1/5] Linear probe (all layers)...")
    r['means'], r['stds'] = probe_all_layers(X, y, groups, make_linear,
                                              "  Linear")
    r['best_layer'] = int(r['means'].argmax())
    print(f"      Best: {r['means'].max():.1%} @ L{r['best_layer']}")

    # 2. MLP probe — all 49 layers (FIXED: full sweep, more iterations)
    print(f"    [2/5] MLP probe (all layers, max_iter=500, no early_stop)...")
    r['mlp_means'], r['mlp_stds'] = probe_all_layers(X, y, groups, make_mlp,
                                                       "  MLP")
    r['mlp_best_layer'] = int(r['mlp_means'].argmax())
    print(f"      Best: {r['mlp_means'].max():.1%} @ L{r['mlp_best_layer']}")

    # 3. Random baseline
    print(f"    [3/5] Random baseline ({args.n_rand} repeats)...")
    rc = run_random_baseline(X, y, groups, n_repeats=args.n_rand)
    r['rand_curves'] = rc
    r['rand_best_mean'] = float(rc.max(axis=1).mean())
    r['rand_best_std'] = float(rc.max(axis=1).std())
    print(f"      Random best: {r['rand_best_mean']:.1%} ± {r['rand_best_std']:.1%}")

    # 4. T/V/T with C-tuning at linear best layer
    bl = r['best_layer']
    print(f"    [4/5] T/V/T @ L{bl}...")
    r['tvt'] = run_train_val_test(X[:, bl, :], y, groups)
    tvt = r['tvt']
    print(f"      C={tvt['best_C']} → Train={tvt['train_acc']:.1%} "
          f"Val={tvt['val_acc']:.1%} Test={tvt['test_acc']:.1%}")

    # 5. Print comparison at key layers
    print(f"    [5/5] Linear vs MLP comparison:")
    for l in sorted(set([r['best_layer'], r['mlp_best_layer']])):
        gap = r['mlp_means'][l] - r['means'][l]
        print(f"      L{l}: Linear={r['means'][l]:.1%} "
              f"MLP={r['mlp_means'][l]:.1%} ({'+' if gap>0 else ''}{gap:.1%})")

    return r


# ══════════════════════════════════════════════════════════════════════
# Page 1: Linear Layer-wise Curves
# ══════════════════════════════════════════════════════════════════════

def page_linear_curves(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(figsize=(14, 7))
    bl = 1.0 / exp_cfg['n_classes']
    for cid, r in sorted(all_results.items()):
        m = r['means']; bl_l = int(m.argmax())
        ax.plot(range(len(m)), m, ls=CFG_STYLES.get(cid, '-'),
                color=CFG_COLORS.get(cid, '#333'), lw=2,
                label=f"cfg{cid} ({r['clabel']}) — {m.max():.1%} @ L{bl_l}")
        ax.scatter(bl_l, m.max(), s=50, color=CFG_COLORS.get(cid, '#333'), zorder=5)
    ax.axhline(bl, color='red', ls='--', alpha=0.4, label=f'Chance {bl:.0%}')
    ax.set(xlabel='Layer', ylabel='Balanced Accuracy')
    ax.set_title(f'{exp_cfg["name"]} — Linear Probe (all layers)\n{tag}',
                 fontweight='bold')
    ax.set_ylim(0.4, 0.9); ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.15)
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 2: MLP Layer-wise Curves (NEW)
# ══════════════════════════════════════════════════════════════════════

def page_mlp_curves(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(figsize=(14, 7))
    bl = 1.0 / exp_cfg['n_classes']
    for cid, r in sorted(all_results.items()):
        m = r['mlp_means']; bl_l = int(m.argmax())
        ax.plot(range(len(m)), m, ls=CFG_STYLES.get(cid, '-'),
                color=CFG_COLORS.get(cid, '#333'), lw=2,
                label=f"cfg{cid} ({r['clabel']}) — {m.max():.1%} @ L{bl_l}")
        ax.scatter(bl_l, m.max(), s=50, color=CFG_COLORS.get(cid, '#333'), zorder=5)
    ax.axhline(bl, color='red', ls='--', alpha=0.4, label=f'Chance {bl:.0%}')
    ax.set(xlabel='Layer', ylabel='Balanced Accuracy')
    ax.set_title(f'{exp_cfg["name"]} — MLP(256) Probe (all layers)\n'
                 f'{tag} | max_iter=500, no early_stopping', fontweight='bold')
    ax.set_ylim(0.4, 0.9); ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.15)
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 3: Combined Linear + MLP per config (NEW)
# ══════════════════════════════════════════════════════════════════════

def page_combined(pdf, all_results, exp_cfg, tag):
    """One subplot per config: linear (solid) vs MLP (dashed) overlay."""
    cfgs = sorted(all_results.keys())
    n = len(cfgs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              squeeze=False)
    bl = 1.0 / exp_cfg['n_classes']

    for idx, cid in enumerate(cfgs):
        r = all_results[cid]
        ax = axes[idx // ncols, idx % ncols]
        lm, mm = r['means'], r['mlp_means']
        ll, ml = int(lm.argmax()), int(mm.argmax())
        layers = range(len(lm))

        ax.plot(layers, lm, 'b-', lw=2, label=f'Linear {lm.max():.1%}@L{ll}')
        ax.plot(layers, mm, 'r--', lw=2, label=f'MLP {mm.max():.1%}@L{ml}')
        ax.fill_between(layers, lm, mm, alpha=0.08,
                        color='green' if mm.max() >= lm.max() else 'red')
        ax.axhline(bl, color='gray', ls='--', alpha=0.3)
        ax.set_title(f'cfg{cid} ({r["clabel"]})', fontsize=9, fontweight='bold')
        ax.set_ylim(0.4, 0.9); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15); ax.tick_params(labelsize=7)
        if idx // ncols == nrows - 1: ax.set_xlabel('Layer', fontsize=8)
        if idx % ncols == 0: ax.set_ylabel('Balanced Accuracy', fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{exp_cfg["name"]} — Linear vs MLP Per Config\n'
                 f'{tag} | Green fill = MLP ≥ Linear, Red fill = MLP < Linear',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 4: Random Baseline + Selectivity
# ══════════════════════════════════════════════════════════════════════

def page_random(pdf, all_results, exp_cfg, tag):
    cfgs = sorted(all_results.keys())
    n = len(cfgs)
    if n == 0: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    real = [all_results[c]['means'].max() for c in cfgs]
    rand = [all_results[c].get('rand_best_mean', 0.5) for c in cfgs]
    sel = [r - b for r, b in zip(real, rand)]
    x = np.arange(n); w = 0.25
    ax.bar(x-w, real, w, label='Real', color='#2196F3')
    ax.bar(x, rand, w, label='Random', color='#FF9800')
    ax.bar(x+w, sel, w, label='Selectivity', color='#4CAF50')
    ax.set_xticks(x); ax.set_xticklabels([f"cfg{c}" for c in cfgs], fontsize=8)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.3)
    ax.set_ylabel('Accuracy / Selectivity')
    ax.set_title('Best-Layer: Real vs Random'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis='y'); ax.set_ylim(0, 1.0)

    ax2 = axes[1]
    bc = cfgs[np.argmax(real)]
    r = all_results[bc]
    ax2.plot(r['means'], 'b-', lw=2, label=f'cfg{bc} real')
    if 'rand_curves' in r:
        rm, rs = r['rand_curves'].mean(0), r['rand_curves'].std(0)
        ax2.plot(rm, 'r--', lw=1.5, label='Random mean')
        ax2.fill_between(range(len(rm)), rm-rs, rm+rs, color='red', alpha=0.1)
    ax2.axhline(0.5, ls='--', color='gray', alpha=0.3)
    ax2.set(xlabel='Layer', ylabel='Balanced Accuracy')
    ax2.set_title(f'cfg{bc}: Real vs Random'); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15); ax2.set_ylim(0.3, 1.0)

    fig.suptitle(f'{exp_cfg["name"]} — Random Label Baseline (Hewitt & Liang 2019)\n'
                 f'Selectivity = Real − Random | {tag}', fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 5: Train/Val/Test with C-tuning
# ══════════════════════════════════════════════════════════════════════

def page_tvt(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(figsize=(14, 10)); ax.axis('off')
    lines = [f'{exp_cfg["name"]} — Train/Val/Test (70/15/15) + C-tuning\n'
             f'{tag} | C from {C_CANDIDATES}\n{"="*80}\n']
    for cid in sorted(all_results.keys()):
        r = all_results[cid]
        tvt = r.get('tvt')
        if not tvt: continue
        lines.append(
            f'\ncfg{cid} ({r["clabel"]}) @ L{r["best_layer"]} | Best C={tvt["best_C"]}\n'
            f'  Train: {tvt["n_train"]} → {tvt["train_acc"]:.1%}\n'
            f'  Val:   {tvt["n_val"]} → {tvt["val_acc"]:.1%}\n'
            f'  Test:  {tvt["n_test"]} → {tvt["test_acc"]:.1%}\n')
        if 'c_results' in tvt:
            lines.append(f'  C Selection:\n  {"C":>8s} {"Train":>8s} {"Val":>8s}\n')
            for C in C_CANDIDATES:
                cr = tvt['c_results'].get(C, {})
                mk = ' ✓' if C == tvt['best_C'] else ''
                lines.append(f'  {C:>8.3f} {cr.get("train",0):.1%} '
                             f'{cr.get("val",0):.1%}{mk}\n')
        lines.append(f'\n  Classification Report (Test):\n')
        for l in tvt['report_str'].split('\n'): lines.append(f'  {l}\n')
        lines.append(f'  {"─"*60}\n')
    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace')
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 6: Selectivity (exp2 only)
# ══════════════════════════════════════════════════════════════════════

def run_selectivity(all_results, dirs, exp):
    if exp != 'exp2': return []
    sel = []
    p = os.path.join(dirs['hidden'], "exp1a_cfg1.npz")
    if not os.path.exists(p): return []
    e1 = np.load(p, allow_pickle=True)
    uwd = {}
    for m, y in zip(e1['meta'], e1['labels']):
        uwd.setdefault(m['user'], []).append(y)

    for cid, r in all_results.items():
        if 'X' not in r: continue
        meta = r['meta']
        cy, vm = [], []
        for m in meta:
            if m['user'] in uwd:
                cy.append(int(np.mean(uwd[m['user']]) > 0.5)); vm.append(True)
            else:
                cy.append(0); vm.append(False)
        if sum(vm) < 100: continue
        vm, cy = np.array(vm), np.array(cy)
        bl = r['best_layer']; Xb = r['X'][:, bl, :]
        cv = StratifiedGroupKFold(n_splits=5)
        ts = cross_val_score(make_linear(), Xb, r['y'], cv=cv,
                             groups=r['groups'], scoring='balanced_accuracy', n_jobs=-1)
        nc = len(set(cy[vm]))
        if nc < 2: continue
        cs = cross_val_score(make_linear(), Xb[vm], cy[vm],
                             cv=StratifiedGroupKFold(n_splits=min(5, nc)),
                             groups=r['groups'][vm], scoring='balanced_accuracy', n_jobs=-1)
        sel.append({'name': f'cfg{cid}', 'target': ts.mean(), 'control': cs.mean()})
        print(f"    Selectivity cfg{cid}: target={ts.mean():.1%} control={cs.mean():.1%}")
    return sel


def page_selectivity(pdf, sel_data, exp_cfg, tag):
    if not sel_data: return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sel_data)); w = 0.35
    ax.bar(x-w/2, [s['target'] for s in sel_data], w, label='Employment', color='#2196F3')
    ax.bar(x+w/2, [s['control'] for s in sel_data], w, label='Weekday(control)', color='#FF9800')
    ax.axhline(0.5, ls='--', color='gray', alpha=0.3, label='Chance')
    ax.set_xticks(x); ax.set_xticklabels([s['name'] for s in sel_data], fontsize=8)
    ax.set_ylabel('Balanced Accuracy'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis='y'); ax.set_ylim(0, 1.0)
    fig.suptitle(f'{exp_cfg["name"]} — Selectivity\n'
                 f'Probe on concept A → test on concept B → should be ~50%\n{tag}',
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 7: Weight Analysis + Cosine Similarity
# ══════════════════════════════════════════════════════════════════════

def page_weights(pdf, all_results, exp, exp_cfg, tag, dirs):
    cfgs = [c for c in sorted(all_results.keys())
            if all_results[c].get('tvt') and
            all_results[c]['tvt'].get('W') is not None]
    if not cfgs: return
    n = len(cfgs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4*n), squeeze=False)
    iter_out = os.path.dirname(os.path.dirname(dirs['results']))
    vdir = os.path.join(iter_out, 'steering', 'vectors')

    for row, cid in enumerate(cfgs):
        r = all_results[cid]; W = r['tvt']['W']; bl = r['best_layer']
        # Left: top 20 dims
        ax = axes[row, 0]
        ti = np.argsort(np.abs(W))[-20:][::-1]
        tv = W[ti]
        ax.barh(range(20), tv, color=['#2196F3' if v>0 else '#FF5722' for v in tv])
        ax.set_yticks(range(20)); ax.set_yticklabels([f"d{i}" for i in ti], fontsize=7)
        ax.set_xlabel('Weight'); ax.set_title(f'cfg{cid}: Top 20 dims @L{bl}', fontsize=9)
        ax.grid(True, alpha=0.15, axis='x'); ax.invert_yaxis()

        # Right: cosine sim
        ax = axes[row, 1]
        vp = os.path.join(vdir, f"{exp}_cfg{cid}_vectors.npz")
        if os.path.exists(vp):
            sv = np.load(vp)['vectors']
            Wn = W / max(np.linalg.norm(W), 1e-10)
            cpl = [np.dot(Wn, sv[l]/max(np.linalg.norm(sv[l]), 1e-10))
                   for l in range(sv.shape[0])]
            cs = cpl[bl]
            r['cosine_sim'] = cs
            ax.plot(cpl, 'g-', lw=1.5)
            ax.axvline(bl, color='red', ls='--', alpha=0.5, label=f'L{bl}')
            ax.scatter([bl], [cs], s=80, c='red', zorder=5)
            ax.set_title(f'cfg{cid}: cos(W,sv)@L{bl}={cs:.4f}', fontsize=9)
            ax.legend(fontsize=8); ax.set_ylim(-1, 1)
        else:
            ax.text(0.5, 0.5, 'No vectors', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set(xlabel='Layer', ylabel='Cosine Sim')
        ax.grid(True, alpha=0.15)

    fig.suptitle(f'{exp_cfg["name"]} — Probe Weights + Cosine Sim\n{tag}',
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Page 8: Summary Table
# ══════════════════════════════════════════════════════════════════════

def page_summary(pdf, all_results, exp_cfg, tag):
    fig, ax = plt.subplots(figsize=(18, 8)); ax.axis('off')
    h = (f"{'cfg':>5} {'label':>22} {'linL':>5} {'lin%':>7} "
         f"{'mlpL':>5} {'mlp%':>7} {'Δ':>6} {'rand':>7} {'sel':>7} "
         f"{'C':>6} {'trn':>7} {'val':>7} {'tst':>7} {'cos':>7}\n")
    sep = '─' * 120 + '\n'
    rows = [f'{exp_cfg["name"]} — Summary | {tag}\n'
            f'MLP: hidden=(256,), max_iter=500, no early_stopping\n'
            f'Linear: saga, max_iter=500, C from {C_CANDIDATES} via val\n\n', h, sep]

    for cid in sorted(all_results.keys()):
        r = all_results[cid]
        ll, la = r['best_layer'], r['means'].max()
        ml, ma = r['mlp_best_layer'], r['mlp_means'].max()
        delta = ma - la
        rand = r.get('rand_best_mean', float('nan'))
        sel = la - rand if not np.isnan(rand) else float('nan')
        tvt = r.get('tvt', {})
        bc = tvt.get('best_C', float('nan'))
        ta = tvt.get('train_acc', float('nan'))
        va = tvt.get('val_acc', float('nan'))
        te = tvt.get('test_acc', float('nan'))
        cs = r.get('cosine_sim', float('nan'))

        rows.append(
            f"{cid:>5} {r['clabel']:>22} L{ll:>3} {la:>7.1%} "
            f"L{ml:>3} {ma:>7.1%} {'+' if delta>=0 else ''}{delta:>5.1%} "
            f"{rand:>7.1%} {sel:>7.1%} {bc:>6.3f} "
            f"{ta:>7.1%} {va:>7.1%} {te:>7.1%} {cs:>7.3f}\n")

    rows.append(sep)
    rows.append('\nlinL/lin% = linear probe best layer/accuracy\n'
                'mlpL/mlp% = MLP probe best layer/accuracy\n'
                'Δ = MLP − Linear (positive = MLP better, Matteo: should be ≥0)\n'
                'rand = random label baseline | sel = selectivity\n'
                'C = best regularization from val | trn/val/tst = T/V/T accuracy\n'
                'cos = cosine(probe_W, steering_vector) at linear best layer\n')

    ax.text(0.02, 0.98, ''.join(rows), transform=ax.transAxes,
            fontsize=7.5, va='top', fontfamily='monospace')
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--cfg', nargs='+', type=int, default=None)
    parser.add_argument('--model', default='12b')
    parser.add_argument('--iter', default='v7', choices=['v7', 'v8'],
                        help='Iteration: v7 (no hint) or v8 (with hint)')
    parser.add_argument('--n-rand', type=int, default=5)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    dirs = get_iter_model_dirs(args.model, args.iter)
    tag = MODEL_REGISTRY[args.model]['tag'] + f' (Iter {args.iter.upper()})'

    for exp in args.exp:
        ec = EXP_CONFIG[exp]
        cfgs = CONFIG_MATRIX[exp]
        if args.cfg:
            cfgs = [c for c in cfgs if c['id'] in args.cfg]

        pdf_path = os.path.join(dirs['results'], f'probe_results_{exp}.pdf')
        print(f"\n{'='*70}")
        print(f"Probe v4: {exp} ({ec['name']})")
        print(f"PDF: {pdf_path}")
        print(f"Cache: {os.path.join(dirs['results'], 'probe_cache_v4/')}")
        print(f"{'='*70}")

        all_results = {}
        for cfg in cfgs:
            cid = cfg['id']; cl = config_label(exp, cid)
            cp = cache_path(dirs, exp, cid)

            if os.path.exists(cp) and not args.force:
                print(f"\n  cfg{cid} ({cl}): cache hit")
                r = load_cache(cp); r['clabel'] = cl
                if exp == 'exp2':
                    d = load_cfg_data(dirs, exp, cid)
                    if d:
                        r['X'], r['y'], r['meta'] = d
                        r['groups'] = make_groups(d[2], exp)
                all_results[cid] = r
                print(f"    Linear: {r['means'].max():.1%}@L{r['best_layer']} | "
                      f"MLP: {r['mlp_means'].max():.1%}@L{r['mlp_best_layer']}")
                continue

            d = load_cfg_data(dirs, exp, cid)
            if d is None:
                print(f"\n  cfg{cid}: not found"); continue
            X, y, meta = d
            groups = make_groups(meta, exp)
            print(f"\n  cfg{cid} ({cl}): N={len(y)}, "
                  f"users={len(set(m['user'] for m in meta))}")

            r = compute_all(X, y, groups, cl, args)
            r['X'], r['y'], r['meta'], r['groups'] = X, y, meta, groups
            save_cache(cp, r)
            print(f"    Cached → {cp}")
            all_results[cid] = r

        if not all_results:
            print("  No data"); continue

        print(f"\n  Generating {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            print("    P1: Linear curves")
            page_linear_curves(pdf, all_results, ec, tag)

            print("    P2: MLP curves")
            page_mlp_curves(pdf, all_results, ec, tag)

            print("    P3: Combined Linear+MLP per config")
            page_combined(pdf, all_results, ec, tag)

            print("    P4: Random baseline")
            page_random(pdf, all_results, ec, tag)

            print("    P5: T/V/T")
            page_tvt(pdf, all_results, ec, tag)

            print("    P6: Selectivity")
            sd = run_selectivity(all_results, dirs, exp)
            if sd: page_selectivity(pdf, sd, ec, tag)

            print("    P7: Weights + cosine")
            page_weights(pdf, all_results, exp, ec, tag, dirs)

            for cid in all_results:
                for k in ['X', 'y', 'meta', 'groups']:
                    all_results[cid].pop(k, None)

            print("    P8: Summary")
            page_summary(pdf, all_results, ec, tag)

        print(f"\n  Done! → {pdf_path}")
    print(f"\n{'='*70}\nAll done!\n{'='*70}")


if __name__ == "__main__":
    main()
