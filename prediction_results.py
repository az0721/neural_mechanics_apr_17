#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Location Prediction Accuracy — Cross-Iteration Comparison.

GoF from Matteo's April 3 whiteboard (exact formula, no absolute sign):

  GoF = 1 − Σ_U Σ_D Σ_i p̄_i·log(q̄_i)·𝟙(q̄_i ≠ p̄*_i)
            ────────────────────────────────────────────────
            Σ_U Σ_D Σ_i p̄_i·log(q̄_i)

Output: cross_iteration_comparison/predictions_accuracy_v7_v8_comparison.pdf

Usage:
    python prediction_results.py --exp exp2                    # default: v7+v8 comparison
    python prediction_results.py --exp exp2 --iter v7          # single iteration
    python prediction_results.py --exp exp2 --config 5         # single config
"""
import sys, os, json, argparse, hashlib, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, CONFIG_MATRIX, BASE_DIR,
                    MODEL_REGISTRY, config_label, get_config)


# ══════════════════════════════════════════════════════════════════════
# GoF — Exact Whiteboard Formula (Apr 3 meeting)
# ══════════════════════════════════════════════════════════════════════

def compute_gof_whiteboard(sample_records, user_dist, epsilon=1e-10):
    """GoF from Matteo's Apr 3 whiteboard — exact formula with smoothing."""
    total_numer = 0.0
    total_denom = 0.0

    for rec in sample_records:
        uid = rec['user']
        pred_geo = rec.get('pred_geo')
        gt_geo = rec.get('gt_geo')

        if uid not in user_dist or pred_geo is None:
            continue

        p_bar = user_dist[uid]
        n_locs = len(p_bar)
        if n_locs < 2:
            continue

        q_base = epsilon / (n_locs - 1)
        sample_ce = 0.0
        sample_penalty = 0.0

        for loc_i, p_i in p_bar.items():
            q_i = (1.0 - epsilon) if loc_i == pred_geo else q_base
            log_q = math.log(q_i)

            sample_ce += p_i * log_q

            q_onehot = 1 if loc_i == pred_geo else 0
            p_star = 1 if loc_i == gt_geo else 0
            if q_onehot != p_star:
                sample_penalty += p_i * log_q

        total_numer += sample_penalty
        total_denom += sample_ce

    if abs(total_denom) < 1e-20:
        return 1.0
    return 1.0 - (total_numer / total_denom)


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_answers(iter_name, model_key, exp_name, config_id):
    """Load answers JSON for a specific iteration."""
    from config import get_iter_output_dir
    tag = MODEL_REGISTRY[model_key]['tag']
    base = get_iter_output_dir(model_key, iter_name)
    path = os.path.join(base, 'answers', tag,
                        f"{exp_name}_cfg{config_id}_answers.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_user_distributions(answers):
    """Build per-user location frequency distribution from all answers."""
    user_geos = defaultdict(list)
    for a in answers:
        uid = a.get('user')
        gt = a.get('gt_geo_id')
        if uid and gt:
            user_geos[uid].append(str(gt))
    user_dist = {}
    for uid, geos in user_geos.items():
        counts = Counter(geos)
        total = len(geos)
        user_dist[uid] = {g: c / total for g, c in counts.items()}
    return user_dist


# ══════════════════════════════════════════════════════════════════════
# Core Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(answers, geo_hash=False, label_col='label'):
    """Compute gen_match + surprise + GoF for one config."""
    if not answers:
        return {}

    user_dist = build_user_distributions(answers)

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    surprises = []
    sample_records = []
    total_correct = 0
    total_valid = 0

    for a in answers:
        gt = a.get('gt_geo_id')
        if gt is None:
            continue

        gt_str = (hashlib.md5(str(gt).encode()).hexdigest()[:8]
                  if geo_hash else str(gt))
        parsed = a.get('parsed_answer', '')
        gen = a.get('generated_text', '')
        cat = a.get(label_col, -1)
        uid = a.get('user')
        gt_raw = str(gt)

        total_valid += 1
        cat_total[cat] += 1

        matched = gt_str in parsed or gt_str in gen
        if matched:
            total_correct += 1
            cat_correct[cat] += 1

        p_gt = 0.0
        if uid in user_dist and gt_raw in user_dist[uid]:
            p_gt = user_dist[uid][gt_raw]
        if p_gt < 1e-10:
            p_gt = 1e-10
        surprise = -math.log2(p_gt)
        surprises.append(surprise)

        sample_records.append({
            'user': uid, 'cat': cat, 'correct': matched, 'p_gt': p_gt,
            'surprise': surprise,
            'pred_geo': parsed.strip() if parsed else None,
            'gt_geo': gt_raw,
        })

    # GoF whiteboard formula
    gof = compute_gof_whiteboard(sample_records, user_dist)
    gof_per_cat = {}
    for cat in sorted(cat_total.keys()):
        cat_recs = [r for r in sample_records if r['cat'] == cat]
        gof_per_cat[cat] = compute_gof_whiteboard(cat_recs, user_dist)

    results = {
        'gen_match': total_correct / max(total_valid, 1),
        'valid': total_valid,
        'per_category': {},
        'mean_surprise': float(np.mean(surprises)) if surprises else 0,
        'median_surprise': float(np.median(surprises)) if surprises else 0,
        'gof': gof,
        'gof_per_category': gof_per_cat,
    }
    for cat in sorted(cat_total.keys()):
        acc = cat_correct[cat] / max(cat_total[cat], 1)
        results['per_category'][cat] = {
            'accuracy': acc, 'n': cat_total[cat], 'correct': cat_correct[cat],
        }
    return results


# ══════════════════════════════════════════════════════════════════════
# Visualization — Cross-Iteration Comparison PDF
# ══════════════════════════════════════════════════════════════════════

ITER_COLORS = {'v7': '#2196F3', 'v8': '#FF5722'}
CAT_HATCHES = {0: '///', 1: '', 2: '\\\\\\', 3: 'xxx', 4: '...'}


def pg_raw_accuracy(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
    """Page: Raw gen_match — bars grouped by cfg, colored by iter."""
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(found_cfgs)*3), 12))

    # ── Row 1: Overall gen_match by iteration ──
    ax = axes[0]
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

    for i, it in enumerate(iters):
        vals = [data.get(it, {}).get(c, {}).get('gen_match', 0) for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=it.upper(),
                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f'{v:.0%}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.1)
    ax.set_title(f'{exp_cfg["name"]} — Raw gen_match', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

    # ── Row 2: Per-category gen_match — hatched bars per category ──
    ax = axes[1]
    cats = sorted(cat_names.keys())
    n_groups = n_iters * len(cats)
    grp_w = 0.85 / max(n_groups, 1)

    for ci, cat in enumerate(cats):
        for ii, it in enumerate(iters):
            vals = []
            for c in found_cfgs:
                m = data.get(it, {}).get(c, {})
                pc = m.get('per_category', {}).get(cat, {})
                vals.append(pc.get('accuracy', 0))
            idx = ci * n_iters + ii
            offset = (idx - n_groups / 2 + 0.5) * grp_w
            bars = ax.bar(x + offset, vals, grp_w,
                           label=f'{it.upper()} {cat_names[cat]}',
                           color=ITER_COLORS.get(it, 'gray'),
                           hatch=CAT_HATCHES.get(cat, ''),
                           edgecolor='black', lw=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.015,
                            f'{v:.0%}', ha='center', va='bottom', fontsize=5.5)

    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.1)
    ax.set_title(f'{exp_cfg["name"]} — Per-Category gen_match '
                 f'(same color=iteration, hatch=category)', fontweight='bold')
    ax.legend(fontsize=6, ncol=min(n_groups, 4), loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_gof(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
    """Page: GoF — same layout as gen_match."""
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(found_cfgs)*3), 12))

    # ── Row 1: Overall GoF ──
    ax = axes[0]
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

    for i, it in enumerate(iters):
        vals = [data.get(it, {}).get(c, {}).get('gof', 0) for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=it.upper(),
                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('GoF'); ax.set_ylim(0, 1.1)
    ax.set_title(f'{exp_cfg["name"]} — GoF (whiteboard formula, Apr 3)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

    # ── Row 2: Per-category GoF ──
    ax = axes[1]
    cats = sorted(cat_names.keys())
    n_groups = n_iters * len(cats)
    grp_w = 0.85 / max(n_groups, 1)

    for ci, cat in enumerate(cats):
        for ii, it in enumerate(iters):
            vals = []
            for c in found_cfgs:
                m = data.get(it, {}).get(c, {})
                gc = m.get('gof_per_category', {})
                vals.append(gc.get(cat, 0))
            idx = ci * n_iters + ii
            offset = (idx - n_groups / 2 + 0.5) * grp_w
            ax.bar(x + offset, vals, grp_w,
                   label=f'{it.upper()} {cat_names[cat]}',
                   color=ITER_COLORS.get(it, 'gray'),
                   hatch=CAT_HATCHES.get(cat, ''),
                   edgecolor='black', lw=0.5, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('GoF'); ax.set_ylim(0, 1.1)
    ax.set_title(f'{exp_cfg["name"]} — Per-Category GoF '
                 f'(same color=iteration, hatch=category)', fontweight='bold')
    ax.legend(fontsize=6, ncol=min(n_groups, 4), loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_surprise(pdf, data, exp, exp_cfg, found_cfgs, iters):
    """Page: Task difficulty (surprise) comparison."""
    fig, ax = plt.subplots(figsize=(max(12, len(found_cfgs)*3), 6))
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

    for i, it in enumerate(iters):
        means = [data.get(it, {}).get(c, {}).get('mean_surprise', 0) for c in found_cfgs]
        meds = [data.get(it, {}).get(c, {}).get('median_surprise', 0) for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, means, w, label=f'{it.upper()} mean',
                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
        for bar, v, md in zip(bars, means, meds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f'{v:.2f}\n(med {md:.2f})', ha='center', fontsize=6)

    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Surprise (bits)'); ax.set_title(
        f'{exp_cfg["name"]} — Task Difficulty (surprise)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_summary_table(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
    """Page: Summary table with all numbers."""
    fig, ax = plt.subplots(figsize=(18, max(8, len(found_cfgs) * len(iters) * 0.8)))
    ax.axis('off')

    lines = [f'{exp_cfg["name"]} — Full Summary\n{"="*90}\n\n']
    cats = sorted(cat_names.keys())

    # Header
    cat_hdr = ''.join(f'{cat_names[c]:>12s} acc  GoF' for c in cats)
    lines.append(f'{"Iter":>5s} {"Cfg":>5s} {"N":>5s} '
                 f'{"gen_match":>10s} {"GoF":>8s} {"surprise":>9s} {cat_hdr}\n')
    lines.append('─' * 90 + '\n')

    for it in iters:
        for cfg_id in found_cfgs:
            m = data.get(it, {}).get(cfg_id, {})
            if not m:
                continue
            cat_str = ''
            for c in cats:
                pc = m.get('per_category', {}).get(c, {})
                gc = m.get('gof_per_category', {}).get(c, 0)
                cat_str += f'{cat_names[c]:>12s} {pc.get("accuracy",0):>4.0%}  {gc:>4.0%}'
            lines.append(
                f'{it.upper():>5s} cfg{cfg_id:>2d} {m.get("valid",0):>5d} '
                f'{m.get("gen_match",0):>10.1%} {m.get("gof",0):>8.1%} '
                f'{m.get("mean_surprise",0):>9.2f} {cat_str}\n')
        lines.append('\n')

    lines.append(f'\nGoF formula: Matteo whiteboard Apr 3 (exact, no |.|, 1-penalty)\n')

    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, choices=['exp1a', 'exp1b', 'exp2'])
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'])
    parser.add_argument('--model', default='12b')
    parser.add_argument('--config', default='all', help='Config ID or "all"')
    args = parser.parse_args()

    exp = args.exp
    exp_cfg = EXP_CONFIG[exp]
    config_ids = ([c['id'] for c in CONFIG_MATRIX[exp]]
                  if args.config == 'all' else [int(args.config)])
    iters = args.iter

    cat_names = {
        'exp1a': {0: 'Weekend', 1: 'Weekday'},
        'exp1b': {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'},
        'exp2':  {0: 'Unemployed', 1: 'Employed'},
    }[exp]

    if args.model == '12b':
        out_dir = os.path.join(BASE_DIR, 'cross_iteration_comparison')
    else:
        model_base = MODEL_REGISTRY[args.model]['hf_name']
        out_dir = os.path.join(model_base, 'cross_iteration_comparison')
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"Prediction Accuracy + GoF: {exp}")
    print(f"  Iterations: {iters}")
    print(f"  Configs: {config_ids}")
    print(f"{'='*70}")

    # ── Load data for all iterations ──
    data = {}  # {iter_name: {cfg_id: metrics}}
    for it in iters:
        data[it] = {}
        for cfg_id in config_ids:
            cfg = get_config(exp, cfg_id)
            clabel = config_label(exp, cfg_id)
            answers = load_answers(it, args.model, exp, cfg_id)
            if answers is None:
                print(f"  {it} cfg{cfg_id}: not found")
                continue

            metrics = compute_metrics(answers, geo_hash=cfg['geo_hash'])
            data[it][cfg_id] = metrics

            print(f"\n  {it.upper()} cfg{cfg_id} ({clabel}):")
            print(f"    gen_match = {metrics['gen_match']:.1%}  |  "
                  f"GoF = {metrics['gof']:.1%}  "
                  f"(N={metrics['valid']})")

            for cat in sorted(metrics['per_category'].keys()):
                info = metrics['per_category'][cat]
                name = cat_names.get(cat, f'Cat{cat}')
                gof_c = metrics['gof_per_category'].get(cat, 0)
                print(f"    {name:12s}: acc={info['accuracy']:.1%} "
                      f"GoF={gof_c:.1%}  "
                      f"({info['correct']}/{info['n']})")

    # ── Generate comparison PDF ──
    found_cfgs = sorted(set(c for it in data for c in data[it]))
    if not found_cfgs:
        print("No data found."); return

    iter_str = '_'.join(iters)
    pdf_path = os.path.join(out_dir,
                            f'predictions_accuracy_{iter_str}_comparison_{exp}.pdf')
    print(f"\nGenerating: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        pg_raw_accuracy(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)
        pg_gof(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)
        pg_surprise(pdf, data, exp, exp_cfg, found_cfgs, iters)
        pg_summary_table(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)

    print(f"\n{'='*70}")
    print(f"Done! → {pdf_path}")
    print(f"{'='*70}")

    # ── Save JSON ──
    json_out = os.path.join(out_dir, f"prediction_{exp}_{iter_str}_metrics.json")
    serializable = {}
    for it in data:
        serializable[it] = {}
        for cfg_id, m in data[it].items():
            s = dict(m)
            s['gof_per_category'] = {str(k): v
                                     for k, v in s['gof_per_category'].items()}
            s['per_category'] = {str(k): v
                                 for k, v in s['per_category'].items()}
            serializable[it][str(cfg_id)] = s
    with open(json_out, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Metrics JSON: {json_out}")


if __name__ == "__main__":
    main()