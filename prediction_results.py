#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Location Prediction Accuracy v2 — Iteration 7.
Outputs: phase1_prediction_accuracy.pdf (both exps by default)

Metrics: gen_match + cross-entropy surprise + GoF (Matteo, Mar 11).

GoF (Goodness-of-Fit):
  Simplified form of Matteo's whiteboard formula:

    GoF = Σ_{correct} p_gt  /  Σ_{all} p_gt

  where p_gt = P(ground truth location | user's historical distribution).
  Easy users (p_gt≈0.9) contribute more weight → getting them wrong hurts more.

Pages per experiment:
  1. Raw gen_match by config
  2. GoF by config
  3. Per-category gen_match (weekday/weekend or emp/unemp)
  4. Per-category GoF  ← NEW
  5. User scatter: accuracy vs difficulty
  6. Summary table

Usage:
    python prediction_results.py                                # both exps
    python prediction_results.py --exp exp2                     # one exp
    python prediction_results.py --exp exp2 --config 5 6        # specific cfgs
"""
import sys, os, json, argparse, hashlib, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, CONFIG_MATRIX, OUTPUT_DIR,
                    MODEL_REGISTRY, get_model_dirs, config_label, get_config)


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_answers(model_key, exp_name, config_id):
    dirs = get_model_dirs(model_key)
    path = os.path.join(dirs['answers'], f"{exp_name}_cfg{config_id}_answers.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_user_distributions(answers):
    """Build per-user location frequency distribution (Matteo's p-bar)."""
    user_geos = defaultdict(list)
    for a in answers:
        uid, gt = a.get('user'), a.get('gt_geo_id')
        if uid and gt:
            user_geos[uid].append(str(gt))
    return {uid: {g: c / len(geos) for g, c in Counter(geos).items()}
            for uid, geos in user_geos.items()}


# ══════════════════════════════════════════════════════════════════════
# Core Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(answers, geo_hash=False):
    if not answers:
        return {}

    user_dist = build_user_distributions(answers)
    cat_correct, cat_total = defaultdict(int), defaultdict(int)
    surprises, records = [], []
    total_correct, total_valid = 0, 0

    for a in answers:
        gt = a.get('gt_geo_id')
        if gt is None:
            continue
        gt_str = (hashlib.md5(str(gt).encode()).hexdigest()[:8]
                  if geo_hash else str(gt))
        parsed = a.get('parsed_answer', '')
        gen = a.get('generated_text', '')
        cat, uid, gt_raw = a.get('label', -1), a.get('user'), str(gt)

        total_valid += 1
        cat_total[cat] += 1
        matched = gt_str in parsed or gt_str in gen
        if matched:
            total_correct += 1
            cat_correct[cat] += 1

        p_gt = max(user_dist.get(uid, {}).get(gt_raw, 0), 1e-10)
        surprises.append(-math.log2(p_gt))
        records.append({'user': uid, 'cat': cat, 'correct': matched,
                        'p_gt': p_gt, 'surprise': surprises[-1]})

    # GoF
    s_all = sum(r['p_gt'] for r in records)
    s_cor = sum(r['p_gt'] for r in records if r['correct'])
    gof = s_cor / max(s_all, 1e-10)

    gof_cat = {}
    for cat in sorted(cat_total):
        recs = [r for r in records if r['cat'] == cat]
        d = sum(r['p_gt'] for r in recs)
        n = sum(r['p_gt'] for r in recs if r['correct'])
        gof_cat[cat] = n / max(d, 1e-10)

    per_cat = {cat: {'accuracy': cat_correct[cat] / max(cat_total[cat], 1),
                     'n': cat_total[cat], 'correct': cat_correct[cat]}
               for cat in sorted(cat_total)}

    ustats = defaultdict(lambda: {'correct': 0, 'total': 0,
                                   'p_gt_sum': 0., 'surprise_sum': 0.})
    for r in records:
        u = ustats[r['user']]
        u['total'] += 1; u['p_gt_sum'] += r['p_gt']
        u['surprise_sum'] += r['surprise']
        if r['correct']: u['correct'] += 1

    user_level = [
        {'user': uid, 'accuracy': u['correct']/max(u['total'],1),
         'mean_surprise': u['surprise_sum']/max(u['total'],1),
         'mean_p_gt': u['p_gt_sum']/max(u['total'],1), 'n': u['total']}
        for uid, u in ustats.items()]

    return {'gen_match': total_correct/max(total_valid,1), 'valid': total_valid,
            'per_category': per_cat,
            'mean_surprise': float(np.mean(surprises)) if surprises else 0,
            'median_surprise': float(np.median(surprises)) if surprises else 0,
            'gof': gof, 'gof_per_category': gof_cat, 'user_level': user_level}


# ══════════════════════════════════════════════════════════════════════
# PDF Pages
# ══════════════════════════════════════════════════════════════════════

def _bar(ax, all_data, cfgs, mks, exp, key, ylabel, title):
    MC = {'1b': '#2196F3', '4b': '#4CAF50', '12b': '#FF5722'}
    x = np.arange(len(cfgs))
    w = 0.8 / max(len(mks), 1)
    for i, mk in enumerate(mks):
        vals = [all_data.get(mk,{}).get(c,{}).get(key,0) for c in cfgs]
        off = (i - len(mks)/2 + 0.5) * w
        bars = ax.bar(x+off, vals, w, label=mk.upper(), color=MC.get(mk,'gray'))
        for b, v in zip(bars, vals):
            if v > 0.01:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+.015,
                        f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cfg{c}\n{config_label(exp,c)}" for c in cfgs], fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=.15, axis='y'); ax.set_ylim(0, 1.15)


def _cat_bar(ax, all_data, cfgs, mk, exp, cats, cat_names, key_fn, ylabel, title):
    CC = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    x = np.arange(len(cfgs))
    w = 0.8 / max(len(cats), 1)
    for i, cat in enumerate(cats):
        vals = [key_fn(all_data.get(mk,{}).get(c,{}), cat) for c in cfgs]
        off = (i - len(cats)/2 + 0.5) * w
        bars = ax.bar(x+off, vals, w, label=cat_names[cat], color=CC[i%len(CC)])
        for b, v in zip(bars, vals):
            if v > 0.01:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+.015,
                        f'{v:.1%}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cfg{c}\n{config_label(exp,c)}" for c in cfgs], fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=.15, axis='y'); ax.set_ylim(0, 1.15)


def pg_match(pdf, ad, cfgs, mks, exp, ec):
    fig, ax = plt.subplots(figsize=(max(10, len(cfgs)*2), 6))
    _bar(ax, ad, cfgs, mks, exp, 'gen_match', 'Accuracy',
         f'{ec["name"]} — Raw gen_match (unweighted)')
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


def pg_gof(pdf, ad, cfgs, mks, exp, ec):
    fig, ax = plt.subplots(figsize=(max(10, len(cfgs)*2), 6))
    _bar(ax, ad, cfgs, mks, exp, 'gof', 'GoF',
         f'{ec["name"]} — GoF (p_gt-weighted accuracy)\n'
         f'GoF = Σ_correct p_gt / Σ_all p_gt')
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


def pg_cat_acc(pdf, ad, cfgs, mks, exp, ec, cn):
    mk = mks[0]; cats = sorted(cn.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(cfgs)*2), 6))
    _cat_bar(ax, ad, cfgs, mk, exp, cats, cn,
             lambda m, c: m.get('per_category',{}).get(c,{}).get('accuracy',0),
             'Accuracy', f'{ec["name"]} — Per-category gen_match ({mk.upper()})')
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


def pg_cat_gof(pdf, ad, cfgs, mks, exp, ec, cn):
    mk = mks[0]; cats = sorted(cn.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(cfgs)*2), 6))
    _cat_bar(ax, ad, cfgs, mk, exp, cats, cn,
             lambda m, c: m.get('gof_per_category',{}).get(c,0),
             'GoF', f'{ec["name"]} — Per-category GoF ({mk.upper()})\n'
                    f'GoF = Σ_correct p_gt / Σ_all p_gt per category')
    plt.tight_layout(); pdf.savefig(fig, dpi=200); plt.close()


def pg_scatter(pdf, ad, cfgs, mks, exp, ec):
    mk = mks[0]
    valid = [c for c in cfgs if ad.get(mk,{}).get(c,{}).get('user_level')]
    if not valid: return
    n = len(valid); nc = min(n, 4); nr = (n+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr), squeeze=False)
    sc = None
    for idx, cid in enumerate(valid):
        ax = axes[idx//nc, idx%nc]
        ul = ad[mk][cid]['user_level']
        sc = ax.scatter([u['mean_surprise'] for u in ul],
                        [u['accuracy'] for u in ul],
                        c=[u['mean_p_gt'] for u in ul],
                        cmap='RdYlGn', s=20, alpha=.7, edgecolors='k',
                        linewidth=.3, vmin=0, vmax=1)
        ax.set_xlabel('Surprise (bits)', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=8)
        ax.set_title(f'cfg{cid}', fontsize=9, fontweight='bold')
        ax.axhline(.5, ls='--', color='gray', alpha=.3)
        ax.grid(True, alpha=.1); ax.tick_params(labelsize=7)
    for idx in range(n, nr*nc):
        axes[idx//nc, idx%nc].set_visible(False)
    if sc:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label='mean p_gt', shrink=.6)
    fig.suptitle(f'{ec["name"]} — User accuracy vs difficulty  |  {mk.upper()}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0,0,.92,.94])
    pdf.savefig(fig, dpi=200); plt.close()


def pg_summary(pdf, ad, cfgs, mks, exp, ec, cn):
    fig, ax = plt.subplots(figsize=(16, 10)); ax.axis('off')
    mk = mks[0]; cats = sorted(cn.keys())
    ch = ''.join(f' {cn[c]+"-acc":>10s} {cn[c]+"-GoF":>10s}' for c in cats)
    lines = [f'{ec["name"]} — Prediction Summary ({mk.upper()})\n',
             f'GoF = Σ_correct p_gt / Σ_all p_gt  (Matteo, Mar 11)\n\n',
             f'{"cfg":>5s} {"label":>25s} {"match":>7s} {"GoF":>7s} '
             f'{"surp":>7s}{ch}\n',
             '─' * (55 + 20*len(cats)) + '\n']
    for cid in cfgs:
        m = ad.get(mk,{}).get(cid,{})
        if not m: continue
        cv = ''
        for c in cats:
            ca = m.get('per_category',{}).get(c,{}).get('accuracy',0)
            cg = m.get('gof_per_category',{}).get(c,0)
            cv += f' {ca:>10.1%} {cg:>10.1%}'
        lines.append(f'{cid:>5d} {config_label(exp,cid):>25s} '
                     f'{m.get("gen_match",0):>7.1%} {m.get("gof",0):>7.1%} '
                     f'{m.get("mean_surprise",0):>7.2f}{cv}\n')
    lines += ['\n', 'match = raw gen_match | GoF = p_gt-weighted accuracy\n',
              'surp = mean surprise bits | Cat-GoF = GoF per label category\n']
    ax.text(.02, .98, ''.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace')
    fig.suptitle(f'{ec["name"]} — Numerical Summary', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,.96]); pdf.savefig(fig, dpi=200); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', default=['12b'])
    parser.add_argument('--exp', nargs='+', default=['exp1a', 'exp2'])
    parser.add_argument('--config', nargs='*', type=int, default=None)
    args = parser.parse_args()

    CN = {'exp1a': {0: 'Weekend', 1: 'Weekday'},
          'exp2':  {0: 'Unemployed', 1: 'Employed'}}

    out_dir = get_model_dirs(args.model[0])['results']
    pdf_path = os.path.join(out_dir, 'phase1_prediction_accuracy.pdf')

    print(f"{'='*70}")
    print(f"Prediction Accuracy + GoF → {pdf_path}")
    print(f"{'='*70}")

    with PdfPages(pdf_path) as pdf:
        for exp in args.exp:
            if exp not in EXP_CONFIG: continue
            ec = EXP_CONFIG[exp]
            cn = CN.get(exp, {0: 'C0', 1: 'C1'})
            cids = ([c['id'] for c in CONFIG_MATRIX[exp]]
                    if args.config is None
                    else [c for c in args.config
                          if any(cc['id']==c for cc in CONFIG_MATRIX[exp])])

            print(f"\n  {exp} ({ec['name']}): cfgs={cids}")
            ad = {}
            for mk in args.model:
                ad[mk] = {}
                for cid in cids:
                    cfg = get_config(exp, cid)
                    ans = load_answers(mk, exp, cid)
                    if ans is None: continue
                    m = compute_metrics(ans, geo_hash=cfg['geo_hash'])
                    ad[mk][cid] = m
                    cl = config_label(exp, cid)
                    print(f"    {mk} cfg{cid} ({cl}): "
                          f"match={m['gen_match']:.1%} GoF={m['gof']:.1%}")
                    for cat in sorted(m['per_category']):
                        info = m['per_category'][cat]
                        gc = m['gof_per_category'].get(cat, 0)
                        print(f"      {cn.get(cat,cat):>12s}: "
                              f"acc={info['accuracy']:.1%} GoF={gc:.1%} "
                              f"({info['correct']}/{info['n']})")

            fc = sorted(set(c for mk in ad for c in ad[mk]))
            if not fc: continue

            print(f"    Generating pages...")
            pg_match(pdf, ad, fc, args.model, exp, ec)
            pg_gof(pdf, ad, fc, args.model, exp, ec)
            pg_cat_acc(pdf, ad, fc, args.model, exp, ec, cn)
            pg_cat_gof(pdf, ad, fc, args.model, exp, ec, cn)
            pg_scatter(pdf, ad, fc, args.model, exp, ec)
            pg_summary(pdf, ad, fc, args.model, exp, ec, cn)

    # JSON export
    jp = os.path.join(out_dir, 'prediction_metrics.json')
    ser = {}
    for exp in args.exp:
        ser[exp] = {}
        for mk in args.model:
            for cid, m in ad.get(mk,{}).items():
                s = dict(m)
                s['gof_per_category'] = {str(k):v for k,v in s.get('gof_per_category',{}).items()}
                s['per_category'] = {str(k):v for k,v in s.get('per_category',{}).items()}
                ser[exp][f"{mk}_cfg{cid}"] = s
    with open(jp, 'w') as f:
        json.dump(ser, f, indent=2, default=str)
    print(f"\n  JSON: {jp}")
    print(f"\n{'='*70}\nDone! → {pdf_path}\n{'='*70}")


if __name__ == "__main__":
    main()