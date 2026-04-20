#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze NL-based steering results v5.1 (Apr 18).

Per question: ONE page with 2×2 layout:
  top-left:     Old method  E[value] heatmap   (layer × coeff)
  top-right:    New method  E[value] heatmap   (layer × 3, neutral center)
  bottom-left:  Old method  pine tree          (ΔE/E_neutral per layer)
  bottom-right: New method  pine tree          (ΔE/E_neutral per layer)

Summaries:
  S1  Best layers (combined v7+v8)
  S2  Cross-question consistency (SEPARATE page per iter)

Fixed scales for cross-prompt comparison:
  pct heatmap:  colorbar [0, 1]  (fraction, 2% bins = 0.02)
  time heatmap: colorbar [5, 12] (hours, displayed as H:MM)
  pine tree X:  [-1, 1], tick spacing 0.005

Usage:
    python steering/analyze_steering_nl.py --model gemma4_31b
"""
import json, os, sys, glob, argparse, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import config_label, get_iter_output_dir, MODEL_REGISTRY
from steering.prompts_nl import ALL_QUESTIONS

# ── Condition display labels ──
COND_LABELS = {
    'exp2':  {'pos': '+V_emp',     'neg': '+V_unemp',
              'pos_short': 'emp',  'neg_short': 'unemp'},
    'exp1a': {'pos': '+V_weekday', 'neg': '+V_weekend',
              'pos_short': 'wkday','neg_short': 'wkend'},
}
NEW_COLS = ['unemployed', 'neutral', 'employed']


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_all(model_key):
    print(f"Loading NL steering results for {model_key}...")
    data = {}
    for it in ['v7', 'v8']:
        out_base = get_iter_output_dir(model_key, it)
        nl_dir = os.path.join(out_base, 'steering', 'nl_results')
        if not os.path.exists(nl_dir):
            data[it] = {}; continue
        d = {}
        for f in sorted(glob.glob(os.path.join(nl_dir, '*_nl.json'))):
            with open(f) as fh:
                j = json.load(fh)
            m = j['meta']
            d[(m['exp'], m['cfg'])] = j
            print(f"  {it}/{m['exp']}/cfg{m['cfg']}: "
                  f"{len(j['results'])} questions")
        data[it] = d
    return data


def build_q_lookup():
    lookup = {}
    for exp, qs in ALL_QUESTIONS.items():
        for q in qs:
            lookup[q['id']] = q
    return lookup


# ══════════════════════════════════════════════════════════════════════
# E[value] — uses FRACTION for pct, HOURS for time
# ══════════════════════════════════════════════════════════════════════

def get_value_array(q_full):
    """Return numeric values for each fg_code.
    pct  → [0.00, 0.02, 0.04, …, 1.00]  (fraction 0-1)
    time → [5.000, 5.083, 5.167, …, 11.917]  (hours)
    """
    if q_full['answer_format'] == 'pct':
        return np.array([i / 100.0 for i in range(0, 102, 2)])
    elif q_full['answer_format'] == 'time':
        return np.array([h + m / 60.0 for h in range(5, 12)
                         for m in range(0, 60, 5)])
    raise ValueError(q_full['answer_format'])


def compute_ev(probs_dict, q_full):
    """E[value] = Σ P(code_i) × value_i"""
    codes = q_full['fg_codes']
    vals = get_value_array(q_full)
    return sum(probs_dict.get(c, 0.0) * v for c, v in zip(codes, vals))


def ev_unit(q_full):
    if q_full['answer_format'] == 'time':
        return 'E[wake-up time] (hours)'
    return 'E[probability] (0-1 fraction)'


# ══════════════════════════════════════════════════════════════════════
# Fixed colorbar scales
# ══════════════════════════════════════════════════════════════════════

def get_heatmap_norm(q_full):
    """Fixed Normalize: pct=[0,1], time=[5,12]."""
    if q_full['answer_format'] == 'pct':
        return Normalize(vmin=0.0, vmax=1.0)
    return Normalize(vmin=5.0, vmax=12.0)


def get_cb_formatter(q_full):
    """Colorbar tick formatter."""
    if q_full['answer_format'] == 'time':
        def _fmt(x, pos):
            h = int(x); m = int(round((x - h) * 60))
            return f"{h}:{m:02d}"
        return mticker.FuncFormatter(_fmt)
    return mticker.FormatStrFormatter('%.3f')


def get_cb_locator(q_full):
    """Colorbar major/minor tick locators."""
    if q_full['answer_format'] == 'pct':
        return mticker.MultipleLocator(0.10), mticker.MultipleLocator(0.02)
    # time: major every 1h, minor every 5min = 1/12 h
    return mticker.MultipleLocator(1.0), mticker.MultipleLocator(1/12)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def get_old_probs(qr, layer, coeff):
    for e in qr['old_method']:
        if e['layer'] == layer and e['coeff'] == coeff:
            return e['probs']
    return {}

def get_new_probs(qr, layer, cond):
    for e in qr['new_method']:
        if e['layer'] == layer:
            return e.get(cond, {})
    return {}

def add_prompt_box(fig, prompt_str):
    wrapped = textwrap.fill(prompt_str.replace('\n', ' '), width=68)
    fig.text(0.01, 0.99, wrapped, transform=fig.transFigure,
             fontsize=4, va='top', ha='left', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5',
                       ec='#cccccc', alpha=0.9))

def dual_y(ax, n_layers, step=5):
    ticks = list(range(0, n_layers, step))
    labels = [f'L{t}' for t in ticks]
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=5)
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(ticks); ax2.set_yticklabels(labels, fontsize=5)

def dual_y_pine(ax, n_layers, step=2):
    ticks = list(range(0, n_layers, step))
    labels = [f'L{t}' for t in ticks]
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=5)
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(ticks); ax2.set_yticklabels(labels, fontsize=5)

def cond_display(exp, cond):
    cl = COND_LABELS.get(exp, COND_LABELS['exp2'])
    if cond == 'neutral':    return 'neutral'
    if cond == 'employed':   return cl['pos']
    if cond == 'unemployed': return cl['neg']
    return cond


# ══════════════════════════════════════════════════════════════════════
# Per-Question Page: 2×2 layout
# ══════════════════════════════════════════════════════════════════════

def pg_question(pdf, qr, q_full, meta, exp, top_layers):
    """One page: 2 heatmaps (top) + 2 pine trees (bottom)."""
    it, cfg   = meta['iter'], meta['cfg']
    n_layers  = meta['n_decoder_layers']
    coeffs    = meta['old_coeffs']
    nc        = len(coeffs)
    cl        = COND_LABELS[exp]

    fig = plt.figure(figsize=(28, 36))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.22,
                          left=0.05, right=0.95, top=0.93, bottom=0.03)

    norm = get_heatmap_norm(q_full)
    cb_fmt = get_cb_formatter(q_full)
    cb_major, cb_minor = get_cb_locator(q_full)

    # ────────────────────────────────────────
    # (0,0) Old E[value] Heatmap
    # ────────────────────────────────────────
    ax_ho = fig.add_subplot(gs[0, 0])

    mat_old = np.zeros((n_layers, nc))
    for li in range(n_layers):
        for ci, c in enumerate(coeffs):
            mat_old[li, ci] = compute_ev(get_old_probs(qr, li, c), q_full)

    im0 = ax_ho.imshow(mat_old, aspect='auto', origin='lower',
                        cmap='coolwarm', norm=norm, interpolation='nearest')
    ax_ho.set_xticks(range(nc))
    ax_ho.set_xticklabels([str(c) for c in coeffs], fontsize=8)
    ax_ho.set_xlabel('Steering coefficient', fontsize=8)
    dual_y(ax_ho, n_layers)
    ax_ho.set_title('Old Method E[value]', fontsize=10, fontweight='bold')

    cb0 = plt.colorbar(im0, ax=ax_ho, shrink=0.7, pad=0.02)
    cb0.set_label(ev_unit(q_full), fontsize=7)
    cb0.ax.yaxis.set_major_locator(cb_major)
    cb0.ax.yaxis.set_minor_locator(cb_minor)
    cb0.ax.yaxis.set_major_formatter(cb_fmt)
    cb0.ax.tick_params(labelsize=5)

    # ────────────────────────────────────────
    # (0,1) New E[value] Heatmap  (neutral center)
    # ────────────────────────────────────────
    ax_hn = fig.add_subplot(gs[0, 1])

    mat_new = np.zeros((n_layers, 3))
    for li in range(n_layers):
        for ci, cond in enumerate(NEW_COLS):
            mat_new[li, ci] = compute_ev(
                get_new_probs(qr, li, cond), q_full)

    xlabels_new = [cond_display(exp, c) for c in NEW_COLS]

    im1 = ax_hn.imshow(mat_new, aspect='auto', origin='lower',
                        cmap='coolwarm', norm=norm, interpolation='nearest')
    ax_hn.set_xticks(range(3))
    ax_hn.set_xticklabels(xlabels_new, fontsize=8, fontweight='bold')
    ax_hn.set_xlabel('Condition', fontsize=8)
    dual_y(ax_hn, n_layers)
    ax_hn.set_title('New Method E[value]', fontsize=10, fontweight='bold')

    cb1 = plt.colorbar(im1, ax=ax_hn, shrink=0.7, pad=0.02)
    cb1.set_label(ev_unit(q_full), fontsize=7)
    cb1.ax.yaxis.set_major_locator(cb_major)
    cb1.ax.yaxis.set_minor_locator(cb_minor)
    cb1.ax.yaxis.set_major_formatter(cb_fmt)
    cb1.ax.tick_params(labelsize=5)

    # ────────────────────────────────────────
    # Compute E arrays for pine trees
    # ────────────────────────────────────────
    neg_c, pos_c = min(coeffs), max(coeffs)
    layers = np.arange(n_layers)

    # Old method
    e_old_n = np.array([compute_ev(get_old_probs(qr, l, 0), q_full)
                        for l in range(n_layers)])
    e_old_lo = np.array([compute_ev(get_old_probs(qr, l, neg_c), q_full)
                         for l in range(n_layers)])
    e_old_hi = np.array([compute_ev(get_old_probs(qr, l, pos_c), q_full)
                         for l in range(n_layers)])
    safe_old = np.where(np.abs(e_old_n) > 1e-10, e_old_n, 1e-10)
    d_old_neg = (e_old_lo - e_old_n) / safe_old
    d_old_pos = (e_old_hi - e_old_n) / safe_old

    # New method
    e_new_n = np.array([compute_ev(get_new_probs(qr, l, 'neutral'), q_full)
                        for l in range(n_layers)])
    e_new_pos = np.array([compute_ev(get_new_probs(qr, l, 'employed'), q_full)
                          for l in range(n_layers)])
    e_new_neg = np.array([compute_ev(get_new_probs(qr, l, 'unemployed'), q_full)
                          for l in range(n_layers)])
    safe_new = np.where(np.abs(e_new_n) > 1e-10, e_new_n, 1e-10)
    d_new_neg = (e_new_neg - e_new_n) / safe_new
    d_new_pos = (e_new_pos - e_new_n) / safe_new

    # ────────────────────────────────────────
    # (1,0) Old Pine Tree
    # ────────────────────────────────────────
    ax_po = fig.add_subplot(gs[1, 0])
    bh = 0.6
    ax_po.barh(layers, d_old_neg, bh, color='#1565C0', alpha=0.7,
               edgecolor='black', lw=0.2,
               label=f'c={neg_c} ({cl["neg_short"]})')
    ax_po.barh(layers, d_old_pos, bh, color='#C62828', alpha=0.7,
               edgecolor='black', lw=0.2,
               label=f'c={pos_c} ({cl["pos_short"]})')
    ax_po.axvline(0, color='black', lw=1)
    for l in top_layers:
        if l < n_layers:
            ax_po.axhline(l, color='gold', ls='-', lw=1.2, alpha=0.4)

    ax_po.set_xlim(-1, 1)
    ax_po.set_ylim(-0.5, n_layers - 0.5)
    # L0 at bottom, L60 at top — NO invert
    dual_y_pine(ax_po, n_layers)

    ax_po.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_po.xaxis.set_minor_locator(mticker.MultipleLocator(0.005))
    ax_po.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    plt.setp(ax_po.get_xticklabels(), fontsize=5, rotation=45)
    ax_po.grid(True, alpha=0.06, axis='x', which='major')
    ax_po.grid(True, alpha=0.03, axis='x', which='minor')
    ax_po.set_xlabel('ΔE/E_neutral', fontsize=8)
    ax_po.set_title('Old Method Pine Tree', fontsize=10, fontweight='bold')
    ax_po.legend(fontsize=7, loc='lower right')

    # ────────────────────────────────────────
    # (1,1) New Pine Tree  (auto-scaled)
    # ────────────────────────────────────────
    ax_pn = fig.add_subplot(gs[1, 1])
    ax_pn.barh(layers, d_new_neg, bh, color='#1565C0', alpha=0.7,
               edgecolor='black', lw=0.2, label=f'{cl["neg"]}')
    ax_pn.barh(layers, d_new_pos, bh, color='#C62828', alpha=0.7,
               edgecolor='black', lw=0.2, label=f'{cl["pos"]}')
    ax_pn.axvline(0, color='black', lw=1)
    for l in top_layers:
        if l < n_layers:
            ax_pn.axhline(l, color='gold', ls='-', lw=1.2, alpha=0.4)

    xlim_new = max(np.abs(d_new_neg).max(), np.abs(d_new_pos).max(),
                   0.0001) * 1.15
    xlim_new = min(xlim_new, 1.0)
    ax_pn.set_xlim(-xlim_new, xlim_new)
    ax_pn.set_ylim(-0.5, n_layers - 0.5)
    dual_y_pine(ax_pn, n_layers)

    ax_pn.xaxis.set_major_locator(mticker.MultipleLocator(0.005))
    ax_pn.xaxis.set_minor_locator(mticker.MultipleLocator(0.001))
    ax_pn.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    plt.setp(ax_pn.get_xticklabels(), fontsize=5, rotation=45)
    ax_pn.grid(True, alpha=0.06, axis='x', which='major')
    ax_pn.grid(True, alpha=0.03, axis='x', which='minor')
    ax_pn.set_xlabel(f'ΔE/E_neutral  (xlim=±{xlim_new:.3f})', fontsize=8)
    ax_pn.set_title('New Method Pine Tree', fontsize=10, fontweight='bold')
    ax_pn.legend(fontsize=7, loc='lower right')

    # ────────────────────────────────────────
    # Title + prompt box
    # ────────────────────────────────────────
    add_prompt_box(fig, q_full['prompt'])

    # E neutral summary
    en_med = float(np.median(e_old_n))
    if q_full['answer_format'] == 'time':
        h, m = int(en_med), int((en_med % 1) * 60)
        en_str = f'{h}:{m:02d}'
    else:
        en_str = f'{en_med:.3f}'

    fig.suptitle(
        f"{q_full['id']}: {q_full['description']}\n"
        f"{it.upper()}/cfg{cfg} ({config_label(exp, cfg)})  |  "
        f"E_neutral median={en_str}",
        fontsize=12, fontweight='bold', y=0.97)

    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# S1: Best Layers Summary (combined v7+v8)
# ══════════════════════════════════════════════════════════════════════

def pg_summary_best(pdf, all_data, exp, q_lookup):
    combos = []
    for it in ['v7', 'v8']:
        for key, run in sorted(all_data[it].items()):
            if key[0] == exp:
                combos.append((it, key[1], run))
    if not combos:
        return set()

    n_layers = combos[0][2]['meta']['n_decoder_layers']
    coeffs   = combos[0][2]['meta']['old_coeffs']
    neg_c, pos_c = min(coeffs), max(coeffs)
    top_global = set()

    fig, ax = plt.subplots(figsize=(22, 16)); ax.axis('off')
    lines = [f'{exp.upper()} — Best Layers (ΔE/E steering effect)\n',
             '='*100 + '\n\n']

    for it, cfg, run in combos:
        old_eff = np.zeros(n_layers)
        new_eff = np.zeros(n_layers)
        n_q = 0
        for qr in run['results']:
            qid = qr['question']['id']
            qf  = q_lookup.get(qid)
            if not qf or qf['type'] != 'finegrain':
                continue
            n_q += 1
            for l in range(n_layers):
                en = compute_ev(get_old_probs(qr, l, 0), qf)
                safe = en if abs(en) > 1e-10 else 1e-10
                e_lo = compute_ev(get_old_probs(qr, l, neg_c), qf)
                e_hi = compute_ev(get_old_probs(qr, l, pos_c), qf)
                old_eff[l] += abs((e_hi - en) / safe) + abs((e_lo - en) / safe)

                en2 = compute_ev(get_new_probs(qr, l, 'neutral'), qf)
                safe2 = en2 if abs(en2) > 1e-10 else 1e-10
                ep = compute_ev(get_new_probs(qr, l, 'employed'), qf)
                eu = compute_ev(get_new_probs(qr, l, 'unemployed'), qf)
                new_eff[l] += abs((ep - en2) / safe2) + abs((eu - en2) / safe2)

        if n_q == 0:
            continue
        old_eff /= n_q; new_eff /= n_q
        ot5 = np.argsort(old_eff)[-5:][::-1]
        nt5 = np.argsort(new_eff)[-5:][::-1]
        both = sorted(set(ot5) & set(nt5))
        top_global.update(ot5.tolist())
        top_global.update(nt5.tolist())

        lines.append(
            f'{it.upper()} cfg{cfg} ({config_label(exp, cfg)})  '
            f'[{n_q} finegrain Qs]\n')
        lines.append('  Old top-5: ' + ', '.join(
            f'L{l}({old_eff[l]:.3f})' for l in ot5) + '\n')
        lines.append('  New top-5: ' + ', '.join(
            f'L{l}({new_eff[l]:.3f})' for l in nt5) + '\n')
        lines.append('  Intersect: ' + (', '.join(
            f'L{l}' for l in both) if both else 'NONE') + '\n\n')

    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace')
    fig.suptitle(f'{exp.upper()} — Best Layers: Old vs New Method',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=150); plt.close()
    return top_global


# ══════════════════════════════════════════════════════════════════════
# S2: Cross-question Consistency (one page per iter)
# ══════════════════════════════════════════════════════════════════════

def pg_summary_consistency_iter(pdf, all_data, exp, iter_name,
                                 top_layers, q_lookup):
    """One page for a single iteration (v7 or v8)."""
    combos = []
    for key, run in sorted(all_data[iter_name].items()):
        if key[0] == exp:
            combos.append((key[1], run))
    if not combos:
        return

    n_layers = combos[0][1]['meta']['n_decoder_layers']
    coeffs   = combos[0][1]['meta']['old_coeffs']
    pos_c    = max(coeffs)
    threshold = 0.001

    fig, ax = plt.subplots(figsize=(22, 16)); ax.axis('off')
    lines = [
        f'{exp.upper()} — Cross-Question ΔE/E Consistency  '
        f'[{iter_name.upper()}]\n',
        f'Threshold: |ΔE/E_neutral| > {threshold:.3f}\n',
        '='*100 + '\n\n']

    for cfg, run in combos:
        fg_results = []
        for qr in run['results']:
            qid = qr['question']['id']
            qf  = q_lookup.get(qid)
            if qf and qf['type'] == 'finegrain':
                fg_results.append((qr, qf))
        n_q = len(fg_results)
        if n_q == 0:
            continue

        lines.append(
            f'{iter_name.upper()} cfg{cfg} '
            f'({config_label(exp, cfg)})  [{n_q} Qs]\n')
        lines.append(
            f'  {"Lyr":>4s}  {"#old>thr":>8s}/{n_q}  '
            f'{"#new>thr":>8s}/{n_q}  '
            f'{"old_mean":>10s}  {"new_mean":>10s}  '
            f'{"verdict":>10s}\n')
        lines.append('  ' + '-'*70 + '\n')

        scores = []
        for l in range(n_layers):
            no, nn = 0, 0
            so, sn = 0.0, 0.0
            for qr, qf in fg_results:
                en = compute_ev(get_old_probs(qr, l, 0), qf)
                safe = en if abs(en) > 1e-10 else 1e-10
                e_hi = compute_ev(get_old_probs(qr, l, pos_c), qf)
                eo = abs((e_hi - en) / safe)
                so += eo
                if eo > threshold: no += 1

                en2 = compute_ev(get_new_probs(qr, l, 'neutral'), qf)
                safe2 = en2 if abs(en2) > 1e-10 else 1e-10
                ep = compute_ev(get_new_probs(qr, l, 'employed'), qf)
                eo2 = abs((ep - en2) / safe2)
                sn += eo2
                if eo2 > threshold: nn += 1

            scores.append((l, no, nn, so / n_q, sn / n_q))

        scores.sort(key=lambda x: x[3] + x[4], reverse=True)
        for l, no, nn, om, nm in scores[:25]:
            if no >= n_q * 0.7 and nn >= n_q * 0.7:
                verdict = 'STRONG'
            elif no >= n_q * 0.5 or nn >= n_q * 0.5:
                verdict = 'moderate'
            else:
                verdict = 'weak'
            marker = ' ***' if l in top_layers else ''
            lines.append(
                f'  L{l:>3d}  {no:>8d}/{n_q}  {nn:>8d}/{n_q}  '
                f'{om:>10.3f}  {nm:>10.3f}  '
                f'{verdict:>10s}{marker}\n')
        lines.append('\n')

    lines.append('-'*70 + '\n*** = top layer from S1\n'
                 'STRONG = >70% Qs exceed threshold for BOTH methods\n')
    ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
            fontsize=6.5, va='top', fontfamily='monospace')
    fig.suptitle(
        f'{exp.upper()} — Consistent ΔE Layers  [{iter_name.upper()}]',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp2', 'exp1a'])
    parser.add_argument('--model', default='gemma4_31b',
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    if args.model == '12b':
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base, 'steering', 'output')
    else:
        model_base = MODEL_REGISTRY[args.model]['hf_name']
        out_dir = os.path.join(model_base, 'steering', 'output')
    os.makedirs(out_dir, exist_ok=True)

    all_data = load_all(args.model)
    q_lookup = build_q_lookup()

    for exp in args.exp:
        questions = ALL_QUESTIONS.get(exp, [])
        if not questions:
            continue

        pdf_path = os.path.join(out_dir, f'nl_steering_{exp}_v5.pdf')
        print(f"\nGenerating: {pdf_path}")

        with PdfPages(pdf_path) as pdf:
            # S1: Best layers (combined)
            print(f"  S1: Best layers summary")
            top_layers = pg_summary_best(
                pdf, all_data, exp, q_lookup)
            top_layers = sorted(top_layers)
            print(f"    Top layers: {top_layers}")

            # Per-question 2×2 pages
            for it in ['v7', 'v8']:
                for key, run in sorted(all_data[it].items()):
                    if key[0] != exp:
                        continue
                    meta = run['meta']; meta['iter'] = it
                    for qr in run['results']:
                        qid = qr['question']['id']
                        qf  = q_lookup.get(qid)
                        if not qf or qf['type'] != 'finegrain':
                            print(f"    SKIP {qid}")
                            continue
                        print(f"    {it}/cfg{key[1]} {qid}: "
                              f"{qf['description']}")
                        pg_question(pdf, qr, qf, meta, exp,
                                    top_layers)

            # S2: Consistency (separate pages per iter)
            for it in ['v7', 'v8']:
                print(f"  S2: Consistency [{it.upper()}]")
                pg_summary_consistency_iter(
                    pdf, all_data, exp, it, top_layers, q_lookup)

        print(f"  Done → {pdf_path}")


if __name__ == "__main__":
    main()