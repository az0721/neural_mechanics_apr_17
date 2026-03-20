# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Steering analysis Phase 2b — logits only, reads per_user_v2/.

# Usage:
#     python steering/analyze_all_users.py
# """
# import json, os, sys, glob, argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.colors import TwoSlopeNorm

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from config import OUTPUT_DIR

# PERUSER_DIR = os.path.join(OUTPUT_DIR, 'steering', 'per_user_v2')
# RESULTS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'results')
# os.makedirs(RESULTS_DIR, exist_ok=True)

# PROMPTS = {
#     'behavioral':        {'target': 'B',   'label': 'P(B = workplace)'},
#     'binary':            {'target': 'Yes', 'label': 'P(Yes = employed)'},
#     'departure':         {'target': 'A',   'label': 'P(A = left before 8AM)'},
#     'return_prediction': {'target': 'B',   'label': 'P(B = still away at noon)'},
# }
# CFGS = [5, 6]
# CFG_NAMES = {5: 'cfg5 (hash/orig/none)', 6: 'cfg6 (hash/orig/cot)'}
# HIGHLIGHT = [6, 12, 20, 25, 26, 28, 31, 33, 40]

# def load_all():
#     files = sorted(glob.glob(os.path.join(PERUSER_DIR, '*.json')))
#     emp, unemp = [], []
#     for f in files:
#         with open(f) as fh: d = json.load(fh)
#         (emp if d['meta']['is_employed'] else unemp).append(d)
#     m = (emp + unemp)[0]['meta']
#     print(f"Loaded {len(emp)} emp + {len(unemp)} unemp | C={m['coeffs']} | L{m['layers'][0]}-L{m['layers'][-1]}")
#     return emp, unemp, m['coeffs'], m['layers']

# def gp(d, cfg, qk, tgt, layers, coeffs):
#     m = np.full((len(layers), len(coeffs)), np.nan)
#     for li, l in enumerate(layers):
#         for ci, c in enumerate(coeffs):
#             p = d['logits'].get(str((cfg, qk, l, c)), {})
#             m[li, ci] = p.get(tgt, np.nan)
#     return m

# def gm(grp, cfg, qk, tgt, layers, coeffs):
#     return np.array([gp(d, cfg, qk, tgt, layers, coeffs) for d in grp])

# # ── Curves ──
# def pg_curves(pdf, em, um, coeffs, layers, qk, info, pls, cfg, lbl=''):
#     nr = min(len(pls), 4)
#     fig, axes = plt.subplots(nr, 1, figsize=(12, 4*nr))
#     if nr == 1: axes = [axes]
#     ne, nu = em.shape[0], um.shape[0]
#     for r in range(nr):
#         if r >= len(pls): break
#         l = pls[r]; ax = axes[r]
#         if l not in layers: continue
#         li = list(layers).index(l)
#         em_ = np.nanmean(em[:, li, :], 0); es = np.nanstd(em[:, li, :], 0)
#         um_ = np.nanmean(um[:, li, :], 0); us = np.nanstd(um[:, li, :], 0)
#         ax.plot(coeffs, em_, 'b-o', ms=5, lw=2.5, label=f'Emp(n={ne})', zorder=5)
#         ax.fill_between(coeffs, em_-es, em_+es, color='blue', alpha=.12)
#         ax.plot(coeffs, um_, 'r-s', ms=5, lw=2.5, label=f'Unemp(n={nu})', zorder=5)
#         ax.fill_between(coeffs, um_-us, um_+us, color='red', alpha=.12)
#         ax.fill_between(coeffs, em_, um_, alpha=.08, color='purple')
#         ax.axhline(.5, color='gray', ls='--', alpha=.5); ax.axvline(0, color='gray', ls='--', alpha=.5)
#         ax.set_ylim(-.05, 1.05); ax.set_ylabel(info['label']); ax.set_title(f'Layer {l}', fontweight='bold')
#         ax.legend(fontsize=8); ax.grid(True, alpha=.15)
#         if r == nr-1: ax.set_xlabel('Steering Coefficient')
#     fig.suptitle(f'{qk.upper()} — {info["label"]}{lbl}\n{CFG_NAMES[cfg]}', fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close()

# # ── Heatmaps ──
# def pg_heat(pdf, em, um, coeffs, layers, qk, info):
#     fig, axes = plt.subplots(2, 2, figsize=(16, 14))
#     zi = list(coeffs).index(0); st = max(1, len(layers)//12)
#     for col, (mats, g) in enumerate([(em, 'Employed'), (um, 'Unemployed')]):
#         mm = np.nanmean(mats, 0)
#         ax = axes[0, col]
#         ax.imshow(mm, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
#         ax.set_title(f'{g} — Mean {info["label"]}', fontweight='bold')
#         ax.set_xticks(range(len(coeffs))); ax.set_xticklabels(coeffs, fontsize=8)
#         ax.set_yticks(range(0, len(layers), st))
#         ax.set_yticklabels([layers[i] for i in range(0, len(layers), st)], fontsize=7)
#         ax.set_ylabel('Layer'); ax.axvline(zi, color='white', ls='--')
#         d = mm - mm[:, zi:zi+1]; fin = d[np.isfinite(d)]
#         vm = max(abs(fin.min()), abs(fin.max()), .01) if len(fin) else .01
#         ax = axes[1, col]
#         ax.imshow(d, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-abs(vm)-1e-6, vcenter=0, vmax=abs(vm)+1e-6), origin='lower')
#         ax.set_title(f'{g} — ΔP', fontweight='bold')
#         ax.set_xticks(range(len(coeffs))); ax.set_xticklabels(coeffs, fontsize=8)
#         ax.set_yticks(range(0, len(layers), st))
#         ax.set_yticklabels([layers[i] for i in range(0, len(layers), st)], fontsize=7)
#         ax.set_ylabel('Layer'); ax.set_xlabel('Coefficient'); ax.axvline(zi, color='black', ls='--')
#     fig.suptitle(f'Heatmap: {qk.upper()} — {info["label"]}', fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,.93]); pdf.savefig(fig); plt.close()

# # ── Delta bars ──
# def pg_delta(pdf, em, um, coeffs, layers, qk, info, cfg):
#     fig, ax = plt.subplots(figsize=(18, 8))
#     zi = list(coeffs).index(0); x = np.arange(len(layers)); w = .2
#     def ds(m, ci):
#         d = m[:, :, ci] - m[:, :, zi]
#         return np.nan_to_num(d.mean(0)), np.nan_to_num(d.std(0))
#     enm, ens = ds(em, 0); epm, eps = ds(em, -1)
#     unm, uns = ds(um, 0); upm, ups = ds(um, -1)
#     ax.bar(x-1.5*w, enm, w, yerr=ens, capsize=2, label=f'Emp c={coeffs[0]}', color='#1565C0', alpha=.85)
#     ax.bar(x-.5*w, epm, w, yerr=eps, capsize=2, label=f'Emp c={coeffs[-1]}', color='#64B5F6', alpha=.85)
#     ax.bar(x+.5*w, unm, w, yerr=uns, capsize=2, label=f'Unemp c={coeffs[0]}', color='#C62828', alpha=.85)
#     ax.bar(x+1.5*w, upm, w, yerr=ups, capsize=2, label=f'Unemp c={coeffs[-1]}', color='#EF9A9A', alpha=.85)
#     st = max(1, len(layers)//25)
#     ax.set_xticks(x[::st]); ax.set_xticklabels([f'L{layers[i]}' for i in range(0,len(layers),st)], fontsize=6, rotation=45)
#     ax.set_ylabel(f'Δ {info["label"]}'); ax.set_title(f'Δ: {qk.upper()} — {CFG_NAMES[cfg]}', fontweight='bold')
#     ax.legend(fontsize=8, ncol=4); ax.axhline(0, color='black', lw=.8); ax.grid(True, alpha=.15, axis='y')
#     for a, c in [(enm, '#1565C0'), (upm, '#EF9A9A')]:
#         if np.all(a == 0): continue
#         i = np.nanargmax(np.abs(a))
#         if abs(a[i]) > .02:
#             ax.annotate(f'L{layers[i]}\n{a[i]:+.3f}', xy=(i,a[i]), fontsize=7, fontweight='bold',
#                         color=c, ha='center', va='bottom' if a[i]>0 else 'top')
#     plt.tight_layout(); pdf.savefig(fig); plt.close()

# # ── Individual overlay ──
# def pg_indiv(pdf, em, um, coeffs, layers, qk, info, cfg, ols):
#     vl = [l for l in ols if l in layers]
#     if not vl: return
#     fig, axes = plt.subplots(len(vl), 1, figsize=(12, 3.5*len(vl)))
#     if len(vl) == 1: axes = [axes]
#     for r, l in enumerate(vl):
#         ax = axes[r]; li = list(layers).index(l)
#         for u in range(em.shape[0]): ax.plot(coeffs, em[u,li,:], 'b-', alpha=.15, lw=.8)
#         for u in range(um.shape[0]): ax.plot(coeffs, um[u,li,:], 'r-', alpha=.15, lw=.8)
#         ax.plot(coeffs, np.nanmean(em[:,li,:],0), 'b-o', ms=5, lw=3, label='Emp mean', zorder=10)
#         ax.plot(coeffs, np.nanmean(um[:,li,:],0), 'r-s', ms=5, lw=3, label='Unemp mean', zorder=10)
#         ax.axhline(.5, color='gray', ls='--', alpha=.5); ax.axvline(0, color='gray', ls='--', alpha=.5)
#         ax.set_ylim(-.05,1.05); ax.set_ylabel(info['label']); ax.set_title(f'Layer {l}', fontweight='bold')
#         ax.legend(fontsize=8); ax.grid(True, alpha=.15)
#         if r == len(vl)-1: ax.set_xlabel('Steering Coefficient')
#     fig.suptitle(f'Individual: {qk.upper()} — {CFG_NAMES[cfg]}', fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close()

# # ── Best layers ──
# def pg_best(pdf, eu, uu, coeffs, layers):
#     fig, axes = plt.subplots(len(PROMPTS), len(CFGS), figsize=(14, 4*len(PROMPTS)), squeeze=False)
#     zi = list(coeffs).index(0); st = max(1, len(layers)//15)
#     for row, (qk, info) in enumerate(PROMPTS.items()):
#         for col, cfg in enumerate(CFGS):
#             ax = axes[row, col]
#             e = gm(eu, cfg, qk, info['target'], layers, coeffs)
#             u = gm(uu, cfg, qk, info['target'], layers, coeffs)
#             bs = np.abs(np.nanmean(e[:,:,zi],0) - np.nanmean(u[:,:,zi],0))
#             es = np.abs(np.nanmean(e[:,:,-1],0) - np.nanmean(u[:,:,-1],0))
#             g = es - bs
#             colors = ['#4CAF50' if (not np.isnan(v) and v>0) else '#FF5722' for v in g]
#             ax.bar(range(len(layers)), g, color=colors, alpha=.8)
#             ax.set_xticks(range(0,len(layers),st))
#             ax.set_xticklabels([f'L{layers[i]}' for i in range(0,len(layers),st)], fontsize=6, rotation=45)
#             ax.set_title(f'{qk} — {CFG_NAMES[cfg]}', fontsize=9, fontweight='bold')
#             ax.axhline(0, color='black', lw=.8); ax.grid(True, alpha=.15, axis='y')
#             if not np.all(np.isnan(g)):
#                 bi = np.nanargmax(g)
#                 if g[bi] > .01:
#                     ax.annotate(f'L{layers[bi]}', xy=(bi,g[bi]), fontsize=8, fontweight='bold',
#                                 color='darkgreen', ha='center', va='bottom')
#     fig.suptitle('Best Layers: Separation Gain | Green=increase Red=decrease', fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close()

# # ── Summary ──
# def pg_summary(pdf, eu, uu, coeffs, layers):
#     fig, ax = plt.subplots(figsize=(18, 16)); ax.axis('off')
#     zi = list(coeffs).index(0)
#     lines = [f'SUMMARY — {len(eu)+len(uu)} users | C={coeffs} | L{layers[0]}-L{layers[-1]}', '']
#     for cfg in CFGS:
#         lines.append(f'{"="*100}\n{CFG_NAMES[cfg]}\n{"="*100}')
#         for qk, info in PROMPTS.items():
#             e = gm(eu, cfg, qk, info['target'], layers, coeffs)
#             u = gm(uu, cfg, qk, info['target'], layers, coeffs)
#             lines.append(f'\n  {qk.upper()} — {info["label"]}')
#             lines.append(f'  {"Lyr":>5s} {"E_base":>8s} {"E_neg":>8s} {"E_pos":>8s} {"U_base":>8s} {"U_neg":>8s} {"U_pos":>8s} {"Sep0":>6s} {"SepE":>6s}')
#             for l in [x for x in HIGHLIGHT if x in layers]:
#                 li = list(layers).index(l)
#                 be,ne,pe = np.nanmean(e[:,li,zi]),np.nanmean(e[:,li,0]),np.nanmean(e[:,li,-1])
#                 bu,nu,pu = np.nanmean(u[:,li,zi]),np.nanmean(u[:,li,0]),np.nanmean(u[:,li,-1])
#                 mx = max(abs(ne-be),abs(pe-be),abs(nu-bu),abs(pu-bu))
#                 f = ' !!!' if mx>.15 else '  **' if mx>.05 else ''
#                 lines.append(f'  L{l:>2d}:  {be:>8.4f}{ne:>8.4f}{pe:>8.4f} {bu:>8.4f}{nu:>8.4f}{pu:>8.4f} {abs(be-bu):>6.4f}{abs(pe-pu):>6.4f}{f}')
#     ax.text(.01,.99,'\n'.join(lines), transform=ax.transAxes, fontsize=5.5, fontfamily='monospace', verticalalignment='top')
#     fig.suptitle('Numerical Summary', fontsize=13, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,.97]); pdf.savefig(fig); plt.close()

# # ── Baselines ──
# def pg_baselines(pdf, eu, uu, coeffs, layers):
#     fig, ax = plt.subplots(figsize=(18, 14)); ax.axis('off')
#     kls = [l for l in [12,25,26,31,33,40] if l in layers]
#     lines = ['PER-USER BASELINES @ coeff=0 — cfg5', '']
#     for qk, info in PROMPTS.items():
#         lines.append(f'  {qk.upper()} — {info["label"]}')
#         h = f'  {"User":>16s} {"E?":>3s}'
#         for l in kls: h += f' L{l:>2d}'
#         lines.append(h); lines.append('  '+'-'*(20+6*len(kls)))
#         for d in eu + uu:
#             r = f'  {d["meta"]["user"][:16]:>16s} {"Y" if d["meta"]["is_employed"] else "N":>3s}'
#             for l in kls:
#                 p = d['logits'].get(str((5,qk,l,0)),{}).get(info['target'],float('nan'))
#                 r += f' {p:>.3f}' if not np.isnan(p) else '   N/A'
#             lines.append(r)
#         lines.append('')
#     ax.text(.01,.99,'\n'.join(lines), transform=ax.transAxes, fontsize=5.5, fontfamily='monospace', verticalalignment='top')
#     fig.suptitle('Per-User Baselines', fontsize=13, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,.97]); pdf.savefig(fig); plt.close()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--min-users', type=int, default=5)
#     args = parser.parse_args()
#     eu, uu, coeffs, layers = load_all()
#     if len(eu) < args.min_users or len(uu) < args.min_users:
#         print(f"Need {args.min_users}/group. Abort."); return
#     out = os.path.join(RESULTS_DIR, 'steering_results_v2.pdf')
#     print(f"\nGenerating: {out}")
#     ols = [l for l in [12,25,26,28,31,33,40] if l in layers]; pc = 0
#     with PdfPages(out) as pdf:
#         for qk, info in PROMPTS.items():
#             print(f"\n  {qk}:")
#             for cfg in CFGS:
#                 e = gm(eu, cfg, qk, info['target'], layers, coeffs)
#                 u = gm(uu, cfg, qk, info['target'], layers, coeffs)
#                 hl = [l for l in HIGHLIGHT if l in layers]
#                 for ps in range(0, len(hl), 4):
#                     pl = hl[ps:ps+4]; pn = ps//4+1; tp = (len(hl)+3)//4
#                     print(f"    {CFG_NAMES[cfg]}: curves {pn}/{tp}")
#                     pg_curves(pdf, e, u, coeffs, layers, qk, info, pl, cfg, f' ({pn}/{tp})'); pc += 1
#                 print(f"    {CFG_NAMES[cfg]}: delta"); pg_delta(pdf, e, u, coeffs, layers, qk, info, cfg); pc += 1
#                 print(f"    {CFG_NAMES[cfg]}: overlay"); pg_indiv(pdf, e, u, coeffs, layers, qk, info, cfg, ols); pc += 1
#             ea = np.nanmean([gm(eu, c, qk, info['target'], layers, coeffs) for c in CFGS], 0)
#             ua = np.nanmean([gm(uu, c, qk, info['target'], layers, coeffs) for c in CFGS], 0)
#             print(f"    heatmaps"); pg_heat(pdf, ea, ua, coeffs, layers, qk, info); pc += 1
#         print("\n  best-layers"); pg_best(pdf, eu, uu, coeffs, layers); pc += 1
#         print("  summary"); pg_summary(pdf, eu, uu, coeffs, layers); pc += 1
#         print("  baselines"); pg_baselines(pdf, eu, uu, coeffs, layers); pc += 1
#     print(f"\n{'='*60}\nDone! {pc} pages -> {out}\n{'='*60}")

# if __name__ == "__main__": main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steering analysis Phase 2b — redesigned visualization.

Changes from previous version:
  - Filter to VALID_COEFFS = [-10, -5, 0, 5, 10] only (±20/±50 cause model collapse
    because steering vectors are raw/unnormalized, so coeff×norm >> hidden state norm)
  - Curve pages: 4×4 grid (16 layers per page), wider landscape layout
  - Heatmaps: per-cfg with proper colorbars on side
  - Delta bars: use valid coeffs only
  - All other pages kept, filtered to valid coeffs

Usage:
    python steering/analyze_all_users.py
"""
import json, os, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR

PERUSER_DIR = os.path.join(OUTPUT_DIR, 'steering', 'per_user_v2')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPTS = {
    'behavioral':        {'target': 'B',   'label': 'P(B = workplace)'},
    'binary':            {'target': 'Yes', 'label': 'P(Yes = employed)'},
    'departure':         {'target': 'A',   'label': 'P(A = left before 8AM)'},
    'return_prediction': {'target': 'B',   'label': 'P(B = still away at noon)'},
}
CFGS = [5, 6]
CFG_NAMES = {5: 'cfg5 (hash/orig/none)', 6: 'cfg6 (hash/orig/cot)'}
VALID_COEFFS = [-10, -5, 0, 5, 10]
HIGHLIGHT = [6, 12, 20, 25, 26, 28, 31, 33, 40]


# ══════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════

def load_all():
    files = sorted(glob.glob(os.path.join(PERUSER_DIR, '*.json')))
    emp, unemp = [], []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        (emp if d['meta']['is_employed'] else unemp).append(d)
    m = (emp + unemp)[0]['meta']
    all_coeffs = m['coeffs']
    layers = m['layers']
    valid_idx = [i for i, c in enumerate(all_coeffs) if c in VALID_COEFFS]
    coeffs = [all_coeffs[i] for i in valid_idx]
    print(f"Loaded {len(emp)} emp + {len(unemp)} unemp")
    print(f"  All coeffs: {all_coeffs} -> filtered to {coeffs}")
    print(f"  Layers: L{layers[0]}-L{layers[-1]} ({len(layers)})")
    return emp, unemp, coeffs, layers, all_coeffs, valid_idx


def gp(d, cfg, qk, tgt, layers, all_coeffs, valid_idx):
    m = np.full((len(layers), len(valid_idx)), np.nan)
    for li, l in enumerate(layers):
        for ci_out, ci_in in enumerate(valid_idx):
            c = all_coeffs[ci_in]
            p = d['logits'].get(str((cfg, qk, l, c)), {})
            m[li, ci_out] = p.get(tgt, np.nan)
    return m


def gm(grp, cfg, qk, tgt, layers, all_coeffs, valid_idx):
    return np.array([gp(d, cfg, qk, tgt, layers, all_coeffs, valid_idx)
                     for d in grp])


# ══════════════════════════════════════════════════════════════════
# Curves — 4×4 grid, all layers
# ══════════════════════════════════════════════════════════════════

def pages_curves_grid(pdf, em, um, coeffs, layers, qk, info, cfg):
    ne, nu = em.shape[0], um.shape[0]
    n_layers = len(layers)
    per_page = 16
    n_pages = (n_layers + per_page - 1) // per_page

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, n_layers)
        page_lis = list(range(start, end))
        n_plots = len(page_lis)
        ncols, nrows = 4, (n_plots + 3) // 4

        fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, li in enumerate(page_lis):
            r, c = idx // ncols, idx % ncols
            ax = axes[r, c]
            layer = layers[li]

            em_ = np.nanmean(em[:, li, :], 0)
            es = np.nanstd(em[:, li, :], 0)
            um_ = np.nanmean(um[:, li, :], 0)
            us = np.nanstd(um[:, li, :], 0)

            ax.plot(coeffs, em_, 'b-o', ms=3, lw=1.8, zorder=5)
            ax.fill_between(coeffs, em_ - es, em_ + es, color='blue', alpha=.10)
            ax.plot(coeffs, um_, 'r-s', ms=3, lw=1.8, zorder=5)
            ax.fill_between(coeffs, um_ - us, um_ + us, color='red', alpha=.10)
            ax.fill_between(coeffs, em_, um_, alpha=.06, color='purple')
            ax.axhline(.5, color='gray', ls='--', alpha=.4, lw=.5)
            ax.axvline(0, color='gray', ls='--', alpha=.4, lw=.5)
            ax.set_ylim(-.05, 1.05)
            ax.set_title(f'L{layer}', fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=7)
            if r == nrows - 1:
                ax.set_xlabel('coeff', fontsize=8)
            if c == 0:
                ax.set_ylabel(info['label'], fontsize=7)

        for idx in range(n_plots, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        handles = [
            plt.Line2D([0], [0], color='blue', marker='o', ms=5, lw=2,
                       label=f'Employed (n={ne})'),
            plt.Line2D([0], [0], color='red', marker='s', ms=5, lw=2,
                       label=f'Unemployed (n={nu})'),
            plt.Line2D([0], [0], color='gray', ls='--', lw=1,
                       label='0.5 baseline'),
        ]
        fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(
            f'{qk.upper()} — {info["label"]}  |  {CFG_NAMES[cfg]}  |  '
            f'Coeffs: {coeffs}  |  L{layers[start]}–L{layers[end-1]}  '
            f'({page+1}/{n_pages})',
            fontsize=13, fontweight='bold', y=1.015)
        plt.tight_layout(rect=[0, 0, 1, 0.955])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


# ══════════════════════════════════════════════════════════════════
# Heatmaps — per cfg, with colorbars
# ══════════════════════════════════════════════════════════════════

def pg_heat(pdf, em, um, coeffs, layers, qk, info, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    zi = list(coeffs).index(0)
    step = max(1, len(layers) // 16)

    for col, (mats, grp) in enumerate([(em, 'Employed'), (um, 'Unemployed')]):
        mm = np.nanmean(mats, 0)

        # Top row: raw probability
        ax = axes[0, col]
        im = ax.imshow(mm, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       origin='lower', interpolation='nearest')
        ax.set_title(f'{grp} — Mean {info["label"]}', fontsize=11,
                     fontweight='bold')
        ax.set_xticks(range(len(coeffs)))
        ax.set_xticklabels(coeffs, fontsize=9)
        ax.set_yticks(range(0, len(layers), step))
        ax.set_yticklabels([f'L{layers[i]}' for i in range(0, len(layers), step)],
                           fontsize=7)
        ax.set_ylabel('Layer', fontsize=10)
        ax.axvline(zi, color='white', ls='--', lw=1.5)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(info['label'], fontsize=9)

        # Bottom row: delta from baseline
        baseline = mm[:, zi:zi + 1]
        delta = mm - baseline
        fin = delta[np.isfinite(delta)]
        vm = max(abs(fin.min()), abs(fin.max()), 0.01) if len(fin) else 0.01
        vm = max(vm, 1e-4)

        ax = axes[1, col]
        im = ax.imshow(delta, aspect='auto', cmap='RdBu_r',
                       norm=TwoSlopeNorm(vmin=-vm - 1e-6, vcenter=0,
                                         vmax=vm + 1e-6),
                       origin='lower', interpolation='nearest')
        ax.set_title(f'{grp} — ΔP from coeff=0', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(coeffs)))
        ax.set_xticklabels(coeffs, fontsize=9)
        ax.set_yticks(range(0, len(layers), step))
        ax.set_yticklabels([f'L{layers[i]}' for i in range(0, len(layers), step)],
                           fontsize=7)
        ax.set_ylabel('Layer', fontsize=10)
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.axvline(zi, color='black', ls='--', lw=1.5)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(f'Δ {info["label"]}', fontsize=9)

    fig.suptitle(
        f'Heatmap: {qk.upper()} — {info["label"]}  |  {CFG_NAMES[cfg]}\n'
        f'Top: raw P(target)  |  Bottom: ΔP from coeff=0',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Delta bars
# ══════════════════════════════════════════════════════════════════

def pg_delta(pdf, em, um, coeffs, layers, qk, info, cfg):
    fig, ax = plt.subplots(figsize=(18, 8))
    zi = list(coeffs).index(0)
    x = np.arange(len(layers))
    w = 0.2
    c_neg, c_pos = coeffs[0], coeffs[-1]

    def ds(m, ci):
        d = m[:, :, ci] - m[:, :, zi]
        return np.nan_to_num(d.mean(0)), np.nan_to_num(d.std(0))

    enm, ens = ds(em, 0);  epm, eps = ds(em, -1)
    unm, uns = ds(um, 0);  upm, ups = ds(um, -1)

    ax.bar(x - 1.5*w, enm, w, yerr=ens, capsize=2,
           label=f'Emp c={c_neg}', color='#1565C0', alpha=.85)
    ax.bar(x - 0.5*w, epm, w, yerr=eps, capsize=2,
           label=f'Emp c={c_pos}', color='#64B5F6', alpha=.85)
    ax.bar(x + 0.5*w, unm, w, yerr=uns, capsize=2,
           label=f'Unemp c={c_neg}', color='#C62828', alpha=.85)
    ax.bar(x + 1.5*w, upm, w, yerr=ups, capsize=2,
           label=f'Unemp c={c_pos}', color='#EF9A9A', alpha=.85)

    step = max(1, len(layers) // 25)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f'L{layers[i]}' for i in range(0, len(layers), step)],
                       fontsize=6, rotation=45)
    ax.set_ylabel(f'Δ {info["label"]}')
    ax.set_title(f'Steering Δ: {qk.upper()} — {CFG_NAMES[cfg]}  |  '
                 f'coeffs: {c_neg} / {c_pos}', fontweight='bold')
    ax.legend(fontsize=8, ncol=4, loc='lower left')
    ax.axhline(0, color='black', lw=.8)
    ax.grid(True, alpha=.15, axis='y')

    for a, color in [(enm, '#1565C0'), (upm, '#EF9A9A')]:
        if np.all(a == 0):
            continue
        i = np.nanargmax(np.abs(a))
        if abs(a[i]) > .02:
            ax.annotate(f'L{layers[i]}\n{a[i]:+.3f}', xy=(i, a[i]),
                        fontsize=7, fontweight='bold', color=color,
                        ha='center', va='bottom' if a[i] > 0 else 'top')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Individual overlay
# ══════════════════════════════════════════════════════════════════

def pg_indiv(pdf, em, um, coeffs, layers, qk, info, cfg, ols):
    vl = [l for l in ols if l in layers]
    if not vl:
        return
    fig, axes = plt.subplots(len(vl), 1, figsize=(12, 3.5 * len(vl)))
    if len(vl) == 1:
        axes = [axes]

    for r, l in enumerate(vl):
        ax = axes[r]
        li = list(layers).index(l)
        for u in range(em.shape[0]):
            ax.plot(coeffs, em[u, li, :], 'b-', alpha=.15, lw=.8)
        for u in range(um.shape[0]):
            ax.plot(coeffs, um[u, li, :], 'r-', alpha=.15, lw=.8)
        ax.plot(coeffs, np.nanmean(em[:, li, :], 0), 'b-o', ms=5, lw=3,
                label='Emp mean', zorder=10)
        ax.plot(coeffs, np.nanmean(um[:, li, :], 0), 'r-s', ms=5, lw=3,
                label='Unemp mean', zorder=10)
        ax.axhline(.5, color='gray', ls='--', alpha=.5)
        ax.axvline(0, color='gray', ls='--', alpha=.5)
        ax.set_ylim(-.05, 1.05)
        ax.set_ylabel(info['label'])
        ax.set_title(f'Layer {l}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=.15)
        if r == len(vl) - 1:
            ax.set_xlabel('Steering Coefficient')

    fig.suptitle(f'Individual: {qk.upper()} — {CFG_NAMES[cfg]}  |  '
                 f'Thin=individual, Bold=mean',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, .94])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Best layers
# ══════════════════════════════════════════════════════════════════

def pg_best(pdf, eu, uu, coeffs, layers, all_coeffs, valid_idx):
    fig, axes = plt.subplots(len(PROMPTS), len(CFGS),
                              figsize=(14, 4 * len(PROMPTS)), squeeze=False)
    zi = list(coeffs).index(0)
    step = max(1, len(layers) // 15)

    for row, (qk, info) in enumerate(PROMPTS.items()):
        for col, cfg in enumerate(CFGS):
            ax = axes[row, col]
            e = gm(eu, cfg, qk, info['target'], layers, all_coeffs, valid_idx)
            u = gm(uu, cfg, qk, info['target'], layers, all_coeffs, valid_idx)
            bs = np.abs(np.nanmean(e[:, :, zi], 0) - np.nanmean(u[:, :, zi], 0))
            es = np.abs(np.nanmean(e[:, :, -1], 0) - np.nanmean(u[:, :, -1], 0))
            g = es - bs
            colors = ['#4CAF50' if (not np.isnan(v) and v > 0)
                      else '#FF5722' for v in g]
            ax.bar(range(len(layers)), g, color=colors, alpha=.8)
            ax.set_xticks(range(0, len(layers), step))
            ax.set_xticklabels([f'L{layers[i]}' for i in
                                range(0, len(layers), step)],
                               fontsize=6, rotation=45)
            ax.set_title(f'{qk} — {CFG_NAMES[cfg]}', fontsize=9,
                         fontweight='bold')
            ax.axhline(0, color='black', lw=.8)
            ax.grid(True, alpha=.15, axis='y')
            if not np.all(np.isnan(g)):
                bi = np.nanargmax(g)
                if g[bi] > .01:
                    ax.annotate(f'L{layers[bi]}', xy=(bi, g[bi]),
                                fontsize=8, fontweight='bold',
                                color='darkgreen', ha='center', va='bottom')

    fig.suptitle(f'Best Layers: Separation Gain at coeff={coeffs[-1]}  |  '
                 f'Green=increase Red=decrease',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, .94])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════

def pg_summary(pdf, eu, uu, coeffs, layers, all_coeffs, valid_idx):
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.axis('off')
    zi = list(coeffs).index(0)
    cn, cp = coeffs[0], coeffs[-1]

    lines = [
        f'SUMMARY — {len(eu)+len(uu)} users  |  '
        f'Valid coeffs: {coeffs}  |  L{layers[0]}-L{layers[-1]}',
        f'(±20/±50 excluded: raw vectors not normalized -> model collapse)',
        '',
    ]

    for cfg in CFGS:
        lines.append(f'{"="*105}')
        lines.append(f'{CFG_NAMES[cfg]}')
        lines.append(f'{"="*105}')
        for qk, info in PROMPTS.items():
            e = gm(eu, cfg, qk, info['target'], layers, all_coeffs, valid_idx)
            u = gm(uu, cfg, qk, info['target'], layers, all_coeffs, valid_idx)
            lines.append(f'\n  {qk.upper()} — {info["label"]}')
            lines.append(
                f'  {"Lyr":>5s}  {"E_base":>8s} {"E_c="+str(cn):>10s} '
                f'{"E_c="+str(cp):>10s}  '
                f'{"U_base":>8s} {"U_c="+str(cn):>10s} '
                f'{"U_c="+str(cp):>10s}  '
                f'{"Sep@0":>7s} {"Sep@"+str(cp):>7s}')
            for l in [x for x in HIGHLIGHT if x in layers]:
                li = list(layers).index(l)
                be = np.nanmean(e[:, li, zi])
                ne = np.nanmean(e[:, li, 0])
                pe = np.nanmean(e[:, li, -1])
                bu = np.nanmean(u[:, li, zi])
                nu = np.nanmean(u[:, li, 0])
                pu = np.nanmean(u[:, li, -1])
                mx = max(abs(ne-be), abs(pe-be), abs(nu-bu), abs(pu-bu))
                flag = ' !!!' if mx > .15 else '  **' if mx > .05 else ''
                lines.append(
                    f'  L{l:>2d}:   {be:>8.4f} {ne:>10.4f} {pe:>10.4f}  '
                    f'{bu:>8.4f} {nu:>10.4f} {pu:>10.4f}  '
                    f'{abs(be-bu):>7.4f} {abs(pe-pu):>7.4f}{flag}')

    ax.text(.01, .99, '\n'.join(lines), transform=ax.transAxes, fontsize=5.8,
            fontfamily='monospace', verticalalignment='top')
    fig.suptitle('Numerical Summary (valid coefficients only)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, .97])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Per-user baselines
# ══════════════════════════════════════════════════════════════════

def pg_baselines(pdf, eu, uu, layers):
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis('off')
    kls = [l for l in [12, 25, 26, 31, 33, 40] if l in layers]
    lines = ['PER-USER BASELINES @ coeff=0 — cfg5', '']

    for qk, info in PROMPTS.items():
        lines.append(f'  {qk.upper()} — {info["label"]}')
        h = f'  {"User":>16s} {"E?":>3s}'
        for l in kls:
            h += f'  L{l:>2d}'
        lines.append(h)
        lines.append('  ' + '-' * (22 + 7 * len(kls)))
        for d in eu + uu:
            r = (f'  {d["meta"]["user"][:16]:>16s} '
                 f'{"Y" if d["meta"]["is_employed"] else "N":>3s}')
            for l in kls:
                p = d['logits'].get(str((5, qk, l, 0)), {}).get(
                    info['target'], float('nan'))
                r += f'  {p:.3f}' if not np.isnan(p) else '    N/A'
            lines.append(r)
        lines.append('')

    ax.text(.01, .99, '\n'.join(lines), transform=ax.transAxes, fontsize=5.5,
            fontfamily='monospace', verticalalignment='top')
    fig.suptitle('Per-User Baselines', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, .97])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-users', type=int, default=5)
    args = parser.parse_args()

    eu, uu, coeffs, layers, all_coeffs, valid_idx = load_all()
    if len(eu) < args.min_users or len(uu) < args.min_users:
        print(f"Need {args.min_users}/group. Abort.")
        return

    out = os.path.join(RESULTS_DIR, 'steering_results_v2.pdf')
    print(f"\nGenerating: {out}")
    ols = [l for l in [12, 25, 26, 28, 31, 33, 40] if l in layers]
    pc = 0

    with PdfPages(out) as pdf:
        for qk, info in PROMPTS.items():
            print(f"\n  {qk}:")
            for cfg in CFGS:
                e = gm(eu, cfg, qk, info['target'], layers, all_coeffs, valid_idx)
                u = gm(uu, cfg, qk, info['target'], layers, all_coeffs, valid_idx)

                print(f"    {CFG_NAMES[cfg]}: curves (4x4)")
                pages_curves_grid(pdf, e, u, coeffs, layers, qk, info, cfg)
                pc += (len(layers) + 15) // 16

                print(f"    {CFG_NAMES[cfg]}: heatmap")
                pg_heat(pdf, e, u, coeffs, layers, qk, info, cfg)
                pc += 1

                print(f"    {CFG_NAMES[cfg]}: delta")
                pg_delta(pdf, e, u, coeffs, layers, qk, info, cfg)
                pc += 1

                print(f"    {CFG_NAMES[cfg]}: overlay")
                pg_indiv(pdf, e, u, coeffs, layers, qk, info, cfg, ols)
                pc += 1

        print("\n  best-layers")
        pg_best(pdf, eu, uu, coeffs, layers, all_coeffs, valid_idx)
        pc += 1

        print("  summary")
        pg_summary(pdf, eu, uu, coeffs, layers, all_coeffs, valid_idx)
        pc += 1

        print("  baselines")
        pg_baselines(pdf, eu, uu, layers)
        pc += 1

    print(f"\n{'='*60}")
    print(f"Done! {pc} pages -> {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()