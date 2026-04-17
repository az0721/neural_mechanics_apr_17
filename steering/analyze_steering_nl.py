# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Analyze NL-based steering results v4 (Apr 16 revision).

# PDF structure per experiment:
#   Section A: Global Overview
#     A1: Old method — layer x coeff heatmap (all 7 coefficients)
#     A2: New method — layer effectiveness bar chart
#     A3: Best layers summary table (old + new, intersection)
#     A4: Cross-iteration hint direction (V8 - V7)

#   Section B: Per-Question Detail
#     B1a: Raw probability heatmaps (old method + new method side by side)
#     B1b: Delta probability heatmaps (change from neutral, diverging cmap)
#     B2:  Horizontal bar pine tree (MCQ only, target choice % change)

#   Section C: Quantitative Summary
#     C1/C2: Tables at top layers (paginated, max ~10 questions per page)
#     C3: Cross-question signal consistency

# Usage:
#     python steering/analyze_steering_nl.py --model gemma4_31b
# """
# import json, os, sys, glob, argparse, textwrap
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.colors import TwoSlopeNorm, Normalize

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from config import config_label, get_iter_output_dir, MODEL_REGISTRY
# from steering.prompts_nl import ALL_QUESTIONS, MCQ_TOKENS, DIGIT_TOKENS

# # ── Condition label mapping ──────────────────────────────────────────
# # JSON keys are always 'employed'/'unemployed' (code reuse from exp2).
# # For exp1a, y==1 is is_weekday, y==0 is weekend.
# COND_LABELS = {
#     'exp2':  {'employed': 'employed',  'unemployed': 'unemployed',
#               'pos': '+V_emp', 'neg': '+V_unemp'},
#     'exp1a': {'employed': 'weekday',   'unemployed': 'weekend',
#               'pos': '+V_weekday', 'neg': '+V_weekend'},
# }
# NEW_CONDS = ['neutral', 'employed', 'unemployed']


# # ══════════════════════════════════════════════════════════════════════
# # Data Loading
# # ══════════════════════════════════════════════════════════════════════

# def load_iter(iter_name, model_key):
#     out_base = get_iter_output_dir(model_key, iter_name)
#     nl_dir = os.path.join(out_base, 'steering', 'nl_results')
#     if not os.path.exists(nl_dir):
#         return {}
#     data = {}
#     for f in sorted(glob.glob(os.path.join(nl_dir, '*_nl.json'))):
#         with open(f) as fh:
#             d = json.load(fh)
#         m = d['meta']
#         data[(m['exp'], m['cfg'])] = d
#         print(f"  {iter_name}/{m['exp']}/cfg{m['cfg']}: "
#               f"{len(d['results'])} questions")
#     return data


# def load_all(model_key):
#     print(f"Loading NL steering results for {model_key}...")
#     return {it: load_iter(it, model_key) for it in ['v7', 'v8']}


# # ══════════════════════════════════════════════════════════════════════
# # Helpers
# # ══════════════════════════════════════════════════════════════════════

# def get_old_probs(qr, layer, coeff, token):
#     for e in qr['old_method']:
#         if e['layer'] == layer and e['coeff'] == coeff:
#             return e['probs'].get(token, 0)
#     return 0

# def get_new_probs(qr, layer, cond, token):
#     for e in qr['new_method']:
#         if e['layer'] == layer:
#             return e[cond].get(token, 0)
#     return 0

# def expected_digit(pd):
#     return sum(int(d) * pd.get(d, 0) for d in DIGIT_TOKENS)

# def pct_change(steered, neutral):
#     if abs(neutral) < 1e-10:
#         return 0.0
#     return (steered - neutral) / neutral

# def steering_eff_mcq(qr, q, layer, coeff):
#     tgt = q.get('target_option_pos')
#     if not tgt:
#         return 0
#     return get_old_probs(qr, layer, coeff, tgt) - get_old_probs(qr, layer, 0, tgt)

# def steering_eff_num(qr, layer, coeff):
#     e_s = expected_digit({d: get_old_probs(qr, layer, coeff, d) for d in DIGIT_TOKENS})
#     e_n = expected_digit({d: get_old_probs(qr, layer, 0, d) for d in DIGIT_TOKENS})
#     return e_s - e_n

# def new_eff_mcq(qr, q, layer):
#     tgt = q.get('target_option_pos')
#     if not tgt:
#         return 0
#     return abs(get_new_probs(qr, layer, 'employed', tgt) -
#                get_new_probs(qr, layer, 'unemployed', tgt))

# def new_eff_num(qr, layer):
#     e_e = expected_digit({d: get_new_probs(qr, layer, 'employed', d)
#                           for d in DIGIT_TOKENS})
#     e_u = expected_digit({d: get_new_probs(qr, layer, 'unemployed', d)
#                           for d in DIGIT_TOKENS})
#     return abs(e_e - e_u)

# def cond_label(exp, cond):
#     return COND_LABELS.get(exp, COND_LABELS['exp2']).get(cond, cond)

# def new_cond_display(exp, cond):
#     m = COND_LABELS.get(exp, COND_LABELS['exp2'])
#     if cond == 'neutral':
#         return 'neutral'
#     if cond == 'employed':
#         return m['pos']
#     if cond == 'unemployed':
#         return m['neg']
#     return cond

# def add_prompt_box(fig, prompt_str):
#     wrapped = textwrap.fill(prompt_str.replace('\n', ' | '), width=80)
#     fig.text(0.99, 0.99, wrapped, transform=fig.transFigure,
#              fontsize=5, va='top', ha='right', fontfamily='monospace',
#              bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#cccccc',
#                        alpha=0.9))


# # ══════════════════════════════════════════════════════════════════════
# # A1: Old Method — Layer x Coeff Heatmap
# # ══════════════════════════════════════════════════════════════════════

# def pg_a1_old_heatmap(pdf, all_data, exp):
#     combos = []
#     for it in ['v7', 'v8']:
#         for key, run in sorted(all_data[it].items()):
#             if key[0] == exp:
#                 combos.append((it, key[1], run))
#     if not combos:
#         return

#     n_layers = combos[0][2]['meta']['n_decoder_layers']
#     coeffs = combos[0][2]['meta']['old_coeffs']
#     nc = len(coeffs)

#     mats = []
#     for it, cfg, run in combos:
#         mat = np.zeros((n_layers, nc))
#         for qr in run['results']:
#             q = qr['question']
#             for ci, c in enumerate(coeffs):
#                 for l in range(n_layers):
#                     if q['type'] == 'mcq':
#                         mat[l, ci] += abs(steering_eff_mcq(qr, q, l, c))
#                     else:
#                         mat[l, ci] += abs(steering_eff_num(qr, l, c))
#         mat /= len(run['results'])
#         mats.append(mat)
#     gv = max(m.max() for m in mats)

#     ncols = min(len(combos), 4)
#     nrows = (len(combos) + ncols - 1) // ncols
#     fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 10*nrows),
#                               squeeze=False)
#     for idx, (it, cfg, run) in enumerate(combos):
#         ax = axes[idx // ncols, idx % ncols]
#         im = ax.imshow(mats[idx], aspect='auto', origin='lower',
#                        cmap='YlOrRd', vmin=0, vmax=gv)
#         ax.set_xticks(range(nc))
#         ax.set_xticklabels([str(c) for c in coeffs], fontsize=8)
#         ax.set_xlabel('Coefficient'); ax.set_ylabel('Layer')
#         ax.set_title(f'{it.upper()} cfg{cfg}\n({config_label(exp, cfg)})',
#                      fontweight='bold', fontsize=9)
#         plt.colorbar(im, ax=ax, shrink=0.6, label='Mean |effect|')
#     for idx in range(len(combos), nrows*ncols):
#         axes[idx // ncols, idx % ncols].set_visible(False)
#     fig.suptitle(f'{exp.upper()} — Old Method: Mean |Steering Effect|',
#                  fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,0.95])
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # A2: New Method — Layer Effectiveness
# # ══════════════════════════════════════════════════════════════════════

# def pg_a2_new_layer(pdf, all_data, exp):
#     combos = []
#     for it in ['v7', 'v8']:
#         for key, run in sorted(all_data[it].items()):
#             if key[0] == exp:
#                 combos.append((it, key[1], run))
#     if not combos:
#         return []

#     n_layers = combos[0][2]['meta']['n_decoder_layers']
#     cl = COND_LABELS[exp]

#     effs = []
#     for it, cfg, run in combos:
#         eff = np.zeros(n_layers)
#         for qr in run['results']:
#             q = qr['question']
#             for l in range(n_layers):
#                 if q['type'] == 'mcq':
#                     eff[l] += new_eff_mcq(qr, q, l)
#                 else:
#                     eff[l] += new_eff_num(qr, l)
#         eff /= len(run['results'])
#         effs.append(eff)
#     gym = max(e.max() for e in effs)

#     fig, axes = plt.subplots(len(combos), 1,
#                               figsize=(16, 3*len(combos)+2), squeeze=False)
#     top_all = set()
#     for idx, (it, cfg, run) in enumerate(combos):
#         ax = axes[idx, 0]
#         eff = effs[idx]
#         top5 = np.argsort(eff)[-5:][::-1]
#         top_all.update(top5.tolist())
#         ax.bar(range(n_layers), eff,
#                color='#2196F3' if it == 'v7' else '#FF5722',
#                alpha=0.7, edgecolor='black', lw=0.3)
#         for l in top5:
#             ax.bar(l, eff[l], color='gold', edgecolor='black', lw=0.8)
#             ax.text(l, eff[l]+gym*0.02, f'L{l}', ha='center',
#                     fontsize=6, fontweight='bold')
#         ax.set_ylabel(f'Mean |{cl["pos"]}-{cl["neg"]}|')
#         ax.set_ylim(0, gym*1.15)
#         ax.set_title(f'{it.upper()} cfg{cfg} ({config_label(exp, cfg)})',
#                      fontweight='bold', fontsize=10)
#         ax.grid(True, alpha=0.1, axis='y')
#         ax.set_xlim(-0.5, n_layers-0.5)
#     axes[-1, 0].set_xlabel('Layer')
#     fig.suptitle(f'{exp.upper()} — New Method: |{cl["pos"]} - {cl["neg"]}| '
#                  f'per layer', fontsize=13, fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,0.93])
#     pdf.savefig(fig, dpi=150); plt.close()
#     return sorted(top_all)


# # ══════════════════════════════════════════════════════════════════════
# # A3: Best Layers Summary Table
# # ══════════════════════════════════════════════════════════════════════

# def pg_a3_summary(pdf, all_data, exp, top_old, top_new):
#     combos = []
#     for it in ['v7', 'v8']:
#         for key, run in sorted(all_data[it].items()):
#             if key[0] == exp:
#                 combos.append((it, key[1], run))
#     if not combos:
#         return
#     n_layers = combos[0][2]['meta']['n_decoder_layers']
#     coeffs = combos[0][2]['meta']['old_coeffs']

#     fig, ax = plt.subplots(figsize=(18, 10)); ax.axis('off')
#     lines = [f'{exp.upper()} — Best Layers Summary\n', '='*100+'\n\n']
#     for it, cfg, run in combos:
#         old_eff = np.zeros(n_layers); new_eff = np.zeros(n_layers)
#         n_q = len(run['results'])
#         for qr in run['results']:
#             q = qr['question']
#             for l in range(n_layers):
#                 for c in coeffs:
#                     if q['type']=='mcq': old_eff[l]+=abs(steering_eff_mcq(qr,q,l,c))
#                     else: old_eff[l]+=abs(steering_eff_num(qr,l,c))
#                 if q['type']=='mcq': new_eff[l]+=new_eff_mcq(qr,q,l)
#                 else: new_eff[l]+=new_eff_num(qr,l)
#         old_eff/=(n_q*len(coeffs)); new_eff/=n_q
#         ot5=np.argsort(old_eff)[-5:][::-1]
#         nt5=np.argsort(new_eff)[-5:][::-1]
#         both=sorted(set(ot5)&set(nt5))
#         lines.append(f'{it.upper()} cfg{cfg} ({config_label(exp,cfg)})\n')
#         lines.append(f'  Old top-5: '+', '.join(f'L{l}({old_eff[l]:.4f})' for l in ot5)+'\n')
#         lines.append(f'  New top-5: '+', '.join(f'L{l}({new_eff[l]:.4f})' for l in nt5)+'\n')
#         lines.append(f'  Intersect: '+(', '.join(f'L{l}' for l in both) if both else 'NONE')+'\n\n')
#     lines.append('-'*100+'\n')
#     bg=sorted(set(top_old)&set(top_new))
#     lines.append(f'Global old top:  {", ".join(f"L{l}" for l in top_old)}\n')
#     lines.append(f'Global new top:  {", ".join(f"L{l}" for l in top_new)}\n')
#     lines.append(f'Global intersect: '+(', '.join(f'L{l}' for l in bg) if bg else 'NONE')+'\n')
#     ax.text(0.02,0.98,''.join(lines),transform=ax.transAxes,fontsize=8,va='top',fontfamily='monospace')
#     fig.suptitle(f'{exp.upper()} — Best Layers: Old vs New Method',fontsize=14,fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,0.96]); pdf.savefig(fig,dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # A4: Hint Direction (V8 - V7)
# # ══════════════════════════════════════════════════════════════════════

# def pg_a4_hint(pdf, all_data, exp, top_layers):
#     v7, v8 = all_data['v7'], all_data['v8']
#     common = [k for k in v7 if k in v8 and k[0]==exp]
#     if not common or not top_layers: return
#     n_layers = v7[common[0]]['meta']['n_decoder_layers']
#     fig, axes = plt.subplots(len(common),1,figsize=(16,5*len(common)),squeeze=False)
#     for ki,key in enumerate(sorted(common)):
#         ax=axes[ki,0]; delta=np.zeros(n_layers); n_q=0
#         for qi,qr7 in enumerate(v7[key]['results']):
#             if qi>=len(v8[key]['results']): break
#             qr8=v8[key]['results'][qi]; q=qr7['question']; n_q+=1
#             for l in range(n_layers):
#                 if q['type']=='mcq':
#                     e7=abs(steering_eff_mcq(qr7,q,l,10)); e8=abs(steering_eff_mcq(qr8,q,l,10))
#                 else:
#                     e7=abs(steering_eff_num(qr7,l,10)); e8=abs(steering_eff_num(qr8,l,10))
#                 delta[l]+=(e8-e7)
#         delta/=max(n_q,1)
#         colors=['#4CAF50' if d>0 else '#FF5722' for d in delta]
#         ax.bar(range(n_layers),delta,color=colors,alpha=0.7,edgecolor='black',lw=0.3)
#         ax.axhline(0,color='black',lw=0.8)
#         for l in top_layers:
#             if l<n_layers: ax.axvline(l,color='blue',ls=':',alpha=0.3)
#         ax.set_title(f'cfg{key[1]} ({config_label(exp,key[1])}) — V8-V7',fontweight='bold')
#         ax.set_ylabel('Delta |effect|'); ax.grid(True,alpha=0.1,axis='y')
#     axes[-1,0].set_xlabel('Layer')
#     fig.suptitle(f'{exp.upper()} — Hint Direction: V8 - V7\n'
#                  f'Green = hint helped, Red = hurt',fontsize=13,fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig,dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # B1a: Raw Probability Heatmaps (old + new method)
# # ══════════════════════════════════════════════════════════════════════

# def pg_b1a_raw(pdf, qr, q, meta, exp):
#     it, cfg = meta['iter'], meta['cfg']
#     n_layers = meta['n_decoder_layers']
#     coeffs = meta['old_coeffs']
#     nc = len(coeffs)
#     is_mcq = q['type'] == 'mcq'
#     tokens = MCQ_TOKENS if is_mcq else DIGIT_TOKENS
#     nt = len(tokens)
#     n_new = len(NEW_CONDS)

#     # Build all matrices
#     old_mats, new_mats = {}, {}
#     for tok in tokens:
#         m = np.zeros((n_layers, nc))
#         for ci, c in enumerate(coeffs):
#             for l in range(n_layers):
#                 m[l, ci] = get_old_probs(qr, l, c, tok)
#         old_mats[tok] = m
#         m2 = np.zeros((n_layers, n_new))
#         for ci2, cond in enumerate(NEW_CONDS):
#             for l in range(n_layers):
#                 m2[l, ci2] = get_new_probs(qr, l, cond, tok)
#         new_mats[tok] = m2
#     gv = max(max(m.max() for m in old_mats.values()),
#              max(m.max() for m in new_mats.values()))
#     gv = max(gv, 0.01)

#     if is_mcq:
#         fig, axes = plt.subplots(2, 3, figsize=(20, 14), squeeze=False)
#         for ti, tok in enumerate(tokens):
#             # Row 1: old method
#             ax = axes[0, ti]
#             im = ax.imshow(old_mats[tok], aspect='auto', origin='lower',
#                            cmap='YlOrRd', vmin=0, vmax=gv)
#             ax.set_xticks(range(nc))
#             ax.set_xticklabels([str(c) for c in coeffs], fontsize=7)
#             ax.set_title(f'Old: P({tok})', fontweight='bold', fontsize=10)
#             if ti == 0: ax.set_ylabel('Layer')
#             plt.colorbar(im, ax=ax, shrink=0.5)
#             # Row 2: new method
#             ax2 = axes[1, ti]
#             im2 = ax2.imshow(new_mats[tok], aspect='auto', origin='lower',
#                              cmap='YlOrRd', vmin=0, vmax=gv)
#             ax2.set_xticks(range(n_new))
#             ax2.set_xticklabels([new_cond_display(exp, c) for c in NEW_CONDS],
#                                 fontsize=7)
#             ax2.set_title(f'New: P({tok})', fontweight='bold', fontsize=10)
#             if ti == 0: ax2.set_ylabel('Layer')
#             plt.colorbar(im2, ax=ax2, shrink=0.5)
#     else:
#         # Numeric: 2 pages (old, new), each 2x5
#         for method, mats_dict, xticks, xlabels, prefix in [
#             ('Old', old_mats, range(nc), [str(c) for c in coeffs], 'old'),
#             ('New', new_mats, range(n_new),
#              [new_cond_display(exp, c) for c in NEW_CONDS], 'new'),
#         ]:
#             fig, axes = plt.subplots(2, 5, figsize=(22, 12), squeeze=False)
#             for ti, tok in enumerate(tokens):
#                 ax = axes[ti//5, ti%5]
#                 im = ax.imshow(mats_dict[tok], aspect='auto', origin='lower',
#                                cmap='YlOrRd', vmin=0, vmax=gv)
#                 ax.set_xticks(list(xticks))
#                 ax.set_xticklabels(xlabels, fontsize=6)
#                 ax.set_title(f'{method}: P({tok})', fontweight='bold', fontsize=9)
#                 if ti%5 == 0: ax.set_ylabel('Layer')
#                 if ti >= 5: ax.set_xlabel('Coeff' if prefix=='old' else 'Condition')
#                 plt.colorbar(im, ax=ax, shrink=0.5)
#             add_prompt_box(fig, q['prompt'])
#             fig.suptitle(f'{q["id"]}: {q["description"]} — Raw P ({method} method)\n'
#                          f'{it.upper()}/cfg{cfg}', fontsize=12, fontweight='bold', y=0.98)
#             plt.tight_layout(rect=[0,0,1,0.94])
#             pdf.savefig(fig, dpi=150); plt.close()
#         return  # already saved pages for numeric

#     add_prompt_box(fig, q['prompt'])
#     fig.suptitle(f'{q["id"]}: {q["description"]} — Raw Probabilities\n'
#                  f'{it.upper()}/cfg{cfg} | Row 1=Old, Row 2=New',
#                  fontsize=12, fontweight='bold', y=0.99)
#     plt.tight_layout(rect=[0,0,1,0.95])
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # B1b: Delta Probability Heatmaps (change from neutral)
# # ══════════════════════════════════════════════════════════════════════

# def pg_b1b_delta(pdf, qr, q, meta, exp):
#     it, cfg = meta['iter'], meta['cfg']
#     n_layers = meta['n_decoder_layers']
#     coeffs = meta['old_coeffs']
#     nc = len(coeffs)
#     is_mcq = q['type'] == 'mcq'
#     tokens = MCQ_TOKENS if is_mcq else DIGIT_TOKENS
#     n_new = len(NEW_CONDS)

#     # Build delta matrices
#     old_deltas, new_deltas = {}, {}
#     gv = 0.001
#     for tok in tokens:
#         # Old: delta = P(tok|coeff) - P(tok|coeff=0)
#         neutral_col = np.array([get_old_probs(qr, l, 0, tok) for l in range(n_layers)])
#         m = np.zeros((n_layers, nc))
#         for ci, c in enumerate(coeffs):
#             for l in range(n_layers):
#                 m[l, ci] = get_old_probs(qr, l, c, tok) - neutral_col[l]
#         old_deltas[tok] = m
#         gv = max(gv, abs(m).max())

#         # New: delta = P(tok|cond) - P(tok|neutral)
#         neutral_new = np.array([get_new_probs(qr, l, 'neutral', tok)
#                                 for l in range(n_layers)])
#         m2 = np.zeros((n_layers, n_new))
#         for ci2, cond in enumerate(NEW_CONDS):
#             for l in range(n_layers):
#                 m2[l, ci2] = get_new_probs(qr, l, cond, tok) - neutral_new[l]
#         new_deltas[tok] = m2
#         gv = max(gv, abs(m2).max())

#     norm = TwoSlopeNorm(vmin=-gv, vcenter=0, vmax=gv)

#     if is_mcq:
#         fig, axes = plt.subplots(2, 3, figsize=(20, 14), squeeze=False)
#         for ti, tok in enumerate(tokens):
#             ax = axes[0, ti]
#             im = ax.imshow(old_deltas[tok], aspect='auto', origin='lower',
#                            cmap='RdBu_r', norm=norm)
#             ax.set_xticks(range(nc))
#             ax.set_xticklabels([str(c) for c in coeffs], fontsize=7)
#             ax.set_title(f'Old: dP({tok})', fontweight='bold', fontsize=10)
#             if ti == 0: ax.set_ylabel('Layer')
#             plt.colorbar(im, ax=ax, shrink=0.5, label='delta')

#             ax2 = axes[1, ti]
#             im2 = ax2.imshow(new_deltas[tok], aspect='auto', origin='lower',
#                              cmap='RdBu_r', norm=norm)
#             ax2.set_xticks(range(n_new))
#             ax2.set_xticklabels([new_cond_display(exp, c) for c in NEW_CONDS],
#                                 fontsize=7)
#             ax2.set_title(f'New: dP({tok})', fontweight='bold', fontsize=10)
#             if ti == 0: ax2.set_ylabel('Layer')
#             plt.colorbar(im2, ax=ax2, shrink=0.5, label='delta')
#     else:
#         for method, deltas, xticks, xlabels in [
#             ('Old', old_deltas, range(nc), [str(c) for c in coeffs]),
#             ('New', new_deltas, range(n_new),
#              [new_cond_display(exp, c) for c in NEW_CONDS]),
#         ]:
#             fig, axes = plt.subplots(2, 5, figsize=(22, 12), squeeze=False)
#             for ti, tok in enumerate(tokens):
#                 ax = axes[ti//5, ti%5]
#                 im = ax.imshow(deltas[tok], aspect='auto', origin='lower',
#                                cmap='RdBu_r', norm=norm)
#                 ax.set_xticks(list(xticks))
#                 ax.set_xticklabels(xlabels, fontsize=6)
#                 ax.set_title(f'{method}: dP({tok})', fontweight='bold', fontsize=9)
#                 if ti%5 == 0: ax.set_ylabel('Layer')
#                 plt.colorbar(im, ax=ax, shrink=0.5)
#             add_prompt_box(fig, q['prompt'])
#             fig.suptitle(f'{q["id"]}: {q["description"]} — Delta P ({method})\n'
#                          f'{it.upper()}/cfg{cfg} | blue=decrease, red=increase vs neutral',
#                          fontsize=12, fontweight='bold', y=0.98)
#             plt.tight_layout(rect=[0,0,1,0.94])
#             pdf.savefig(fig, dpi=150); plt.close()
#         return

#     add_prompt_box(fig, q['prompt'])
#     fig.suptitle(f'{q["id"]}: {q["description"]} — Delta P (vs neutral)\n'
#                  f'{it.upper()}/cfg{cfg} | blue=decrease, red=increase',
#                  fontsize=12, fontweight='bold', y=0.99)
#     plt.tight_layout(rect=[0,0,1,0.95])
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # B2: Pine Tree — MCQ Only, Target Choice % Change
# # ══════════════════════════════════════════════════════════════════════

# def pg_b2_pine_mcq(pdf, qr, q, meta, top_layers, exp):
#     if q['type'] != 'mcq':
#         return

#     it, cfg = meta['iter'], meta['cfg']
#     n_layers = meta['n_decoder_layers']
#     coeffs = meta['old_coeffs']
#     layers = np.arange(n_layers)
#     neg_c, pos_c = min(coeffs), max(coeffs)
#     cl = COND_LABELS[exp]

#     tgt = q.get('target_option_pos', 'C')
#     neutral = [get_old_probs(qr, l, 0, tgt) for l in range(n_layers)]

#     # Old method: % change at ±10
#     d_neg = [pct_change(get_old_probs(qr, l, neg_c, tgt), neutral[l])
#              for l in range(n_layers)]
#     d_pos = [pct_change(get_old_probs(qr, l, pos_c, tgt), neutral[l])
#              for l in range(n_layers)]
#     # New method: % change vs neutral
#     d_vpos = [pct_change(get_new_probs(qr, l, 'employed', tgt),
#                          get_new_probs(qr, l, 'neutral', tgt))
#               for l in range(n_layers)]
#     d_vneg = [pct_change(get_new_probs(qr, l, 'unemployed', tgt),
#                          get_new_probs(qr, l, 'neutral', tgt))
#               for l in range(n_layers)]

#     fig, ax = plt.subplots(figsize=(10, 16))
#     bh = 0.35

#     # Old method: solid bars
#     ax.barh(layers + bh/2, d_neg, bh, color='#1565C0', alpha=0.7,
#             edgecolor='black', lw=0.3, label=f'Old c={neg_c}')
#     ax.barh(layers + bh/2, d_pos, bh, color='#C62828', alpha=0.7,
#             edgecolor='black', lw=0.3, label=f'Old c={pos_c}')

#     # New method: hatched bars
#     ax.barh(layers - bh/2, d_vneg, bh, color='#1565C0', alpha=0.4,
#             edgecolor='#1565C0', lw=0.8, hatch='///',
#             label=f'New {cl["neg"]}')
#     ax.barh(layers - bh/2, d_vpos, bh, color='#C62828', alpha=0.4,
#             edgecolor='#C62828', lw=0.8, hatch='\\\\\\',
#             label=f'New {cl["pos"]}')

#     ax.axvline(0, color='black', lw=1.2)
#     for l in top_layers:
#         if l < n_layers:
#             ax.axhline(l, color='gold', ls='-', lw=1.5, alpha=0.5)

#     ax.set_xlim(-1, 1)
#     ax.set_ylabel('Layer')
#     ax.set_xlabel(f'% change P({tgt}) relative to neutral')
#     ax.set_ylim(-1, n_layers)
#     ax.legend(fontsize=8, loc='lower right')
#     ax.grid(True, alpha=0.08, axis='x')
#     ax.invert_yaxis()

#     add_prompt_box(fig, q['prompt'])
#     fig.suptitle(f'{q["id"]}: {q["description"]} — Pine Tree (target={tgt})\n'
#                  f'{it.upper()}/cfg{cfg} | solid=old, hatched=new',
#                  fontsize=11, fontweight='bold', y=0.99)
#     plt.tight_layout(rect=[0,0,1,0.97])
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # C1/C2: Quantitative Tables (paginated)
# # ══════════════════════════════════════════════════════════════════════

# def pg_c_tables(pdf, run_data, meta, top_layers, exp):
#     it = meta['iter']
#     cfg = meta['cfg']
#     n_layers = meta['n_decoder_layers']
#     show = sorted([l for l in top_layers if l < n_layers])[:6]
#     if not show:
#         show = [n_layers // 2]
#     cl = COND_LABELS[exp]

#     results = run_data['results']
#     chunk_size = 10
#     for chunk_start in range(0, len(results), chunk_size):
#         chunk = results[chunk_start:chunk_start + chunk_size]

#         fig, ax = plt.subplots(figsize=(22, 14))
#         ax.axis('off')
#         lines = [
#             f'{exp.upper()} — Quantitative Results at Top Layers '
#             f'(Q{chunk_start+1}-{chunk_start+len(chunk)})\n',
#             f'{it.upper()} cfg{cfg} ({config_label(exp, cfg)})\n',
#             '='*120 + '\n\n']

#         for qr in chunk:
#             q = qr['question']
#             is_mcq = q['type'] == 'mcq'
#             lines.append(f'{q["id"]}: {q["description"]}\n')

#             if is_mcq:
#                 tokens = MCQ_TOKENS
#                 hdr = f'  {"Lyr":>4s}'
#                 for tok in tokens:
#                     hdr += (f'  neut_{tok} c-10_{tok} c+10_{tok}'
#                             f'  {cl["pos"][:4]}_{tok} {cl["neg"][:4]}_{tok}')
#                 lines.append(hdr + '\n')
#                 lines.append('  ' + '-'*110 + '\n')
#                 for l in show:
#                     row = f'  L{l:>3d}'
#                     for tok in tokens:
#                         pn = get_old_probs(qr, l, 0, tok)
#                         pm = get_old_probs(qr, l, -10, tok)
#                         pp = get_old_probs(qr, l, 10, tok)
#                         pe = get_new_probs(qr, l, 'employed', tok)
#                         pu = get_new_probs(qr, l, 'unemployed', tok)
#                         row += f'  {pn:5.3f} {pm:5.3f} {pp:5.3f} {pe:5.3f} {pu:5.3f}'
#                     lines.append(row + '\n')
#             else:
#                 hdr = (f'  {"Lyr":>4s}  {"neut_E":>7s} {"c-10_E":>7s} '
#                        f'{"c+10_E":>7s} {cl["pos"][:4]+"_E":>7s} '
#                        f'{cl["neg"][:4]+"_E":>7s} '
#                        f'{"d_old":>8s} {"d_new":>8s}')
#                 lines.append(hdr + '\n')
#                 lines.append('  ' + '-'*70 + '\n')
#                 for l in show:
#                     en = expected_digit({d: get_old_probs(qr, l, 0, d) for d in DIGIT_TOKENS})
#                     em = expected_digit({d: get_old_probs(qr, l, -10, d) for d in DIGIT_TOKENS})
#                     ep = expected_digit({d: get_old_probs(qr, l, 10, d) for d in DIGIT_TOKENS})
#                     ee = expected_digit({d: get_new_probs(qr, l, 'employed', d) for d in DIGIT_TOKENS})
#                     eu = expected_digit({d: get_new_probs(qr, l, 'unemployed', d) for d in DIGIT_TOKENS})
#                     lines.append(
#                         f'  L{l:>3d}  {en:7.3f} {em:7.3f} {ep:7.3f} '
#                         f'{ee:7.3f} {eu:7.3f} {ep-em:>+8.3f} {ee-eu:>+8.3f}\n')
#             lines.append('\n')

#         ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
#                 fontsize=6.5, va='top', fontfamily='monospace')
#         plt.tight_layout()
#         pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # C3: Cross-Question Signal Consistency
# # ══════════════════════════════════════════════════════════════════════

# def pg_c3_consistency(pdf, all_data, exp, top_layers):
#     combos = []
#     for it in ['v7', 'v8']:
#         for key, run in sorted(all_data[it].items()):
#             if key[0] == exp:
#                 combos.append((it, key[1], run))
#     if not combos:
#         return
#     n_layers = combos[0][2]['meta']['n_decoder_layers']
#     threshold = 0.01

#     fig, ax = plt.subplots(figsize=(20, 14)); ax.axis('off')
#     lines = [f'{exp.upper()} — Cross-Question Signal Consistency\n',
#              f'Threshold: |effect| > {threshold}\n', '='*110+'\n\n']

#     for it, cfg, run in combos:
#         n_q = len(run['results'])
#         lines.append(f'{it.upper()} cfg{cfg} ({config_label(exp,cfg)})\n')
#         hdr = (f'  {"Lyr":>4s}  {"#old>thr":>8s}/{n_q:>2d}  '
#                f'{"#new>thr":>8s}/{n_q:>2d}  '
#                f'{"old_mean":>9s}  {"new_mean":>9s}  {"verdict":>10s}\n')
#         lines.append(hdr); lines.append('  '+'-'*80+'\n')
#         scores = []
#         for l in range(n_layers):
#             no, nn, os_, ns_ = 0, 0, 0, 0
#             for qr in run['results']:
#                 q = qr['question']
#                 if q['type']=='mcq':
#                     eo=(abs(steering_eff_mcq(qr,q,l,10))+abs(steering_eff_mcq(qr,q,l,-10)))/2
#                     en=new_eff_mcq(qr,q,l)
#                 else:
#                     eo=(abs(steering_eff_num(qr,l,10))+abs(steering_eff_num(qr,l,-10)))/2
#                     en=new_eff_num(qr,l)
#                 os_+=eo; ns_+=en
#                 if eo>threshold: no+=1
#                 if en>threshold: nn+=1
#             scores.append((l, no, nn, os_/n_q, ns_/n_q))
#         scores.sort(key=lambda x: x[3]+x[4], reverse=True)
#         for l, no, nn, om, nm in scores[:15]:
#             if no>=n_q*0.7 and nn>=n_q*0.7: verdict='STRONG'
#             elif no>=n_q*0.5 or nn>=n_q*0.5: verdict='moderate'
#             else: verdict='weak'
#             marker=' ***' if l in top_layers else ''
#             lines.append(f'  L{l:>3d}  {no:>8d}/{n_q:>2d}  {nn:>8d}/{n_q:>2d}  '
#                          f'{om:>9.4f}  {nm:>9.4f}  {verdict:>10s}{marker}\n')
#         lines.append('\n')
#     lines.append('-'*80+'\n*** = in top layers from Section A\n'
#                  'STRONG = >70% questions for BOTH methods\n')
#     ax.text(0.01,0.99,''.join(lines),transform=ax.transAxes,fontsize=7,va='top',fontfamily='monospace')
#     fig.suptitle(f'{exp.upper()} — Which Layers Show Consistent Signal?',fontsize=14,fontweight='bold')
#     plt.tight_layout(rect=[0,0,1,0.96]); pdf.savefig(fig,dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # Pre-compute globals
# # ══════════════════════════════════════════════════════════════════════

# def compute_old_top_layers(all_data, exp):
#     top_all = set()
#     for it in ['v7', 'v8']:
#         for key, run in all_data[it].items():
#             if key[0] != exp: continue
#             n_layers = run['meta']['n_decoder_layers']
#             coeffs = run['meta']['old_coeffs']
#             eff = np.zeros(n_layers); cnt = 0
#             for qr in run['results']:
#                 q = qr['question']
#                 for l in range(n_layers):
#                     for c in coeffs:
#                         if q['type']=='mcq': eff[l]+=abs(steering_eff_mcq(qr,q,l,c))
#                         else: eff[l]+=abs(steering_eff_num(qr,l,c))
#                 cnt += 1
#             eff /= max(cnt*len(coeffs), 1)
#             top_all.update(np.argsort(eff)[-5:][::-1].tolist())
#     return sorted(top_all)


# # ══════════════════════════════════════════════════════════════════════
# # Main
# # ══════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--exp', nargs='+', default=['exp2', 'exp1a'])
#     parser.add_argument('--model', default='12b',
#                         choices=list(MODEL_REGISTRY.keys()))
#     args = parser.parse_args()

#     if args.model == '12b':
#         base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         out_dir = os.path.join(base, 'steering', 'output')
#     else:
#         model_base = MODEL_REGISTRY[args.model]['hf_name']
#         out_dir = os.path.join(model_base, 'steering', 'output')
#     os.makedirs(out_dir, exist_ok=True)

#     all_data = load_all(args.model)

#     for exp in args.exp:
#         questions = ALL_QUESTIONS.get(exp, [])
#         if not questions: continue

#         pdf_path = os.path.join(out_dir, f'nl_steering_{exp}_v4.pdf')
#         print(f"\nGenerating: {pdf_path}")

#         print(f"  Computing global scales...")
#         top_old = compute_old_top_layers(all_data, exp)
#         print(f"    Old top layers: {top_old}")

#         with PdfPages(pdf_path) as pdf:
#             # ── Section A ──
#             print(f"  A1: Old method heatmap")
#             pg_a1_old_heatmap(pdf, all_data, exp)

#             print(f"  A2: New method layer effectiveness")
#             top_new = pg_a2_new_layer(pdf, all_data, exp)
#             print(f"    New top layers: {top_new}")

#             top_all = sorted(set(top_old) | set(top_new))
#             print(f"    Combined: {top_all}")

#             print(f"  A3: Best layers summary")
#             pg_a3_summary(pdf, all_data, exp, top_old, top_new)

#             print(f"  A4: Hint direction")
#             pg_a4_hint(pdf, all_data, exp, top_all)

#             # ── Section B ──
#             print(f"  B: Per-question detail")
#             for it in ['v7', 'v8']:
#                 for key, run in sorted(all_data[it].items()):
#                     if key[0] != exp: continue
#                     meta = run['meta']; meta['iter'] = it
#                     for qr in run['results']:
#                         q = qr['question']
#                         print(f"    {it}/cfg{key[1]} {q['id']}")
#                         pg_b1a_raw(pdf, qr, q, meta, exp)
#                         pg_b1b_delta(pdf, qr, q, meta, exp)
#                         pg_b2_pine_mcq(pdf, qr, q, meta, top_all, exp)

#             # ── Section C ──
#             print(f"  C: Quantitative tables")
#             for it in ['v7', 'v8']:
#                 for key, run in sorted(all_data[it].items()):
#                     if key[0] != exp: continue
#                     meta = run['meta']; meta['iter'] = it
#                     print(f"    {it}/cfg{key[1]}: tables")
#                     pg_c_tables(pdf, run, meta, top_all, exp)

#             print(f"  C3: Cross-question consistency")
#             pg_c3_consistency(pdf, all_data, exp, top_all)

#         print(f"  Done! -> {pdf_path}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze NL-based steering results v4.2 (Apr 17 revision).

Changes from v4.1:
  - Heatmaps: Raw P fixed [0,1], Delta P fixed [-1,1], colorbar ticks 0.02
  - Pine tree: x-axis fixed [-1,1] with 0.02 ticks, y-axis L0 bottom to Lmax top
  - Pine tree split: old method and new method on separate pages

Usage:
    python steering/analyze_steering_nl.py --model gemma4_31b
"""
import json, os, sys, glob, argparse, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import config_label, get_iter_output_dir, MODEL_REGISTRY
from steering.prompts_nl import ALL_QUESTIONS, MCQ_TOKENS, DIGIT_TOKENS

COND_LABELS = {
    'exp2':  {'employed': 'employed',  'unemployed': 'unemployed',
              'pos': '+V_emp', 'neg': '+V_unemp'},
    'exp1a': {'employed': 'weekday',   'unemployed': 'weekend',
              'pos': '+V_weekday', 'neg': '+V_weekend'},
}
NEW_CONDS = ['neutral', 'employed', 'unemployed']

RAW_CMAP = 'inferno'
DELTA_CMAP = 'coolwarm'

# Fixed scales
RAW_VMIN, RAW_VMAX = 0.0, 1.0
DELTA_VMIN, DELTA_VMAX = -1.0, 1.0
TICK_STEP = 0.02
PINE_XLIM = (-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_iter(iter_name, model_key):
    out_base = get_iter_output_dir(model_key, iter_name)
    nl_dir = os.path.join(out_base, 'steering', 'nl_results')
    if not os.path.exists(nl_dir):
        return {}
    data = {}
    for f in sorted(glob.glob(os.path.join(nl_dir, '*_nl.json'))):
        with open(f) as fh:
            d = json.load(fh)
        m = d['meta']
        data[(m['exp'], m['cfg'])] = d
        print(f"  {iter_name}/{m['exp']}/cfg{m['cfg']}: "
              f"{len(d['results'])} questions")
    return data

def load_all(model_key):
    print(f"Loading NL steering results for {model_key}...")
    return {it: load_iter(it, model_key) for it in ['v7', 'v8']}


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def get_old_probs(qr, layer, coeff, token):
    for e in qr['old_method']:
        if e['layer'] == layer and e['coeff'] == coeff:
            return e['probs'].get(token, 0)
    return 0

def get_new_probs(qr, layer, cond, token):
    for e in qr['new_method']:
        if e['layer'] == layer:
            return e[cond].get(token, 0)
    return 0

def expected_digit(pd):
    return sum(int(d) * pd.get(d, 0) for d in DIGIT_TOKENS)

def pct_change(steered, neutral):
    if abs(neutral) < 1e-10:
        return 0.0
    return (steered - neutral) / neutral

def steering_eff_mcq(qr, q, layer, coeff):
    tgt = q.get('target_option_pos')
    if not tgt: return 0
    return get_old_probs(qr, layer, coeff, tgt) - get_old_probs(qr, layer, 0, tgt)

def steering_eff_num(qr, layer, coeff):
    e_s = expected_digit({d: get_old_probs(qr, layer, coeff, d) for d in DIGIT_TOKENS})
    e_n = expected_digit({d: get_old_probs(qr, layer, 0, d) for d in DIGIT_TOKENS})
    return e_s - e_n

def new_eff_mcq(qr, q, layer):
    tgt = q.get('target_option_pos')
    if not tgt: return 0
    return abs(get_new_probs(qr, layer, 'employed', tgt) -
               get_new_probs(qr, layer, 'unemployed', tgt))

def new_eff_num(qr, layer):
    e_e = expected_digit({d: get_new_probs(qr, layer, 'employed', d) for d in DIGIT_TOKENS})
    e_u = expected_digit({d: get_new_probs(qr, layer, 'unemployed', d) for d in DIGIT_TOKENS})
    return abs(e_e - e_u)

def new_cond_display(exp, cond):
    m = COND_LABELS.get(exp, COND_LABELS['exp2'])
    if cond == 'neutral': return 'neutral'
    if cond == 'employed': return m['pos']
    if cond == 'unemployed': return m['neg']
    return cond

def add_prompt_box(fig, prompt_str):
    wrapped = textwrap.fill(prompt_str.replace('\n', ' | '), width=80)
    fig.text(0.99, 0.99, wrapped, transform=fig.transFigure,
             fontsize=5, va='top', ha='right', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#cccccc',
                       alpha=0.9))


def _dual_y_layers(ax, n_layers):
    tick_step = 5
    ticks = list(range(0, n_layers, tick_step))
    labels = [f'L{t}' for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6)
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels, fontsize=6)
    return ax2


def _add_colorbar_02(fig, im, ax, label, vmin, vmax):
    """Add colorbar with 0.02 tick granularity."""
    cb = plt.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label(label, fontsize=8)
    # Use 0.1 major ticks to avoid overcrowding, 0.02 minor
    major_ticks = np.arange(vmin, vmax + 0.01, 0.1)
    minor_ticks = np.arange(vmin, vmax + 0.01, TICK_STEP)
    cb.set_ticks(major_ticks)
    cb.ax.yaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    cb.ax.tick_params(labelsize=6)
    return cb


# ══════════════════════════════════════════════════════════════════════
# A1: Old Method — Layer x Coeff Heatmap
# ══════════════════════════════════════════════════════════════════════

def pg_a1_old_heatmap(pdf, all_data, exp):
    combos = []
    for it in ['v7', 'v8']:
        for key, run in sorted(all_data[it].items()):
            if key[0] == exp: combos.append((it, key[1], run))
    if not combos: return

    n_layers = combos[0][2]['meta']['n_decoder_layers']
    coeffs = combos[0][2]['meta']['old_coeffs']
    nc = len(coeffs)

    mats = []
    for it, cfg, run in combos:
        mat = np.zeros((n_layers, nc))
        for qr in run['results']:
            q = qr['question']
            for ci, c in enumerate(coeffs):
                for l in range(n_layers):
                    if q['type'] == 'mcq': mat[l,ci] += abs(steering_eff_mcq(qr,q,l,c))
                    else: mat[l,ci] += abs(steering_eff_num(qr,l,c))
        mat /= len(run['results'])
        mats.append(mat)

    ncols = min(len(combos), 4)
    nrows = (len(combos)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 10*nrows), squeeze=False)
    norm = Normalize(vmin=RAW_VMIN, vmax=RAW_VMAX)
    for idx, (it, cfg, run) in enumerate(combos):
        ax = axes[idx//ncols, idx%ncols]
        im = ax.imshow(mats[idx], aspect='auto', origin='lower', cmap=RAW_CMAP, norm=norm)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([str(c) for c in coeffs], fontsize=8)
        ax.set_xlabel('Coefficient')
        _dual_y_layers(ax, n_layers)
        ax.set_title(f'{it.upper()} cfg{cfg}\n({config_label(exp,cfg)})',
                     fontweight='bold', fontsize=9)
        _add_colorbar_02(fig, im, ax, 'Mean |effect|', RAW_VMIN, RAW_VMAX)
    for idx in range(len(combos), nrows*ncols):
        axes[idx//ncols, idx%ncols].set_visible(False)
    fig.suptitle(f'{exp.upper()} — Old Method: Mean |Steering Effect|',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# A2: New Method — Layer Effectiveness
# ══════════════════════════════════════════════════════════════════════

def pg_a2_new_layer(pdf, all_data, exp):
    combos = []
    for it in ['v7', 'v8']:
        for key, run in sorted(all_data[it].items()):
            if key[0] == exp: combos.append((it, key[1], run))
    if not combos: return []
    n_layers = combos[0][2]['meta']['n_decoder_layers']
    cl = COND_LABELS[exp]
    effs = []
    for it, cfg, run in combos:
        eff = np.zeros(n_layers)
        for qr in run['results']:
            q = qr['question']
            for l in range(n_layers):
                if q['type']=='mcq': eff[l]+=new_eff_mcq(qr,q,l)
                else: eff[l]+=new_eff_num(qr,l)
        eff /= len(run['results']); effs.append(eff)
    gym = max(e.max() for e in effs)
    fig, axes = plt.subplots(len(combos),1,figsize=(16,3*len(combos)+2),squeeze=False)
    top_all = set()
    for idx,(it,cfg,run) in enumerate(combos):
        ax=axes[idx,0]; eff=effs[idx]
        top5=np.argsort(eff)[-5:][::-1]; top_all.update(top5.tolist())
        ax.bar(range(n_layers),eff,color='#2196F3' if it=='v7' else '#FF5722',
               alpha=0.7,edgecolor='black',lw=0.3)
        for l in top5:
            ax.bar(l,eff[l],color='gold',edgecolor='black',lw=0.8)
            ax.text(l,eff[l]+gym*0.02,f'L{l}',ha='center',fontsize=6,fontweight='bold')
        ax.set_ylabel(f'Mean |{cl["pos"]}-{cl["neg"]}|')
        ax.set_ylim(0,gym*1.15)
        ax.set_title(f'{it.upper()} cfg{cfg} ({config_label(exp,cfg)})',fontweight='bold',fontsize=10)
        ax.grid(True,alpha=0.1,axis='y'); ax.set_xlim(-0.5,n_layers-0.5)
    axes[-1,0].set_xlabel('Layer')
    fig.suptitle(f'{exp.upper()} — New Method: |{cl["pos"]} - {cl["neg"]}| per layer',
                 fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig,dpi=150); plt.close()
    return sorted(top_all)


# ══════════════════════════════════════════════════════════════════════
# A3: Best Layers Summary Table
# ══════════════════════════════════════════════════════════════════════

def pg_a3_summary(pdf, all_data, exp, top_old, top_new):
    combos = []
    for it in ['v7','v8']:
        for key,run in sorted(all_data[it].items()):
            if key[0]==exp: combos.append((it,key[1],run))
    if not combos: return
    n_layers=combos[0][2]['meta']['n_decoder_layers']
    coeffs=combos[0][2]['meta']['old_coeffs']
    fig,ax=plt.subplots(figsize=(18,10)); ax.axis('off')
    lines=[f'{exp.upper()} — Best Layers Summary\n','='*100+'\n\n']
    for it,cfg,run in combos:
        old_eff=np.zeros(n_layers); new_eff=np.zeros(n_layers); n_q=len(run['results'])
        for qr in run['results']:
            q=qr['question']
            for l in range(n_layers):
                for c in coeffs:
                    if q['type']=='mcq': old_eff[l]+=abs(steering_eff_mcq(qr,q,l,c))
                    else: old_eff[l]+=abs(steering_eff_num(qr,l,c))
                if q['type']=='mcq': new_eff[l]+=new_eff_mcq(qr,q,l)
                else: new_eff[l]+=new_eff_num(qr,l)
        old_eff/=(n_q*len(coeffs)); new_eff/=n_q
        ot5=np.argsort(old_eff)[-5:][::-1]; nt5=np.argsort(new_eff)[-5:][::-1]
        both=sorted(set(ot5)&set(nt5))
        lines.append(f'{it.upper()} cfg{cfg} ({config_label(exp,cfg)})\n')
        lines.append(f'  Old top-5: '+', '.join(f'L{l}({old_eff[l]:.4f})' for l in ot5)+'\n')
        lines.append(f'  New top-5: '+', '.join(f'L{l}({new_eff[l]:.4f})' for l in nt5)+'\n')
        lines.append(f'  Intersect: '+(', '.join(f'L{l}' for l in both) if both else 'NONE')+'\n\n')
    lines.append('-'*100+'\n')
    bg=sorted(set(top_old)&set(top_new))
    lines.append(f'Global old top:  {", ".join(f"L{l}" for l in top_old)}\n')
    lines.append(f'Global new top:  {", ".join(f"L{l}" for l in top_new)}\n')
    lines.append(f'Global intersect: '+(', '.join(f'L{l}' for l in bg) if bg else 'NONE')+'\n')
    ax.text(0.02,0.98,''.join(lines),transform=ax.transAxes,fontsize=8,va='top',fontfamily='monospace')
    fig.suptitle(f'{exp.upper()} — Best Layers: Old vs New Method',fontsize=14,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); pdf.savefig(fig,dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# A4: Hint Direction (V8 - V7)
# ══════════════════════════════════════════════════════════════════════

def pg_a4_hint(pdf, all_data, exp, top_layers):
    v7,v8=all_data['v7'],all_data['v8']
    common=[k for k in v7 if k in v8 and k[0]==exp]
    if not common or not top_layers: return
    n_layers=v7[common[0]]['meta']['n_decoder_layers']
    fig,axes=plt.subplots(len(common),1,figsize=(16,5*len(common)),squeeze=False)
    for ki,key in enumerate(sorted(common)):
        ax=axes[ki,0]; delta=np.zeros(n_layers); n_q=0
        for qi,qr7 in enumerate(v7[key]['results']):
            if qi>=len(v8[key]['results']): break
            qr8=v8[key]['results'][qi]; q=qr7['question']; n_q+=1
            for l in range(n_layers):
                if q['type']=='mcq':
                    e7=abs(steering_eff_mcq(qr7,q,l,10)); e8=abs(steering_eff_mcq(qr8,q,l,10))
                else:
                    e7=abs(steering_eff_num(qr7,l,10)); e8=abs(steering_eff_num(qr8,l,10))
                delta[l]+=(e8-e7)
        delta/=max(n_q,1)
        colors=['#4CAF50' if d>0 else '#FF5722' for d in delta]
        ax.bar(range(n_layers),delta,color=colors,alpha=0.7,edgecolor='black',lw=0.3)
        ax.axhline(0,color='black',lw=0.8)
        for l in top_layers:
            if l<n_layers: ax.axvline(l,color='blue',ls=':',alpha=0.3)
        ax.set_title(f'cfg{key[1]} ({config_label(exp,key[1])}) — V8-V7',fontweight='bold')
        ax.set_ylabel('Delta |effect|'); ax.grid(True,alpha=0.1,axis='y')
    axes[-1,0].set_xlabel('Layer')
    fig.suptitle(f'{exp.upper()} — Hint Direction: V8 - V7\n'
                 f'Green = hint helped, Red = hurt',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig,dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# B1a: Raw Probability Heatmaps (fixed 0-1 scale)
# ══════════════════════════════════════════════════════════════════════

def _plot_heatmap_grid(fig, axes_list, mats_dict, tokens, xticks, xlabels,
                       n_layers, vmin, vmax, cmap, title_prefix, is_delta=False):
    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if is_delta
            else Normalize(vmin=vmin, vmax=vmax))
    label = 'delta P' if is_delta else 'P'
    for ti, tok in enumerate(tokens):
        ax = axes_list[ti]
        im = ax.imshow(mats_dict[tok], aspect='auto', origin='lower',
                       cmap=cmap, norm=norm)
        ax.set_xticks(list(xticks))
        ax.set_xticklabels(xlabels, fontsize=7)
        _dual_y_layers(ax, n_layers)
        ax.set_title(f'{title_prefix}({tok})', fontweight='bold', fontsize=10)
        _add_colorbar_02(fig, im, ax, label, vmin, vmax)


def pg_b1a_raw(pdf, qr, q, meta, exp):
    it,cfg=meta['iter'],meta['cfg']
    n_layers=meta['n_decoder_layers']
    coeffs=meta['old_coeffs']; nc=len(coeffs)
    is_mcq=q['type']=='mcq'
    tokens=MCQ_TOKENS if is_mcq else DIGIT_TOKENS
    n_new=len(NEW_CONDS)

    old_mats,new_mats={},{}
    for tok in tokens:
        m=np.zeros((n_layers,nc))
        for ci,c in enumerate(coeffs):
            for l in range(n_layers): m[l,ci]=get_old_probs(qr,l,c,tok)
        old_mats[tok]=m
        m2=np.zeros((n_layers,n_new))
        for ci2,cond in enumerate(NEW_CONDS):
            for l in range(n_layers): m2[l,ci2]=get_new_probs(qr,l,cond,tok)
        new_mats[tok]=m2

    new_xl=[new_cond_display(exp,c) for c in NEW_CONDS]
    old_xl=[str(c) for c in coeffs]

    if is_mcq:
        fig,axes=plt.subplots(2,3,figsize=(20,14),squeeze=False)
        _plot_heatmap_grid(fig,[axes[0,ti] for ti in range(3)],
                           old_mats,tokens,range(nc),old_xl,n_layers,
                           RAW_VMIN,RAW_VMAX,RAW_CMAP,'Old: P')
        _plot_heatmap_grid(fig,[axes[1,ti] for ti in range(3)],
                           new_mats,tokens,range(n_new),new_xl,n_layers,
                           RAW_VMIN,RAW_VMAX,RAW_CMAP,'New: P')
        add_prompt_box(fig,q['prompt'])
        fig.suptitle(f'{q["id"]}: {q["description"]} — Raw P\n'
                     f'{it.upper()}/cfg{cfg} | Row1=Old, Row2=New',
                     fontsize=12,fontweight='bold',y=0.99)
        plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig,dpi=150); plt.close()
    else:
        for method,mats,xticks,xlabels in [
            ('Old',old_mats,range(nc),old_xl),
            ('New',new_mats,range(n_new),new_xl),
        ]:
            fig,axes=plt.subplots(2,5,figsize=(22,12),squeeze=False)
            _plot_heatmap_grid(fig,[axes[ti//5,ti%5] for ti in range(len(tokens))],
                               mats,tokens,xticks,xlabels,n_layers,
                               RAW_VMIN,RAW_VMAX,RAW_CMAP,f'{method}: P')
            add_prompt_box(fig,q['prompt'])
            fig.suptitle(f'{q["id"]}: {q["description"]} — Raw P ({method})\n'
                         f'{it.upper()}/cfg{cfg}',fontsize=12,fontweight='bold',y=0.98)
            plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig,dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# B1b: Delta Probability Heatmaps (fixed -1 to 1 scale)
# ══════════════════════════════════════════════════════════════════════

def pg_b1b_delta(pdf, qr, q, meta, exp):
    it,cfg=meta['iter'],meta['cfg']
    n_layers=meta['n_decoder_layers']
    coeffs=meta['old_coeffs']; nc=len(coeffs)
    is_mcq=q['type']=='mcq'
    tokens=MCQ_TOKENS if is_mcq else DIGIT_TOKENS
    n_new=len(NEW_CONDS)

    old_deltas,new_deltas={},{}
    for tok in tokens:
        neutral_old=np.array([get_old_probs(qr,l,0,tok) for l in range(n_layers)])
        m=np.zeros((n_layers,nc))
        for ci,c in enumerate(coeffs):
            for l in range(n_layers): m[l,ci]=get_old_probs(qr,l,c,tok)-neutral_old[l]
        old_deltas[tok]=m
        neutral_new=np.array([get_new_probs(qr,l,'neutral',tok) for l in range(n_layers)])
        m2=np.zeros((n_layers,n_new))
        for ci2,cond in enumerate(NEW_CONDS):
            for l in range(n_layers): m2[l,ci2]=get_new_probs(qr,l,cond,tok)-neutral_new[l]
        new_deltas[tok]=m2

    new_xl=[new_cond_display(exp,c) for c in NEW_CONDS]
    old_xl=[str(c) for c in coeffs]

    if is_mcq:
        fig,axes=plt.subplots(2,3,figsize=(20,14),squeeze=False)
        _plot_heatmap_grid(fig,[axes[0,ti] for ti in range(3)],
                           old_deltas,tokens,range(nc),old_xl,n_layers,
                           DELTA_VMIN,DELTA_VMAX,DELTA_CMAP,'Old: dP',is_delta=True)
        _plot_heatmap_grid(fig,[axes[1,ti] for ti in range(3)],
                           new_deltas,tokens,range(n_new),new_xl,n_layers,
                           DELTA_VMIN,DELTA_VMAX,DELTA_CMAP,'New: dP',is_delta=True)
        add_prompt_box(fig,q['prompt'])
        fig.suptitle(f'{q["id"]}: {q["description"]} — Delta P (vs neutral)\n'
                     f'{it.upper()}/cfg{cfg} | blue=decrease, red=increase',
                     fontsize=12,fontweight='bold',y=0.99)
        plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig,dpi=150); plt.close()
    else:
        for method,deltas,xticks,xlabels in [
            ('Old',old_deltas,range(nc),old_xl),
            ('New',new_deltas,range(n_new),new_xl),
        ]:
            fig,axes=plt.subplots(2,5,figsize=(22,12),squeeze=False)
            _plot_heatmap_grid(fig,[axes[ti//5,ti%5] for ti in range(len(tokens))],
                               deltas,tokens,xticks,xlabels,n_layers,
                               DELTA_VMIN,DELTA_VMAX,DELTA_CMAP,f'{method}: dP',is_delta=True)
            add_prompt_box(fig,q['prompt'])
            fig.suptitle(f'{q["id"]}: {q["description"]} — Delta P ({method})\n'
                         f'{it.upper()}/cfg{cfg} | blue=decrease, red=increase',
                         fontsize=12,fontweight='bold',y=0.98)
            plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig,dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# B2: Pine Tree — MCQ Only, separate old/new, fixed [-1,1], L0 bottom
# ══════════════════════════════════════════════════════════════════════

def _pine_page(pdf, layers, d_left, d_right, n_layers, top_layers,
               label_left, label_right, title, xlabel, q):
    fig, ax = plt.subplots(figsize=(12, 16))
    bh = 0.6

    ax.barh(layers, d_left, bh, color='#1565C0', alpha=0.7,
            edgecolor='black', lw=0.3, label=label_left)
    ax.barh(layers, d_right, bh, color='#C62828', alpha=0.7,
            edgecolor='black', lw=0.3, label=label_right)

    ax.axvline(0, color='black', lw=1.2)
    for l in top_layers:
        if l < n_layers:
            ax.axhline(l, color='gold', ls='-', lw=1.5, alpha=0.5)

    # Fixed x-axis [-1, 1] with 0.02 ticks
    ax.set_xlim(PINE_XLIM)
    major_x = np.arange(-1.0, 1.01, 0.1)
    minor_x = np.arange(-1.0, 1.01, TICK_STEP)
    ax.set_xticks(major_x)
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_x))
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, alpha=0.08, axis='x', which='major')
    ax.grid(True, alpha=0.04, axis='x', which='minor')

    # Y-axis: L0 at bottom, Lmax at top (origin='lower' style)
    tick_step = 2
    ticks = list(range(0, n_layers, tick_step))
    labels = [f'L{t}' for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(-1, n_layers)
    # Right side labels
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels, fontsize=6)
    # Do NOT invert — L0 at bottom, Lmax at top

    ax.set_ylabel('Layer')
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=9, loc='upper right')

    add_prompt_box(fig, q['prompt'])
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0,0,1,0.97])
    pdf.savefig(fig, dpi=150); plt.close()


def pg_b2_pine_mcq(pdf, qr, q, meta, top_layers, exp):
    if q['type'] != 'mcq': return
    it,cfg=meta['iter'],meta['cfg']
    n_layers=meta['n_decoder_layers']
    coeffs=meta['old_coeffs']
    layers=np.arange(n_layers)
    neg_c,pos_c=min(coeffs),max(coeffs)
    cl=COND_LABELS[exp]
    tgt=q.get('target_option_pos','C')

    neutral_old=[get_old_probs(qr,l,0,tgt) for l in range(n_layers)]
    neutral_new=[get_new_probs(qr,l,'neutral',tgt) for l in range(n_layers)]

    d_neg=[pct_change(get_old_probs(qr,l,neg_c,tgt),neutral_old[l]) for l in range(n_layers)]
    d_pos=[pct_change(get_old_probs(qr,l,pos_c,tgt),neutral_old[l]) for l in range(n_layers)]
    d_vpos=[pct_change(get_new_probs(qr,l,'employed',tgt),neutral_new[l]) for l in range(n_layers)]
    d_vneg=[pct_change(get_new_probs(qr,l,'unemployed',tgt),neutral_new[l]) for l in range(n_layers)]

    # Clamp to [-1, 1]
    d_neg=[max(-1,min(1,v)) for v in d_neg]
    d_pos=[max(-1,min(1,v)) for v in d_pos]
    d_vpos=[max(-1,min(1,v)) for v in d_vpos]
    d_vneg=[max(-1,min(1,v)) for v in d_vneg]

    _pine_page(pdf, layers, d_neg, d_pos, n_layers, top_layers,
               f'Old c={neg_c}', f'Old c={pos_c}',
               f'{q["id"]}: {q["description"]} — Old Method (target={tgt})\n'
               f'{it.upper()}/cfg{cfg} | blue=c={neg_c}, red=c={pos_c}',
               f'% change P({tgt}) vs neutral', q)

    _pine_page(pdf, layers, d_vneg, d_vpos, n_layers, top_layers,
               f'New {cl["neg"]}', f'New {cl["pos"]}',
               f'{q["id"]}: {q["description"]} — New Method (target={tgt})\n'
               f'{it.upper()}/cfg{cfg} | blue={cl["neg"]}, red={cl["pos"]}',
               f'% change P({tgt}) vs neutral', q)


# ══════════════════════════════════════════════════════════════════════
# C1/C2: Quantitative Tables (paginated)
# ══════════════════════════════════════════════════════════════════════

def pg_c_tables(pdf, run_data, meta, top_layers, exp):
    it=meta['iter']; cfg=meta['cfg']; n_layers=meta['n_decoder_layers']
    show=sorted([l for l in top_layers if l<n_layers])[:6]
    if not show: show=[n_layers//2]
    cl=COND_LABELS[exp]; results=run_data['results']
    chunk_size=10
    for cs in range(0,len(results),chunk_size):
        chunk=results[cs:cs+chunk_size]
        fig,ax=plt.subplots(figsize=(22,14)); ax.axis('off')
        lines=[f'{exp.upper()} — Quantitative (Q{cs+1}-{cs+len(chunk)})\n',
               f'{it.upper()} cfg{cfg} ({config_label(exp,cfg)})\n','='*120+'\n\n']
        for qr in chunk:
            q=qr['question']; is_mcq=q['type']=='mcq'
            lines.append(f'{q["id"]}: {q["description"]}\n')
            if is_mcq:
                hdr=f'  {"Lyr":>4s}'
                for tok in MCQ_TOKENS:
                    hdr+=f'  neut_{tok} c-10_{tok} c+10_{tok}  {cl["pos"][:4]}_{tok} {cl["neg"][:4]}_{tok}'
                lines.append(hdr+'\n'); lines.append('  '+'-'*110+'\n')
                for l in show:
                    row=f'  L{l:>3d}'
                    for tok in MCQ_TOKENS:
                        pn=get_old_probs(qr,l,0,tok); pm=get_old_probs(qr,l,-10,tok)
                        pp=get_old_probs(qr,l,10,tok); pe=get_new_probs(qr,l,'employed',tok)
                        pu=get_new_probs(qr,l,'unemployed',tok)
                        row+=f'  {pn:5.3f} {pm:5.3f} {pp:5.3f} {pe:5.3f} {pu:5.3f}'
                    lines.append(row+'\n')
            else:
                hdr=(f'  {"Lyr":>4s}  {"neut_E":>7s} {"c-10_E":>7s} {"c+10_E":>7s} '
                     f'{cl["pos"][:4]+"_E":>7s} {cl["neg"][:4]+"_E":>7s} '
                     f'{"d_old":>8s} {"d_new":>8s}')
                lines.append(hdr+'\n'); lines.append('  '+'-'*70+'\n')
                for l in show:
                    en=expected_digit({d:get_old_probs(qr,l,0,d) for d in DIGIT_TOKENS})
                    em=expected_digit({d:get_old_probs(qr,l,-10,d) for d in DIGIT_TOKENS})
                    ep=expected_digit({d:get_old_probs(qr,l,10,d) for d in DIGIT_TOKENS})
                    ee=expected_digit({d:get_new_probs(qr,l,'employed',d) for d in DIGIT_TOKENS})
                    eu=expected_digit({d:get_new_probs(qr,l,'unemployed',d) for d in DIGIT_TOKENS})
                    lines.append(f'  L{l:>3d}  {en:7.3f} {em:7.3f} {ep:7.3f} '
                                 f'{ee:7.3f} {eu:7.3f} {ep-em:>+8.3f} {ee-eu:>+8.3f}\n')
            lines.append('\n')
        ax.text(0.01,0.99,''.join(lines),transform=ax.transAxes,
                fontsize=6.5,va='top',fontfamily='monospace')
        plt.tight_layout(); pdf.savefig(fig,dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# C3: Cross-Question Signal Consistency
# ══════════════════════════════════════════════════════════════════════

def pg_c3_consistency(pdf, all_data, exp, top_layers):
    combos=[]
    for it in ['v7','v8']:
        for key,run in sorted(all_data[it].items()):
            if key[0]==exp: combos.append((it,key[1],run))
    if not combos: return
    n_layers=combos[0][2]['meta']['n_decoder_layers']; threshold=0.01
    fig,ax=plt.subplots(figsize=(20,14)); ax.axis('off')
    lines=[f'{exp.upper()} — Cross-Question Signal Consistency\n',
           f'Threshold: |effect| > {threshold}\n','='*110+'\n\n']
    for it,cfg,run in combos:
        n_q=len(run['results'])
        lines.append(f'{it.upper()} cfg{cfg} ({config_label(exp,cfg)})\n')
        hdr=(f'  {"Lyr":>4s}  {"#old>thr":>8s}/{n_q:>2d}  '
             f'{"#new>thr":>8s}/{n_q:>2d}  '
             f'{"old_mean":>9s}  {"new_mean":>9s}  {"verdict":>10s}\n')
        lines.append(hdr); lines.append('  '+'-'*80+'\n')
        scores=[]
        for l in range(n_layers):
            no,nn,os_,ns_=0,0,0,0
            for qr in run['results']:
                q=qr['question']
                if q['type']=='mcq':
                    eo=(abs(steering_eff_mcq(qr,q,l,10))+abs(steering_eff_mcq(qr,q,l,-10)))/2
                    en=new_eff_mcq(qr,q,l)
                else:
                    eo=(abs(steering_eff_num(qr,l,10))+abs(steering_eff_num(qr,l,-10)))/2
                    en=new_eff_num(qr,l)
                os_+=eo; ns_+=en
                if eo>threshold: no+=1
                if en>threshold: nn+=1
            scores.append((l,no,nn,os_/n_q,ns_/n_q))
        scores.sort(key=lambda x:x[3]+x[4],reverse=True)
        for l,no,nn,om,nm in scores[:15]:
            if no>=n_q*0.7 and nn>=n_q*0.7: verdict='STRONG'
            elif no>=n_q*0.5 or nn>=n_q*0.5: verdict='moderate'
            else: verdict='weak'
            marker=' ***' if l in top_layers else ''
            lines.append(f'  L{l:>3d}  {no:>8d}/{n_q:>2d}  {nn:>8d}/{n_q:>2d}  '
                         f'{om:>9.4f}  {nm:>9.4f}  {verdict:>10s}{marker}\n')
        lines.append('\n')
    lines.append('-'*80+'\n*** = in top layers from Section A\n'
                 'STRONG = >70% questions for BOTH methods\n')
    ax.text(0.01,0.99,''.join(lines),transform=ax.transAxes,fontsize=7,va='top',fontfamily='monospace')
    fig.suptitle(f'{exp.upper()} — Which Layers Show Consistent Signal?',fontsize=14,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); pdf.savefig(fig,dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Pre-compute
# ══════════════════════════════════════════════════════════════════════

def compute_old_top_layers(all_data, exp):
    top_all=set()
    for it in ['v7','v8']:
        for key,run in all_data[it].items():
            if key[0]!=exp: continue
            n_layers=run['meta']['n_decoder_layers']; coeffs=run['meta']['old_coeffs']
            eff=np.zeros(n_layers); cnt=0
            for qr in run['results']:
                q=qr['question']
                for l in range(n_layers):
                    for c in coeffs:
                        if q['type']=='mcq': eff[l]+=abs(steering_eff_mcq(qr,q,l,c))
                        else: eff[l]+=abs(steering_eff_num(qr,l,c))
                cnt+=1
            eff/=max(cnt*len(coeffs),1)
            top_all.update(np.argsort(eff)[-5:][::-1].tolist())
    return sorted(top_all)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp',nargs='+',default=['exp2','exp1a'])
    parser.add_argument('--model',default='12b',choices=list(MODEL_REGISTRY.keys()))
    args=parser.parse_args()

    if args.model=='12b':
        base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir=os.path.join(base,'steering','output')
    else:
        model_base=MODEL_REGISTRY[args.model]['hf_name']
        out_dir=os.path.join(model_base,'steering','output')
    os.makedirs(out_dir,exist_ok=True)

    all_data=load_all(args.model)

    for exp in args.exp:
        questions=ALL_QUESTIONS.get(exp,[])
        if not questions: continue

        pdf_path=os.path.join(out_dir,f'nl_steering_{exp}_v4.pdf')
        print(f"\nGenerating: {pdf_path}")

        top_old=compute_old_top_layers(all_data,exp)
        print(f"    Old top layers: {top_old}")

        with PdfPages(pdf_path) as pdf:
            print(f"  A1: Old method heatmap")
            pg_a1_old_heatmap(pdf,all_data,exp)

            print(f"  A2: New method layer effectiveness")
            top_new=pg_a2_new_layer(pdf,all_data,exp)
            print(f"    New top layers: {top_new}")

            top_all=sorted(set(top_old)|set(top_new))
            print(f"    Combined: {top_all}")

            print(f"  A3: Best layers summary")
            pg_a3_summary(pdf,all_data,exp,top_old,top_new)

            print(f"  A4: Hint direction")
            pg_a4_hint(pdf,all_data,exp,top_all)

            print(f"  B: Per-question detail")
            for it in ['v7','v8']:
                for key,run in sorted(all_data[it].items()):
                    if key[0]!=exp: continue
                    meta=run['meta']; meta['iter']=it
                    for qr in run['results']:
                        q=qr['question']
                        print(f"    {it}/cfg{key[1]} {q['id']}")
                        pg_b1a_raw(pdf,qr,q,meta,exp)
                        pg_b1b_delta(pdf,qr,q,meta,exp)
                        pg_b2_pine_mcq(pdf,qr,q,meta,top_all,exp)

            print(f"  C: Quantitative tables")
            for it in ['v7','v8']:
                for key,run in sorted(all_data[it].items()):
                    if key[0]!=exp: continue
                    meta=run['meta']; meta['iter']=it
                    print(f"    {it}/cfg{key[1]}: tables")
                    pg_c_tables(pdf,run,meta,top_all,exp)

            print(f"  C3: Cross-question consistency")
            pg_c3_consistency(pdf,all_data,exp,top_all)

        print(f"  Done! -> {pdf_path}")


if __name__=="__main__":
    main()