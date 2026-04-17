# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Deterministic Prediction Accuracy — Gemma 4 31B.

# Extracts answers ONLY from <answer>...</answer> tags in generated_text.
# NO fallback parsing (no regex cascade, no backtick/bold extraction).
# This eliminates false positives flagged by Matteo (Apr 8 meeting).

# Tracks:
#   - parse_rate: fraction of responses with valid <answer> tags
#   - gen_match:  fraction of parsed answers matching ground truth
#   - gof:        Matteo's whiteboard GoF formula (Apr 3)

# Output: gemma-4-.../cross_iteration_comparison/
#         predictions_accuracy_*_comparison_*_deterministic_gemma4_31b.pdf

# Usage:
#     python prediction_results_gemma4_31b.py --exp exp2
#     python prediction_results_gemma4_31b.py --exp exp2 --iter v7
#     python prediction_results_gemma4_31b.py --exp exp1a --iter v7 v8
# """
# import sys, os, json, re, argparse, hashlib, math
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from collections import Counter, defaultdict

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from config import (EXP_CONFIG, CONFIG_MATRIX, BASE_DIR,
#                     MODEL_REGISTRY, config_label, get_config,
#                     get_iter_output_dir, COT_ONLY_IDS)


# MODEL_KEY = 'gemma4_31b'


# # ══════════════════════════════════════════════════════════════════════
# # Deterministic Answer Extraction — <answer> tag ONLY
# # ══════════════════════════════════════════════════════════════════════

# def extract_answer_tag(generated_text):
#     """Extract answer from <answer>...</answer> tags ONLY.

#     Returns the extracted string, or None if no valid tag found.
#     No fallbacks, no regex cascade, no backtick/bold parsing.
#     """
#     if not generated_text:
#         return None
#     match = re.search(r'<answer>\s*(\S+?)\s*</answer>', generated_text)
#     if match:
#         return match.group(1).strip()
#     return None


# # ══════════════════════════════════════════════════════════════════════
# # GoF — Exact Whiteboard Formula (Apr 3 meeting)
# # ══════════════════════════════════════════════════════════════════════

# def compute_gof_whiteboard(sample_records, user_dist, epsilon=1e-10):
#     """GoF from Matteo's Apr 3 whiteboard — exact formula with smoothing.
#     sample_records is a list，each elements here is a dict：{user, pred_geo, gt_geo}
#     user_dist: is the historical distribution of the person (i.e. {'HOME': 0.7, 'WORK': 0.25, 'SHOP': 0.05})
    
#     p̄i: User's historical frequency at location i. Continuous, sums to 1 per user.
#     p̄*i: Ground truth. One-hot: 1 for the correct location, 0 for all others.
#     q̄i: Model's prediction. One-hot (smoothed): 1−ε for predicted location, ε/(N−1) for others.
#     𝟙()Indicator: 1 when prediction ≠ ground truth, 0 otherwise.
#     ΣU,D: Sum over all users (U) and target days (D).
#     Σi: Sum over all locations in user's vocabulary.
    
#     """
#     total_numer = 0.0
#     total_denom = 0.0

#     for rec in sample_records: #← Σ_U Σ_D 
#         uid = rec['user']
#         pred_geo = rec.get('pred_geo')
#         gt_geo = rec.get('gt_geo')
#         # user doesnt have the historical distribution then skip and mark this as false
#         if uid not in user_dist or pred_geo is None:
#             continue

#         p_bar = user_dist[uid]
#         #user's unique visited spots
#         n_locs = len(p_bar)

#         # if user only went to one palce then q_base = epsilon / (n_locs - 1) will devided by 0..
#         if n_locs < 2:
#             continue

#         q_base = epsilon / (n_locs - 1)
#         sample_ce = 0.0
#         sample_penalty = 0.0

#         # loop through ← Σ_i all places user visited
#         for loc_i, p_i in p_bar.items():
#             # if the prediciton match or not
            
#             q_i = (1.0 - epsilon) if loc_i == pred_geo else q_base
#             log_q = math.log(q_i)

#             sample_ce += p_i * log_q

#             q_onehot = 1 if loc_i == pred_geo else 0
#             p_star = 1 if loc_i == gt_geo else 0
#             if q_onehot != p_star:
#                 sample_penalty += p_i * log_q

#         total_numer += sample_penalty
#         total_denom += sample_ce

#     if abs(total_denom) < 1e-20:
#         return 1.0
#     return 1.0 - (total_numer / total_denom)


# # ══════════════════════════════════════════════════════════════════════
# # Data Loading
# # ══════════════════════════════════════════════════════════════════════

# def load_answers(iter_name, model_key, exp_name, config_id):
#     """Load answers JSON for a specific iteration."""
#     tag = MODEL_REGISTRY[model_key]['tag']
#     base = get_iter_output_dir(model_key, iter_name)
#     path = os.path.join(base, 'answers', tag,
#                         f"{exp_name}_cfg{config_id}_answers.json")
#     if not os.path.exists(path):
#         return None
#     with open(path) as f:
#         return json.load(f)


# def build_user_distributions(answers):
#     """Build per-user location frequency distribution from all answers."""
#     user_geos = defaultdict(list)
#     for a in answers:
#         uid = a.get('user')
#         gt = a.get('gt_geo_id')
#         if uid and gt:
#             user_geos[uid].append(str(gt))
#     user_dist = {}
#     for uid, geos in user_geos.items():
#         counts = Counter(geos)
#         total = len(geos)
#         user_dist[uid] = {g: c / total for g, c in counts.items()}
#     return user_dist


# # ══════════════════════════════════════════════════════════════════════
# # Core Metrics — Deterministic
# # ══════════════════════════════════════════════════════════════════════

# def compute_metrics(answers, geo_hash=False):
#     """Compute deterministic gen_match + GoF for one config.

#     Key difference from original: uses extract_answer_tag() on
#     generated_text, NOT the pre-computed parsed_answer field.
#     """
#     if not answers:
#         return {}

#     user_dist = build_user_distributions(answers)

#     cat_correct = defaultdict(int)
#     cat_total = defaultdict(int)
#     surprises = []
#     sample_records = []
#     total_correct = 0
#     total_valid = 0
#     total_parsed = 0
#     total_unparsed = 0

#     for a in answers:
#         gt = a.get('gt_geo_id')
#         if gt is None:
#             continue

#         # Deterministic extraction: <answer> tag ONLY
#         det_answer = extract_answer_tag(a.get('generated_text', ''))

#         gt_str = (hashlib.md5(str(gt).encode()).hexdigest()[:8]
#                   if geo_hash else str(gt))
#         cat = a.get('label', -1)
#         uid = a.get('user')
#         gt_raw = str(gt)

#         total_valid += 1
#         cat_total[cat] += 1

#         if det_answer is None:
#             total_unparsed += 1
#             matched = False
#         else:
#             total_parsed += 1
#             matched = (det_answer == gt_str)

#         if matched:
#             total_correct += 1
#             cat_correct[cat] += 1

#         # Surprise
#         p_gt = 0.0
#         if uid in user_dist and gt_raw in user_dist[uid]:
#             p_gt = user_dist[uid][gt_raw]
#         if p_gt < 1e-10:
#             p_gt = 1e-10
#         surprise = -math.log2(p_gt)
#         surprises.append(surprise)

#         sample_records.append({
#             'user': uid, 'cat': cat, 'correct': matched, 'p_gt': p_gt,
#             'surprise': surprise,
#             'pred_geo': det_answer,
#             'gt_geo': gt_raw,
#         })

#     # GoF whiteboard formula
#     gof = compute_gof_whiteboard(sample_records, user_dist)
#     gof_per_cat = {}
#     for cat in sorted(cat_total.keys()):
#         cat_recs = [r for r in sample_records if r['cat'] == cat]
#         gof_per_cat[cat] = compute_gof_whiteboard(cat_recs, user_dist)

#     parse_rate = total_parsed / max(total_valid, 1)

#     results = {
#         'gen_match': total_correct / max(total_valid, 1),
#         'valid': total_valid,
#         'parsed': total_parsed,
#         'unparsed': total_unparsed,
#         'parse_rate': parse_rate,
#         'per_category': {},
#         'mean_surprise': float(np.mean(surprises)) if surprises else 0,
#         'median_surprise': float(np.median(surprises)) if surprises else 0,
#         'gof': gof,
#         'gof_per_category': gof_per_cat,
#     }
#     for cat in sorted(cat_total.keys()):
#         acc = cat_correct[cat] / max(cat_total[cat], 1)
#         results['per_category'][cat] = {
#             'accuracy': acc, 'n': cat_total[cat], 'correct': cat_correct[cat],
#         }
#     return results


# # ══════════════════════════════════════════════════════════════════════
# # Visualization
# # ══════════════════════════════════════════════════════════════════════

# ITER_COLORS = {'v7': '#2196F3', 'v8': '#FF5722'}
# CAT_HATCHES = {0: '///', 1: '', 2: '\\\\\\', 3: 'xxx', 4: '...'}


# def _bar_labels(ax, bars, fmt='{:.0%}', fontsize=7):
#     """Add percentage labels on top of every bar."""
#     for bar in bars:
#         h = bar.get_height()
#         if h > 0.005:
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     h + 0.012,
#                     fmt.format(h), ha='center', va='bottom', fontsize=fontsize,
#                     fontweight='bold')


# def pg_parse_rate(pdf, data, exp, exp_cfg, found_cfgs, iters):
#     """Page: <answer> tag parse rate per config."""
#     fig, ax = plt.subplots(figsize=(max(12, len(found_cfgs) * 3), 6))
#     n_iters = len(iters)
#     x = np.arange(len(found_cfgs))
#     w = 0.8 / max(n_iters, 1)
#     cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

#     for i, it in enumerate(iters):
#         vals = [data.get(it, {}).get(c, {}).get('parse_rate', 0) for c in found_cfgs]
#         offset = (i - n_iters / 2 + 0.5) * w
#         bars = ax.bar(x + offset, vals, w, label=it.upper(),
#                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
#         _bar_labels(ax, bars)

#     ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
#     ax.set_ylabel('Parse Rate'); ax.set_ylim(0, 1.15)
#     ax.set_title(f'{exp_cfg["name"]} — <answer> Tag Parse Rate\n'
#                  f'Fraction of responses with valid <answer>...</answer> tags',
#                  fontweight='bold')
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')
#     plt.tight_layout()
#     pdf.savefig(fig, dpi=200); plt.close()


# def pg_raw_accuracy(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
#     """Page: Deterministic gen_match + per-category accuracy."""
#     fig, axes = plt.subplots(2, 1, figsize=(max(12, len(found_cfgs) * 3), 12))

#     # Row 1: Overall gen_match
#     ax = axes[0]
#     n_iters = len(iters)
#     x = np.arange(len(found_cfgs))
#     w = 0.8 / max(n_iters, 1)
#     cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

#     for i, it in enumerate(iters):
#         vals = [data.get(it, {}).get(c, {}).get('gen_match', 0) for c in found_cfgs]
#         offset = (i - n_iters / 2 + 0.5) * w
#         bars = ax.bar(x + offset, vals, w, label=it.upper(),
#                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
#         _bar_labels(ax, bars)

#     ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
#     ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.15)
#     ax.set_title(f'{exp_cfg["name"]} — Deterministic Accuracy (gen_match)\n'
#                  f'<answer> tag extraction only, no fallback parsing',
#                  fontweight='bold')
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

#     # Row 2: Per-category accuracy
#     ax = axes[1]
#     cats = sorted(cat_names.keys())
#     n_groups = n_iters * len(cats)
#     grp_w = 0.85 / max(n_groups, 1)

#     for ci, cat in enumerate(cats):
#         for ii, it in enumerate(iters):
#             vals = []
#             for c in found_cfgs:
#                 m = data.get(it, {}).get(c, {})
#                 pc = m.get('per_category', {})
#                 vals.append(pc.get(cat, {}).get('accuracy', 0))
#             idx_g = ci * n_iters + ii
#             offset = (idx_g - n_groups / 2 + 0.5) * grp_w
#             bars = ax.bar(x + offset, vals, grp_w,
#                           label=f'{it.upper()} {cat_names[cat]}',
#                           color=ITER_COLORS.get(it, 'gray'),
#                           hatch=CAT_HATCHES.get(cat, ''),
#                           edgecolor='black', lw=0.5, alpha=0.85)
#             _bar_labels(ax, bars)

#     ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
#     ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.15)
#     ax.set_title(f'{exp_cfg["name"]} — Per-Category Deterministic Accuracy',
#                  fontweight='bold')
#     ax.legend(fontsize=6, ncol=min(n_groups, 4), loc='upper right')
#     ax.grid(True, alpha=0.15, axis='y')

#     plt.tight_layout()
#     pdf.savefig(fig, dpi=200); plt.close()


# def pg_gof(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
#     """Page: GoF (whiteboard formula)."""
#     fig, axes = plt.subplots(2, 1, figsize=(max(12, len(found_cfgs) * 3), 12))
#     n_iters = len(iters)
#     x = np.arange(len(found_cfgs))
#     w = 0.8 / max(n_iters, 1)
#     cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

#     # Row 1: Overall GoF
#     ax = axes[0]
#     for i, it in enumerate(iters):
#         vals = [data.get(it, {}).get(c, {}).get('gof', 0) for c in found_cfgs]
#         offset = (i - n_iters / 2 + 0.5) * w
#         bars = ax.bar(x + offset, vals, w, label=it.upper(),
#                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
#         _bar_labels(ax, bars, fmt='{:.1%}')
#     ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
#     ax.set_ylabel('GoF'); ax.set_ylim(0, 1.15)
#     ax.set_title(f'{exp_cfg["name"]} — GoF (whiteboard formula, Apr 3)',
#                  fontweight='bold')
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

#     # Row 2: Per-category GoF
#     ax = axes[1]
#     cats = sorted(cat_names.keys())
#     n_groups = n_iters * len(cats)
#     grp_w = 0.85 / max(n_groups, 1)

#     for ci, cat in enumerate(cats):
#         for ii, it in enumerate(iters):
#             vals = []
#             for c in found_cfgs:
#                 m = data.get(it, {}).get(c, {})
#                 gc = m.get('gof_per_category', {})
#                 vals.append(gc.get(cat, 0))
#             idx_g = ci * n_iters + ii
#             offset = (idx_g - n_groups / 2 + 0.5) * grp_w
#             bars = ax.bar(x + offset, vals, grp_w,
#                           label=f'{it.upper()} {cat_names[cat]}',
#                           color=ITER_COLORS.get(it, 'gray'),
#                           hatch=CAT_HATCHES.get(cat, ''),
#                           edgecolor='black', lw=0.5, alpha=0.85)
#             _bar_labels(ax, bars, fmt='{:.1%}')

#     ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
#     ax.set_ylabel('GoF'); ax.set_ylim(0, 1.15)
#     ax.set_title(f'{exp_cfg["name"]} — Per-Category GoF', fontweight='bold')
#     ax.legend(fontsize=6, ncol=min(n_groups, 4), loc='upper right')
#     ax.grid(True, alpha=0.15, axis='y')

#     plt.tight_layout()
#     pdf.savefig(fig, dpi=200); plt.close()


# def pg_surprise(pdf, data, exp, exp_cfg, found_cfgs, iters):
#     """Page: Task difficulty (surprise)."""
#     fig, ax = plt.subplots(figsize=(max(12, len(found_cfgs) * 3), 6))
#     n_iters = len(iters)
#     x = np.arange(len(found_cfgs))
#     w = 0.8 / max(n_iters, 1)
#     cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]

#     for i, it in enumerate(iters):
#         means = [data.get(it, {}).get(c, {}).get('mean_surprise', 0) for c in found_cfgs]
#         meds = [data.get(it, {}).get(c, {}).get('median_surprise', 0) for c in found_cfgs]
#         offset = (i - n_iters / 2 + 0.5) * w
#         bars = ax.bar(x + offset, means, w, label=f'{it.upper()} mean',
#                       color=ITER_COLORS.get(it, 'gray'), edgecolor='black', lw=0.5)
#         for bar, v, md in zip(bars, means, meds):
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + 0.05,
#                     f'{v:.2f}\n(med {md:.2f})', ha='center', fontsize=6)

#     ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
#     ax.set_ylabel('Surprise (bits)')
#     ax.set_title(f'{exp_cfg["name"]} — Task Difficulty (surprise)', fontweight='bold')
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')
#     plt.tight_layout()
#     pdf.savefig(fig, dpi=200); plt.close()


# def pg_summary_table(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
#     """Page: Summary table with all numbers."""
#     fig, ax = plt.subplots(figsize=(20, max(8, len(found_cfgs) * len(iters) * 0.8)))
#     ax.axis('off')

#     cats = sorted(cat_names.keys())
#     lines = [
#         f'{exp_cfg["name"]} — Deterministic Prediction Summary | gemma4_31b\n'
#         f'Answer extraction: <answer>...</answer> tag ONLY (no fallbacks)\n'
#         f'GoF formula: Matteo whiteboard Apr 3 (exact, no |.|, 1-penalty)\n\n'
#     ]

#     # Header
#     cat_hdr = ''.join(f'{cat_names[c]:>12s} acc  GoF' for c in cats)
#     lines.append(
#         f'{"Iter":>5s} {"Cfg":>5s} {"N":>5s} {"parsed":>7s} '
#         f'{"parse%":>7s} {"gen_match":>10s} {"GoF":>8s} '
#         f'{"surprise":>9s} {cat_hdr}\n')
#     lines.append('─' * 120 + '\n')

#     for it in iters:
#         for cfg_id in found_cfgs:
#             m = data.get(it, {}).get(cfg_id, {})
#             if not m:
#                 continue
#             cat_str = ''
#             for c in cats:
#                 pc = m.get('per_category', {}).get(c, {})
#                 gc = m.get('gof_per_category', {}).get(c, 0)
#                 cat_str += f'{cat_names[c]:>12s} {pc.get("accuracy", 0):>4.0%}  {gc:>4.0%}'
#             lines.append(
#                 f'{it.upper():>5s} cfg{cfg_id:>2d} {m.get("valid", 0):>5d} '
#                 f'{m.get("parsed", 0):>7d} '
#                 f'{m.get("parse_rate", 0):>7.0%} '
#                 f'{m.get("gen_match", 0):>10.1%} {m.get("gof", 0):>8.1%} '
#                 f'{m.get("mean_surprise", 0):>9.2f} {cat_str}\n')
#         lines.append('\n')

#     lines.append('─' * 120 + '\n')
#     lines.append('\nparse% = fraction with valid <answer> tags '
#                  '(Matteo: current fallbacks give false positives)\n')
#     lines.append('gen_match = correct / total_valid (unparsed = wrong)\n')
#     lines.append('GoF = 1 - penalty/CE (Matteo whiteboard, Apr 3)\n')

#     ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
#             fontsize=7, va='top', fontfamily='monospace')
#     plt.tight_layout()
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # Main
# # ══════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--exp', required=True, choices=['exp1a', 'exp1b', 'exp2'])
#     parser.add_argument('--iter', nargs='+', default=['v7', 'v8'])
#     parser.add_argument('--config', default='cot',
#                         help='Config IDs: "cot" (default, CoT-only), "all", or comma-separated')
#     args = parser.parse_args()

#     exp = args.exp
#     exp_cfg = EXP_CONFIG[exp]
#     iters = args.iter

#     # Determine which configs to process
#     if args.config == 'cot':
#         config_ids = COT_ONLY_IDS[exp]
#     elif args.config == 'all':
#         config_ids = [c['id'] for c in CONFIG_MATRIX[exp]]
#     else:
#         config_ids = [int(x) for x in args.config.split(',')]

#     cat_names = {
#         'exp1a': {0: 'Weekend', 1: 'Weekday'},
#         'exp1b': {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'},
#         'exp2':  {0: 'Unemployed', 1: 'Employed'},
#     }[exp]

#     model_base = MODEL_REGISTRY[MODEL_KEY]['hf_name']
#     out_dir = os.path.join(model_base, 'cross_iteration_comparison')
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"{'='*70}")
#     print(f"Deterministic Prediction Accuracy: {exp} (Gemma 4 31B)")
#     print(f"  Iterations: {iters}")
#     print(f"  Configs: {config_ids}")
#     print(f"  Extraction: <answer>...</answer> tag ONLY")
#     print(f"{'='*70}")

#     # ── Load data for all iterations ──
#     data = {}
#     for it in iters:
#         data[it] = {}
#         for cfg_id in config_ids:
#             cfg = get_config(exp, cfg_id)
#             clabel = config_label(exp, cfg_id)
#             answers = load_answers(it, MODEL_KEY, exp, cfg_id)
#             if answers is None:
#                 print(f"  {it} cfg{cfg_id}: not found")
#                 continue

#             metrics = compute_metrics(answers, geo_hash=cfg['geo_hash'])
#             data[it][cfg_id] = metrics

#             print(f"\n  {it.upper()} cfg{cfg_id} ({clabel}):")
#             print(f"    parse_rate = {metrics['parse_rate']:.0%} "
#                   f"({metrics['parsed']}/{metrics['valid']})")
#             print(f"    gen_match  = {metrics['gen_match']:.1%}  |  "
#                   f"GoF = {metrics['gof']:.1%}")

#             for cat in sorted(metrics['per_category'].keys()):
#                 info = metrics['per_category'][cat]
#                 name = cat_names.get(cat, f'Cat{cat}')
#                 gof_c = metrics['gof_per_category'].get(cat, 0)
#                 print(f"    {name:12s}: acc={info['accuracy']:.1%} "
#                       f"GoF={gof_c:.1%}  "
#                       f"({info['correct']}/{info['n']})")

#     # ── Generate comparison PDF ──
#     found_cfgs = sorted(set(c for it in data for c in data[it]))
#     if not found_cfgs:
#         print("No data found."); return

#     iter_str = '_'.join(iters)
#     pdf_path = os.path.join(
#         out_dir,
#         f'predictions_accuracy_{iter_str}_comparison_{exp}'
#         f'_deterministic_gemma4_31b.pdf')
#     print(f"\nGenerating: {pdf_path}")

#     with PdfPages(pdf_path) as pdf:
#         pg_parse_rate(pdf, data, exp, exp_cfg, found_cfgs, iters)
#         pg_raw_accuracy(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)
#         pg_gof(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)
#         pg_surprise(pdf, data, exp, exp_cfg, found_cfgs, iters)
#         pg_summary_table(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)

#     print(f"\n{'='*70}")
#     print(f"Done! → {pdf_path}")
#     print(f"{'='*70}")

#     # ── Save JSON ──
#     json_out = os.path.join(
#         out_dir,
#         f'prediction_{exp}_{iter_str}_deterministic_metrics.json')
#     serializable = {}
#     for it in data:
#         serializable[it] = {}
#         for cfg_id, m in data[it].items():
#             s = dict(m)
#             s['gof_per_category'] = {str(k): v
#                                      for k, v in s['gof_per_category'].items()}
#             s['per_category'] = {str(k): v
#                                  for k, v in s['per_category'].items()}
#             serializable[it][str(cfg_id)] = s
#     with open(json_out, 'w') as f:
#         json.dump(serializable, f, indent=2, default=str)
#     print(f"  Metrics JSON: {json_out}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deterministic Prediction Accuracy — Gemma 4 31B.

Extracts answers ONLY from <answer>...</answer> tags in generated_text.
NO fallback parsing (no regex cascade, no backtick/bold extraction).
This eliminates false positives flagged by Matteo (Apr 8 meeting).

Tracks:
  - parse_rate: fraction of responses with valid <answer> tags
  - gen_match:  fraction of parsed answers matching ground truth
  - gof:        Matteo's whiteboard GoF formula (Apr 3)

Bugfixes (Apr 16):
  - Hash mismatch: for hashed configs, user_dist and gt_geo are now
    in the same space as pred_geo (hashed). Previously pred_geo was
    hashed but user_dist keys were raw -> loc_i == pred_geo never matched.
  - Parse failures: pred_geo=None no longer skipped in GoF. Now treated
    as wrong prediction (sentinel "__UNPARSED__"), consistent with
    accuracy which already counts parse failures as incorrect.

Output: gemma-4-.../cross_iteration_comparison/
        predictions_accuracy_*_comparison_*_deterministic_gemma4_31b.pdf

Usage:
    python prediction_results_gemma4_31b.py --exp exp2
    python prediction_results_gemma4_31b.py --exp exp2 --iter v7
    python prediction_results_gemma4_31b.py --exp exp1a --iter v7 v8
"""
import sys, os, json, re, argparse, hashlib, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (EXP_CONFIG, CONFIG_MATRIX, BASE_DIR,
                    MODEL_REGISTRY, config_label, get_config,
                    get_iter_output_dir, COT_ONLY_IDS)


MODEL_KEY = 'gemma4_31b'

_UNPARSED = '__UNPARSED__'


# ======================================================================
# Deterministic Answer Extraction — <answer> tag ONLY
# ======================================================================

def extract_answer_tag(generated_text):
    """Extract answer from <answer>...</answer> tags ONLY."""
    if not generated_text:
        return None
    match = re.search(r'<answer>\s*(\S+?)\s*</answer>', generated_text)
    if match:
        return match.group(1).strip()
    return None


# ======================================================================
# GoF — Exact Whiteboard Formula (Apr 3 meeting)
# ======================================================================

def compute_gof_whiteboard(sample_records, user_dist, epsilon=1e-10):
    """GoF from Apr 3 whiteboard — exact formula with smoothing.
    Goodness-of-Fit from Matteo's Apr 3 whiteboard formula:

        GoF = 1 - Σ_U Σ_D Σ_i  p̄_i · log(q̄_i) · 𝟙(q̄_i ≠ p̄*_i)
                  ─────────────────────────────────────────────────
                  Σ_U Σ_D Σ_i  p̄_i · log(q̄_i)

    Notation:
        p̄_i   = user's historical visit frequency at location i
                 (e.g. HOME=0.6, WORK=0.3, SHOP=0.1; sums to 1)
        p̄*_i  = ground truth, one-hot
                 (1 for the correct location, 0 for all others)
        q̄_i   = model's prediction, smoothed one-hot
                 (1-ε for predicted location, ε/(N-1) for others)
        𝟙()   = indicator function: 1 when prediction ≠ ground truth
        Σ_U,D = sum over all users and target days (outer loop)
        Σ_i   = sum over all locations in user's vocabulary (inner loop)
        
    inputs:
    sample_records is a list，each elements here is a dict：{user, pred_geo, gt_geo}
    user_dist is the historical distribution of the person (i.e. {'HOME': 0.7, 'WORK': 0.25, 'SHOP': 0.05})
    """
    total_numer = 0.0
    total_denom = 0.0

    # ── Outer loop: Σ_U Σ_D (one iteration per user-day sample) ──
    for rec in sample_records:
        uid = rec['user']
        pred_geo = rec.get('pred_geo', _UNPARSED)
        gt_geo = rec.get('gt_geo')

        # p̄: this user's historical location distribution
        p_bar = user_dist[uid]
        n_locs = len(p_bar)

        # Construct q̄ (model prediction as a smoothed probability distribution):
        #   predicted location gets q = 1 - epsilon  ≈ 1.0
        #   all other locations share the remaining epsilon equally
        #   so each non-predicted location gets q = epsilon / (N-1)
        #   total: (1 - epsilon) + (N-1) * epsilon/(N-1) = 1.0 
        q_base = epsilon / (n_locs - 1)

        # Denominator contribution: Σ_i p̄_i · log(q̄_i)
        # = cross-entropy of user's real distribution w.r.t. model's prediction
        # This is always ≤ 0 because log(q̄_i) ≤ 0 and p̄_i ≥ 0.
        # Interpretation: the total "information cost" of this prediction.
        # Used as normalization so that easy users (few locations, one dominant)
        # and hard users (many locations, spread out) are on the same scale.
        sample_ce = 0.0

        # Numerator contribution: Σ_i p̄_i · log(q̄_i) · 𝟙(q̄_i ≠ p̄*_i)
        # = same cross-entropy, but ONLY at locations where prediction
        #   disagrees with ground truth. If prediction is correct,
        #   q̄ and p̄* agree everywhere → this sum = 0 → no penalty.
        sample_penalty = 0.0

        # ── Inner loop: Σ_i (one iteration per location in user's vocab) ──
        for loc_i, p_i in p_bar.items():
            # if predit correct then full score with almost 1
            # q̄_i: model's predicted probability for this location
            q_i = (1.0 - epsilon) if loc_i == pred_geo else q_base
            log_q = math.log(q_i)
            # Denominator: accumulate p̄_i · log(q̄_i)
            sample_ce += p_i * log_q

            # Numerator: only accumulate where prediction ≠ ground truth
            # q_onehot = 1 if model predicted this location, else 0
            # p_star   = 1 if ground truth is this location, else 0
            # They disagree if:
            #  loc_i != pred but loc_i == gt  (false negative)
            #       → log_q ≈ -24, penalty = p_gt × (-24) ( PENALTY)
            q_onehot = 1 if loc_i == pred_geo else 0
            p_star = 1 if loc_i == gt_geo else 0
            if q_onehot != p_star:
                # and pi is the historical visit frequency at location i
                sample_penalty += p_i * log_q

        total_numer += sample_penalty
        total_denom += sample_ce

    return 1.0 - (total_numer / total_denom)


# ======================================================================
# Data Loading
# ======================================================================

def load_answers(iter_name, model_key, exp_name, config_id):
    """Load answers JSON for a specific iteration."""
    tag = MODEL_REGISTRY[model_key]['tag']
    base = get_iter_output_dir(model_key, iter_name)
    path = os.path.join(base, 'answers', tag,
                        f"{exp_name}_cfg{config_id}_answers.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _geo_key(raw_geo, geo_hash):
    """Convert raw geo_id to the appropriate space."""
    s = str(raw_geo)
    if geo_hash:
        return hashlib.md5(s.encode()).hexdigest()[:8]
    return s


def build_user_distributions(answers, geo_hash=False):
    """Build per-user location frequency distribution.

    When geo_hash=True, keys are hashed geo_ids (matching model output).
    When geo_hash=False, keys are raw geo_ids.
    """
    user_geos = defaultdict(list)
    for a in answers:
        uid = a.get('user')
        gt = a.get('gt_geo_id')
        if uid and gt:
            user_geos[uid].append(_geo_key(gt, geo_hash))
    user_dist = {}
    for uid, geos in user_geos.items():
        counts = Counter(geos)
        total = len(geos)
        user_dist[uid] = {g: c / total for g, c in counts.items()}
    return user_dist


# ======================================================================
# Core Metrics — Deterministic
# ======================================================================

def compute_metrics(answers, geo_hash=False):
    """Compute deterministic gen_match + GoF for one config.

    Bugfix: user_dist, pred_geo, gt_geo are all in the same space
    (hashed when geo_hash=True, raw otherwise).
    """
    if not answers:
        return {}

    # Build user_dist in the correct space
    user_dist = build_user_distributions(answers, geo_hash=geo_hash)

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    surprises = []
    sample_records = []
    total_correct = 0
    total_valid = 0
    total_parsed = 0
    total_unparsed = 0

    for a in answers:
        gt = a.get('gt_geo_id')
        if gt is None:
            continue

        # Deterministic extraction: <answer> tag ONLY
        det_answer = extract_answer_tag(a.get('generated_text', ''))

        # gt_str in model-output space (hashed or raw)
        gt_str = _geo_key(gt, geo_hash)
        cat = a.get('label', -1)
        uid = a.get('user')

        total_valid += 1
        cat_total[cat] += 1

        if det_answer is None:
            total_unparsed += 1
            matched = False
        else:
            total_parsed += 1
            matched = (det_answer == gt_str)

        if matched:
            total_correct += 1
            cat_correct[cat] += 1

        # Surprise — use the same-space distribution
        p_gt = 0.0
        if uid in user_dist and gt_str in user_dist[uid]:
            p_gt = user_dist[uid][gt_str]
        if p_gt < 1e-10:
            p_gt = 1e-10
        surprise = -math.log2(p_gt)
        surprises.append(surprise)

        # GoF record — all in same space
        # Parse failure -> _UNPARSED sentinel (max penalty)
        sample_records.append({
            'user': uid, 'cat': cat, 'correct': matched, 'p_gt': p_gt,
            'surprise': surprise,
            'pred_geo': det_answer if det_answer is not None else _UNPARSED,
            'gt_geo': gt_str,
        })

    # GoF whiteboard formula
    gof = compute_gof_whiteboard(sample_records, user_dist)
    gof_per_cat = {}
    for cat in sorted(cat_total.keys()):
        cat_recs = [r for r in sample_records if r['cat'] == cat]
        gof_per_cat[cat] = compute_gof_whiteboard(cat_recs, user_dist)

    parse_rate = total_parsed / max(total_valid, 1)

    results = {
        'gen_match': total_correct / max(total_valid, 1),
        'valid': total_valid,
        'parsed': total_parsed,
        'unparsed': total_unparsed,
        'parse_rate': parse_rate,
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


# ======================================================================
# Visualization
# ======================================================================

ITER_COLORS = {'v7': '#2196F3', 'v8': '#FF5722'}
CAT_HATCHES = {0: '///', 1: '', 2: '\\\\\\', 3: 'xxx', 4: '...'}


def _bar_labels(ax, bars, fmt='{:.0%}', fontsize=7):
    for bar in bars:
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    fmt.format(h), ha='center', va='bottom', fontsize=fontsize,
                    fontweight='bold')


def pg_parse_rate(pdf, data, exp, exp_cfg, found_cfgs, iters):
    fig, ax = plt.subplots(figsize=(max(12, len(found_cfgs) * 3), 6))
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]
    for i, it in enumerate(iters):
        vals = [data.get(it, {}).get(c, {}).get('parse_rate', 0)
                for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=it.upper(),
                      color=ITER_COLORS.get(it, 'gray'),
                      edgecolor='black', lw=0.5)
        _bar_labels(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Parse Rate'); ax.set_ylim(0, 1.15)
    ax.set_title(f'{exp_cfg["name"]} -- <answer> Tag Parse Rate\n'
                 f'Fraction of responses with valid <answer>...</answer> tags',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_raw_accuracy(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(found_cfgs) * 3), 12))
    ax = axes[0]
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]
    for i, it in enumerate(iters):
        vals = [data.get(it, {}).get(c, {}).get('gen_match', 0)
                for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=it.upper(),
                      color=ITER_COLORS.get(it, 'gray'),
                      edgecolor='black', lw=0.5)
        _bar_labels(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.15)
    ax.set_title(f'{exp_cfg["name"]} -- Deterministic Accuracy (gen_match)\n'
                 f'<answer> tag extraction only, no fallback parsing',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

    ax = axes[1]
    cats = sorted(cat_names.keys())
    n_groups = n_iters * len(cats)
    grp_w = 0.85 / max(n_groups, 1)
    for ci, cat in enumerate(cats):
        for ii, it in enumerate(iters):
            vals = []
            for c in found_cfgs:
                m = data.get(it, {}).get(c, {})
                pc = m.get('per_category', {})
                vals.append(pc.get(cat, {}).get('accuracy', 0))
            idx_g = ci * n_iters + ii
            offset = (idx_g - n_groups / 2 + 0.5) * grp_w
            bars = ax.bar(x + offset, vals, grp_w,
                          label=f'{it.upper()} {cat_names[cat]}',
                          color=ITER_COLORS.get(it, 'gray'),
                          hatch=CAT_HATCHES.get(cat, ''),
                          edgecolor='black', lw=0.5, alpha=0.85)
            _bar_labels(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.15)
    ax.set_title(f'{exp_cfg["name"]} -- Per-Category Deterministic Accuracy',
                 fontweight='bold')
    ax.legend(fontsize=6, ncol=min(n_groups, 4), loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_gof(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(found_cfgs) * 3), 12))
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]
    ax = axes[0]
    for i, it in enumerate(iters):
        vals = [data.get(it, {}).get(c, {}).get('gof', 0) for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=it.upper(),
                      color=ITER_COLORS.get(it, 'gray'),
                      edgecolor='black', lw=0.5)
        _bar_labels(ax, bars, fmt='{:.1%}')
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('GoF'); ax.set_ylim(0, 1.15)
    ax.set_title(f'{exp_cfg["name"]} -- GoF (whiteboard formula, Apr 3)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

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
            idx_g = ci * n_iters + ii
            offset = (idx_g - n_groups / 2 + 0.5) * grp_w
            bars = ax.bar(x + offset, vals, grp_w,
                          label=f'{it.upper()} {cat_names[cat]}',
                          color=ITER_COLORS.get(it, 'gray'),
                          hatch=CAT_HATCHES.get(cat, ''),
                          edgecolor='black', lw=0.5, alpha=0.85)
            _bar_labels(ax, bars, fmt='{:.1%}')
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('GoF'); ax.set_ylim(0, 1.15)
    ax.set_title(f'{exp_cfg["name"]} -- Per-Category GoF', fontweight='bold')
    ax.legend(fontsize=6, ncol=min(n_groups, 4), loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_surprise(pdf, data, exp, exp_cfg, found_cfgs, iters):
    fig, ax = plt.subplots(figsize=(max(12, len(found_cfgs) * 3), 6))
    n_iters = len(iters)
    x = np.arange(len(found_cfgs))
    w = 0.8 / max(n_iters, 1)
    cfg_labels = [f"cfg{c}\n{config_label(exp, c)}" for c in found_cfgs]
    for i, it in enumerate(iters):
        means = [data.get(it, {}).get(c, {}).get('mean_surprise', 0)
                 for c in found_cfgs]
        meds = [data.get(it, {}).get(c, {}).get('median_surprise', 0)
                for c in found_cfgs]
        offset = (i - n_iters / 2 + 0.5) * w
        bars = ax.bar(x + offset, means, w, label=f'{it.upper()} mean',
                      color=ITER_COLORS.get(it, 'gray'),
                      edgecolor='black', lw=0.5)
        for bar, v, md in zip(bars, means, meds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f'{v:.2f}\n(med {md:.2f})', ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.set_ylabel('Surprise (bits)')
    ax.set_title(f'{exp_cfg["name"]} -- Task Difficulty (surprise)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    pdf.savefig(fig, dpi=200); plt.close()


def pg_summary_table(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names):
    fig, ax = plt.subplots(
        figsize=(20, max(8, len(found_cfgs) * len(iters) * 0.8)))
    ax.axis('off')
    cats = sorted(cat_names.keys())
    lines = [
        f'{exp_cfg["name"]} -- Deterministic Prediction Summary | gemma4_31b\n'
        f'Answer extraction: <answer>...</answer> tag ONLY (no fallbacks)\n'
        f'GoF: Matteo whiteboard Apr 3 (exact, 1-penalty)\n'
        f'Parse failures: counted as WRONG in both accuracy and GoF\n'
        f'Hash fix: user_dist keys in same space as model output\n\n'
    ]
    cat_hdr = ''.join(f'{cat_names[c]:>12s} acc  GoF' for c in cats)
    lines.append(
        f'{"Iter":>5s} {"Cfg":>5s} {"N":>5s} {"parsed":>7s} '
        f'{"parse%":>7s} {"gen_match":>10s} {"GoF":>8s} '
        f'{"surprise":>9s} {cat_hdr}\n')
    lines.append('-' * 120 + '\n')
    for it in iters:
        for cfg_id in found_cfgs:
            m = data.get(it, {}).get(cfg_id, {})
            if not m:
                continue
            cat_str = ''
            for c in cats:
                pc = m.get('per_category', {}).get(c, {})
                gc = m.get('gof_per_category', {}).get(c, 0)
                cat_str += (f'{cat_names[c]:>12s} '
                            f'{pc.get("accuracy", 0):>4.0%}  {gc:>4.0%}')
            lines.append(
                f'{it.upper():>5s} cfg{cfg_id:>2d} {m.get("valid", 0):>5d} '
                f'{m.get("parsed", 0):>7d} '
                f'{m.get("parse_rate", 0):>7.0%} '
                f'{m.get("gen_match", 0):>10.1%} {m.get("gof", 0):>8.1%} '
                f'{m.get("mean_surprise", 0):>9.2f} {cat_str}\n')
        lines.append('\n')
    lines.append('-' * 120 + '\n')
    lines.append('\nparse% = fraction with valid <answer> tags\n')
    lines.append('gen_match = correct / total_valid (unparsed = wrong)\n')
    lines.append('GoF = 1 - penalty/CE (Matteo whiteboard, Apr 3)\n')
    lines.append('  * Parse failures penalized in GoF (not skipped)\n')
    lines.append('  * Hashed configs: user_dist keys match model output\n')
    ax.text(0.02, 0.98, ''.join(lines), transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True,
                        choices=['exp1a', 'exp1b', 'exp2'])
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'])
    parser.add_argument('--config', default='cot',
                        help='"cot" (default), "all", or comma-separated')
    args = parser.parse_args()

    exp = args.exp
    exp_cfg = EXP_CONFIG[exp]
    iters = args.iter

    if args.config == 'cot':
        config_ids = COT_ONLY_IDS[exp]
    elif args.config == 'all':
        config_ids = [c['id'] for c in CONFIG_MATRIX[exp]]
    else:
        config_ids = [int(x) for x in args.config.split(',')]

    cat_names = {
        'exp1a': {0: 'Weekend', 1: 'Weekday'},
        'exp1b': {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'},
        'exp2':  {0: 'Unemployed', 1: 'Employed'},
    }[exp]

    model_base = MODEL_REGISTRY[MODEL_KEY]['hf_name']
    out_dir = os.path.join(model_base, 'cross_iteration_comparison')
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"Deterministic Prediction Accuracy: {exp} (Gemma 4 31B)")
    print(f"  Iterations: {iters}")
    print(f"  Configs: {config_ids}")
    print(f"  Extraction: <answer>...</answer> tag ONLY")
    print(f"  GoF bugfixes: hash-space alignment + parse-failure penalty")
    print(f"{'='*70}")

    data = {}
    for it in iters:
        data[it] = {}
        for cfg_id in config_ids:
            cfg = get_config(exp, cfg_id)
            clabel = config_label(exp, cfg_id)
            answers = load_answers(it, MODEL_KEY, exp, cfg_id)
            if answers is None:
                print(f"  {it} cfg{cfg_id}: not found")
                continue

            metrics = compute_metrics(answers, geo_hash=cfg['geo_hash'])
            data[it][cfg_id] = metrics

            print(f"\n  {it.upper()} cfg{cfg_id} ({clabel}):")
            print(f"    parse_rate = {metrics['parse_rate']:.0%} "
                  f"({metrics['parsed']}/{metrics['valid']})")
            print(f"    gen_match  = {metrics['gen_match']:.1%}  |  "
                  f"GoF = {metrics['gof']:.1%}")
            for cat in sorted(metrics['per_category'].keys()):
                info = metrics['per_category'][cat]
                name = cat_names.get(cat, f'Cat{cat}')
                gof_c = metrics['gof_per_category'].get(cat, 0)
                print(f"    {name:12s}: acc={info['accuracy']:.1%} "
                      f"GoF={gof_c:.1%}  "
                      f"({info['correct']}/{info['n']})")

    found_cfgs = sorted(set(c for it in data for c in data[it]))
    if not found_cfgs:
        print("No data found."); return

    iter_str = '_'.join(iters)
    pdf_path = os.path.join(
        out_dir,
        f'predictions_accuracy_{iter_str}_comparison_{exp}'
        f'_deterministic_gemma4_31b.pdf')
    print(f"\nGenerating: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        pg_parse_rate(pdf, data, exp, exp_cfg, found_cfgs, iters)
        pg_raw_accuracy(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)
        pg_gof(pdf, data, exp, exp_cfg, found_cfgs, iters, cat_names)
        pg_surprise(pdf, data, exp, exp_cfg, found_cfgs, iters)
        pg_summary_table(pdf, data, exp, exp_cfg, found_cfgs, iters,
                         cat_names)

    print(f"\n{'='*70}")
    print(f"Done! -> {pdf_path}")
    print(f"{'='*70}")

    json_out = os.path.join(
        out_dir,
        f'prediction_{exp}_{iter_str}_deterministic_metrics.json')
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