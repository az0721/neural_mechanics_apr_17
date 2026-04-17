# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# SAE Feature Analysis — Do GPS-derived hidden states activate employment/weekday features?

# Loads pre-trained GemmaScope 2 SAEs (JumpReLU), encodes Phase 1 hidden states,
# and checks if NeuronPedia-labeled features differentially activate between classes.

# Two experiments:
#   Exp2 (employment):  employed vs unemployed → check employment SAE features
#   Exp1a (weekday):    weekday vs weekend → check weekday/holiday SAE features

# Output: sae/output/sae_feature_analysis.pdf

# Usage:
#     python sae/sae_analysis.py                         # all exps, V7 + V8
#     python sae/sae_analysis.py --exp exp2 --iter v7    # specific
#     python sae/sae_analysis.py --exp exp1a --cfgs 3 4  # specific cfgs
# """
# import sys, os, argparse, json
# import numpy as np
# import torch
# from safetensors.torch import load_file
# from scipy import stats
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from config import MODEL_REGISTRY, EXP_CONFIG, config_label

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SAE_DIR = os.path.join(BASE_DIR, 'sae', 'resid_post')
# OUT_DIR = os.path.join(BASE_DIR, 'sae', 'output')
# os.makedirs(OUT_DIR, exist_ok=True)


# # ══════════════════════════════════════════════════════════════════════
# # Target Features from NeuronPedia
# # ══════════════════════════════════════════════════════════════════════

# # Exp2: Employment concept
# EXP2_FEATURES = {
#     # (sae_layer, width, feature_id): label
#     # Layer 41, 65k
#     (41, '65k', 22394): 'employment',
#     (41, '65k', 10086): 'workplace or office',
#     (41, '65k', 45212): 'job losses',
#     # Layer 31, 65k
#     (31, '65k', 36004): 'doing or working',
#     (31, '65k', 41583): 'free time',
#     # Layer 41, 262k
#     (41, '262k', 171963): 'employment',
#     (41, '262k', 19373): 'employment',
#     (41, '262k', 202142): 'employment',
#     (41, '262k', 28067): 'jobs and employment',
#     (41, '262k', 187898): 'employment',
#     (41, '262k', 97122): 'employment',
#     (41, '262k', 247538): 'business and employment',
#     (41, '262k', 40558): 'business, employment',
#     (41, '262k', 2949): 'employment quantity',
#     (41, '262k', 22733): 'hiring, employment',
#     # Layer 12, 262k
#     (12, '262k', 172041): 'employment and law',
# }

# # Exp1a: Weekday/Weekend concept
# EXP1A_FEATURES = {
#     # Layer 41, 65k
#     (41, '65k', 47289): 'week',
#     (41, '65k', 20474): 'week',
#     (41, '65k', 65361): 'holidays',
#     (41, '65k', 7422): 'holidays',
#     (41, '65k', 27819): 'Week',
#     (41, '65k', 42844): 'holiday',
#     (41, '65k', 37325): 'days of week',
#     (41, '65k', 16397): 'days of week',
#     # Layer 31, 65k
#     (31, '65k', 38778): 'morning and weekend events',
#     # Layer 12, 65k
#     (12, '65k', 19244): 'holiday weekends',
# }

# # Control features (random, should NOT differ between groups)
# CONTROL_FEATURES = {
#     (41, '65k', 100): 'CONTROL_100',
#     (41, '65k', 500): 'CONTROL_500',
#     (41, '65k', 1000): 'CONTROL_1000',
#     (41, '65k', 5000): 'CONTROL_5000',
#     (41, '65k', 10000): 'CONTROL_10000',
# }

# ALL_FEATURES = {
#     'exp2': {**EXP2_FEATURES, **CONTROL_FEATURES},
#     'exp1a': {**EXP1A_FEATURES, **CONTROL_FEATURES},
# }


# # ══════════════════════════════════════════════════════════════════════
# # SAE Loading + Encoding
# # ══════════════════════════════════════════════════════════════════════

# def get_sae_path(layer, width):
#     """Map (layer, width) to local SAE directory."""
#     w = {'65k': '65k', '262k': '262k'}[width]
#     dirname = f"layer_{layer}_width_{w}_l0_medium"
#     path = os.path.join(SAE_DIR, dirname)
#     if not os.path.exists(path):
#         return None
#     return path


# def load_sae(sae_path):
#     """Load JumpReLU SAE parameters."""
#     params = load_file(os.path.join(sae_path, 'params.safetensors'))
#     with open(os.path.join(sae_path, 'config.json')) as f:
#         config = json.load(f)

#     return {
#         'w_enc': params['w_enc'],           # (hidden_dim, n_features)
#         'b_enc': params['b_enc'],           # (n_features,)
#         'w_dec': params['w_dec'],           # (n_features, hidden_dim)
#         'b_dec': params['b_dec'],           # (hidden_dim,)
#         'threshold': params['threshold'],    # (n_features,) JumpReLU threshold
#         'config': config,
#     }


# def encode_sae(hidden_states, sae):
#     """Encode hidden states through JumpReLU SAE → sparse feature activations."""
#     h = torch.from_numpy(hidden_states).float()
#     b_dec = sae['b_dec'].float()
#     w_enc = sae['w_enc'].float()
#     b_enc = sae['b_enc'].float()
#     threshold = sae['threshold'].float()

#     # JumpReLU: z = max(0, W_enc @ (x - b_dec) + b_enc) where pre_act > threshold
#     pre_act = (h - b_dec) @ w_enc + b_enc
#     z = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))

#     return z.numpy()  # (n_samples, n_features)


# # ══════════════════════════════════════════════════════════════════════
# # Analysis
# # ══════════════════════════════════════════════════════════════════════

# def analyze_features(activations, labels, features, class_names):
#     """Compute per-feature statistics: mean activation, t-test, effect size."""
#     results = []
#     for (sae_layer, width, feat_id), label in features.items():
#         col = activations.get((sae_layer, width))
#         if col is None:
#             continue
#         act = col[:, feat_id]
#         c0 = act[labels == 0]
#         c1 = act[labels == 1]

#         # Stats
#         m0, m1 = c0.mean(), c1.mean()
#         s0, s1 = c0.std(), c1.std()
#         pooled_std = np.sqrt((s0**2 + s1**2) / 2) if (s0 + s1) > 0 else 1e-10
#         cohen_d = (m1 - m0) / pooled_std
#         t_stat, p_val = stats.ttest_ind(c1, c0, equal_var=False)
#         frac_active_0 = (c0 > 0).mean()
#         frac_active_1 = (c1 > 0).mean()

#         results.append({
#             'sae_layer': sae_layer, 'width': width, 'feat_id': feat_id,
#             'label': label,
#             'mean_c0': m0, 'mean_c1': m1,
#             'std_c0': s0, 'std_c1': s1,
#             'cohen_d': cohen_d, 't_stat': t_stat, 'p_val': p_val,
#             'frac_active_c0': frac_active_0, 'frac_active_c1': frac_active_1,
#             'is_control': label.startswith('CONTROL'),
#         })

#     return results


# # ══════════════════════════════════════════════════════════════════════
# # Visualization Pages
# # ══════════════════════════════════════════════════════════════════════

# def pg_boxplots(pdf, activations, labels, features, class_names, exp_cfg, title_extra):
#     """Box plots: class 0 vs class 1 activation per target feature."""
#     target_feats = [(k, v) for k, v in features.items() if not v.startswith('CONTROL')]
#     if not target_feats:
#         return

#     n = len(target_feats)
#     ncols = min(4, n)
#     nrows = (n + ncols - 1) // ncols
#     fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

#     for idx, ((sae_layer, width, feat_id), label) in enumerate(target_feats):
#         ax = axes[idx // ncols, idx % ncols]
#         col = activations.get((sae_layer, width))
#         if col is None:
#             ax.set_visible(False); continue

#         act = col[:, feat_id]
#         c0 = act[labels == 0]
#         c1 = act[labels == 1]

#         bp = ax.boxplot([c0, c1], labels=[class_names[0], class_names[1]],
#                         patch_artist=True, widths=0.5)
#         bp['boxes'][0].set_facecolor('#FF5722'); bp['boxes'][0].set_alpha(0.6)
#         bp['boxes'][1].set_facecolor('#2196F3'); bp['boxes'][1].set_alpha(0.6)

#         # Scatter overlay
#         for i, data in enumerate([c0, c1]):
#             x = np.random.normal(i + 1, 0.04, len(data))
#             ax.scatter(x, data, alpha=0.05, s=3, c='black')

#         t, p = stats.ttest_ind(c1, c0, equal_var=False)
#         sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
#         ax.set_title(f'L{sae_layer}_{width} #{feat_id}\n"{label}" (p={p:.2e} {sig})',
#                      fontsize=8, fontweight='bold')
#         ax.tick_params(labelsize=7)
#         ax.grid(True, alpha=0.15, axis='y')

#     for idx in range(n, nrows * ncols):
#         axes[idx // ncols, idx % ncols].set_visible(False)

#     fig.suptitle(f'{exp_cfg["name"]} — SAE Feature Activations\n'
#                  f'{title_extra} | Target features (NeuronPedia)',
#                  fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     pdf.savefig(fig, dpi=150); plt.close()


# def pg_control(pdf, activations, labels, features, class_names, exp_cfg, title_extra):
#     """Control features — should show no significant difference."""
#     ctrl_feats = [(k, v) for k, v in features.items() if v.startswith('CONTROL')]
#     if not ctrl_feats:
#         return

#     n = len(ctrl_feats)
#     fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)

#     for idx, ((sae_layer, width, feat_id), label) in enumerate(ctrl_feats):
#         ax = axes[0, idx]
#         col = activations.get((sae_layer, width))
#         if col is None:
#             ax.set_visible(False); continue

#         act = col[:, feat_id]
#         c0, c1 = act[labels == 0], act[labels == 1]
#         bp = ax.boxplot([c0, c1], labels=[class_names[0], class_names[1]],
#                         patch_artist=True, widths=0.5)
#         bp['boxes'][0].set_facecolor('#FF5722'); bp['boxes'][0].set_alpha(0.6)
#         bp['boxes'][1].set_facecolor('#2196F3'); bp['boxes'][1].set_alpha(0.6)

#         t, p = stats.ttest_ind(c1, c0, equal_var=False)
#         sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
#         ax.set_title(f'CONTROL #{feat_id}\np={p:.2e} {sig}', fontsize=8)
#         ax.tick_params(labelsize=7)

#     fig.suptitle(f'{exp_cfg["name"]} — Control Features (should be ns)\n{title_extra}',
#                  fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#     pdf.savefig(fig, dpi=150); plt.close()


# def pg_activation_rate(pdf, results, exp_cfg, class_names, title_extra):
#     """Bar chart: fraction of samples where feature is active (>0), by class."""
#     target = [r for r in results if not r['is_control']]
#     if not target:
#         return

#     fig, ax = plt.subplots(figsize=(max(14, len(target) * 1.2), 7))
#     x = np.arange(len(target))
#     w = 0.35

#     bars0 = [r['frac_active_c0'] for r in target]
#     bars1 = [r['frac_active_c1'] for r in target]
#     labels = [f"L{r['sae_layer']}_{r['width']}\n#{r['feat_id']}\n{r['label'][:15]}"
#               for r in target]

#     ax.bar(x - w/2, bars0, w, label=class_names[0], color='#FF5722', alpha=0.7)
#     ax.bar(x + w/2, bars1, w, label=class_names[1], color='#2196F3', alpha=0.7)

#     # Add significance stars
#     for i, r in enumerate(target):
#         sig = '***' if r['p_val'] < 0.001 else '**' if r['p_val'] < 0.01 else '*' if r['p_val'] < 0.05 else ''
#         if sig:
#             ax.text(i, max(bars0[i], bars1[i]) + 0.02, sig,
#                     ha='center', fontsize=9, fontweight='bold', color='red')

#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
#     ax.set_ylabel('Fraction of samples with activation > 0')
#     ax.set_title(f'{exp_cfg["name"]} — Feature Activation Rate\n{title_extra}',
#                  fontweight='bold')
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.15, axis='y')
#     ax.set_ylim(0, 1.05)
#     plt.tight_layout()
#     pdf.savefig(fig, dpi=150); plt.close()


# def pg_effect_size(pdf, results, exp_cfg, class_names, title_extra):
#     """Cohen's d effect size bar chart."""
#     target = [r for r in results if not r['is_control']]
#     ctrl = [r for r in results if r['is_control']]
#     all_r = target + ctrl

#     fig, ax = plt.subplots(figsize=(max(14, len(all_r) * 1.2), 7))
#     x = np.arange(len(all_r))
#     colors = ['#4CAF50' if not r['is_control'] else '#9E9E9E' for r in all_r]
#     ds = [r['cohen_d'] for r in all_r]
#     labels_list = [f"{'CTRL ' if r['is_control'] else ''}L{r['sae_layer']}\n#{r['feat_id']}"
#                    for r in all_r]

#     bars = ax.bar(x, ds, color=colors, alpha=0.8)

#     # Significance markers
#     for i, r in enumerate(all_r):
#         sig = '***' if r['p_val'] < 0.001 else '**' if r['p_val'] < 0.01 else '*' if r['p_val'] < 0.05 else ''
#         if sig:
#             y = ds[i] + (0.02 if ds[i] >= 0 else -0.05)
#             ax.text(i, y, sig, ha='center', fontsize=8, color='red')

#     ax.set_xticks(x)
#     ax.set_xticklabels(labels_list, fontsize=6, rotation=45, ha='right')
#     ax.set_ylabel("Cohen's d (positive = higher in class 1)")
#     ax.axhline(0, color='black', lw=0.8)
#     ax.axhline(0.2, color='gray', ls=':', alpha=0.3)
#     ax.axhline(-0.2, color='gray', ls=':', alpha=0.3)
#     ax.set_title(f'{exp_cfg["name"]} — Effect Size (Cohen\'s d)\n'
#                  f'{title_extra} | Green=target, Gray=control',
#                  fontweight='bold')
#     ax.grid(True, alpha=0.15, axis='y')
#     plt.tight_layout()
#     pdf.savefig(fig, dpi=150); plt.close()


# def pg_summary(pdf, results, exp_cfg, class_names, title_extra):
#     """Summary table."""
#     fig, ax = plt.subplots(figsize=(20, 12)); ax.axis('off')
#     lines = [f'{exp_cfg["name"]} — SAE Feature Summary | {title_extra}\n\n']
#     lines.append(f'{"Layer":>6s} {"Width":>6s} {"Feat#":>8s} {"Label":>25s} '
#                  f'{"mean_c0":>8s} {"mean_c1":>8s} {"d":>7s} '
#                  f'{"p-val":>10s} {"sig":>4s} '
#                  f'{"act%_c0":>8s} {"act%_c1":>8s}\n')
#     lines.append('─' * 120 + '\n')

#     for r in sorted(results, key=lambda x: (x['is_control'], x['p_val'])):
#         sig = '***' if r['p_val'] < 0.001 else '**' if r['p_val'] < 0.01 else '*' if r['p_val'] < 0.05 else 'ns'
#         lines.append(
#             f'L{r["sae_layer"]:>4d} {r["width"]:>6s} {r["feat_id"]:>8d} '
#             f'{r["label"]:>25s} '
#             f'{r["mean_c0"]:>8.4f} {r["mean_c1"]:>8.4f} {r["cohen_d"]:>+7.3f} '
#             f'{r["p_val"]:>10.2e} {sig:>4s} '
#             f'{r["frac_active_c0"]:>8.1%} {r["frac_active_c1"]:>8.1%}\n')

#     lines.append('─' * 120 + '\n')
#     lines.append(f'\nc0 = {class_names[0]}, c1 = {class_names[1]}\n')
#     lines.append(f"Cohen's d: |d|>0.2 small, |d|>0.5 medium, |d|>0.8 large\n")
#     lines.append(f'act% = fraction of samples where feature activation > 0\n')

#     n_sig = sum(1 for r in results if r['p_val'] < 0.05 and not r['is_control'])
#     n_target = sum(1 for r in results if not r['is_control'])
#     n_ctrl_sig = sum(1 for r in results if r['p_val'] < 0.05 and r['is_control'])
#     lines.append(f'\nSignificant target features: {n_sig}/{n_target}\n')
#     lines.append(f'Significant control features: {n_ctrl_sig}/{len(CONTROL_FEATURES)} '
#                  f'(should be ~0)\n')

#     ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
#             fontsize=7, va='top', fontfamily='monospace')
#     fig.suptitle(f'Summary', fontsize=13, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # Top-K Discovery (not just target features)
# # ══════════════════════════════════════════════════════════════════════

# def pg_top_features(pdf, activations, labels, class_names, exp_cfg, sae_layer, width, title_extra):
#     """Find and display top-20 most differentially activated features (any feature)."""
#     col = activations.get((sae_layer, width))
#     if col is None:
#         return

#     n_features = col.shape[1]
#     mean_c0 = col[labels == 0].mean(0)
#     mean_c1 = col[labels == 1].mean(0)
#     diff = mean_c1 - mean_c0  # positive = higher in class 1

#     # Top 10 higher in class 1, top 10 higher in class 0
#     top_c1 = np.argsort(diff)[-10:][::-1]
#     top_c0 = np.argsort(diff)[:10]

#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))

#     for ax, idxs, title, color in [
#         (axes[0], top_c1, f'Top 10 higher in {class_names[1]}', '#2196F3'),
#         (axes[1], top_c0, f'Top 10 higher in {class_names[0]}', '#FF5722'),
#     ]:
#         feats = [f'#{i}' for i in idxs]
#         vals_c0 = [mean_c0[i] for i in idxs]
#         vals_c1 = [mean_c1[i] for i in idxs]
#         x = np.arange(len(idxs))
#         w = 0.35
#         ax.barh(x - w/2, vals_c0, w, label=class_names[0], color='#FF5722', alpha=0.7)
#         ax.barh(x + w/2, vals_c1, w, label=class_names[1], color='#2196F3', alpha=0.7)
#         ax.set_yticks(x)
#         ax.set_yticklabels(feats, fontsize=8)
#         ax.set_xlabel('Mean Activation')
#         ax.set_title(title, fontweight='bold', fontsize=10)
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.15, axis='x')

#     fig.suptitle(f'{exp_cfg["name"]} — Top-20 Differential Features\n'
#                  f'L{sae_layer} {width} | {title_extra}',
#                  fontsize=12, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     pdf.savefig(fig, dpi=150); plt.close()


# # ══════════════════════════════════════════════════════════════════════
# # Main
# # ══════════════════════════════════════════════════════════════════════

# def run_one(pdf, iter_name, exp, cfg_id):
#     """Run SAE analysis for one (iter, exp, cfg) combination."""
#     exp_cfg = EXP_CONFIG[exp]
#     clabel = config_label(exp, cfg_id)
#     tag = MODEL_REGISTRY['12b']['tag']

#     # Class names
#     if 'employed' in exp_cfg['label_col']:
#         class_names = ['Unemployed', 'Employed']
#     else:
#         class_names = ['Weekend', 'Weekday']

#     title_extra = f'{iter_name} / cfg{cfg_id} ({clabel})'

#     # Load hidden states
#     hidden_dir = os.path.join(BASE_DIR, f'outputs_{iter_name}', 'hidden_states', tag)
#     hpath = os.path.join(hidden_dir, f"{exp}_cfg{cfg_id}.npz")
#     if not os.path.exists(hpath):
#         print(f"  SKIP: {hpath} not found")
#         return

#     data = np.load(hpath, allow_pickle=True)
#     X = data['hidden_states']  # (N, 49, 3840)
#     y = data['labels']         # (N,)
#     print(f"\n  {title_extra}: N={X.shape[0]}, "
#           f"c0={int((y==0).sum())}, c1={int((y==1).sum())}")

#     features = ALL_FEATURES[exp]

#     # Find needed (layer, width) SAEs
#     needed_saes = set()
#     for (sae_layer, width, _), _ in features.items():
#         needed_saes.add((sae_layer, width))

#     # Load SAEs and encode
#     activations = {}  # (sae_layer, width) → (N, n_features) numpy
#     for sae_layer, width in sorted(needed_saes):
#         sae_path = get_sae_path(sae_layer, width)
#         if sae_path is None:
#             print(f"    SAE L{sae_layer}_{width}: NOT FOUND (skip)")
#             continue

#         print(f"    Loading SAE L{sae_layer}_{width}...", end=' ')
#         sae = load_sae(sae_path)

#         # Hidden state index: sae_layer=41 → X[:, 42, :] (index 0 = embedding)
#         h_idx = sae_layer + 1
#         h = X[:, h_idx, :]  # (N, 3840) float32

#         print(f"encoding {h.shape[0]} samples...", end=' ')
#         act = encode_sae(h, sae)
#         n_active = (act > 0).sum(1).mean()
#         print(f"done (avg {n_active:.0f} active features)")

#         activations[(sae_layer, width)] = act

#         # Free SAE memory
#         del sae

#     if not activations:
#         print("    No SAEs loaded, skipping")
#         return

#     # Analyze
#     results = analyze_features(activations, y, features, class_names)

#     # Generate pages
#     print(f"    Generating PDF pages...")
#     pg_boxplots(pdf, activations, y, features, class_names, exp_cfg, title_extra)
#     pg_control(pdf, activations, y, features, class_names, exp_cfg, title_extra)
#     pg_activation_rate(pdf, results, exp_cfg, class_names, title_extra)
#     pg_effect_size(pdf, results, exp_cfg, class_names, title_extra)

#     # Top-K discovery for main SAE layers
#     for sae_layer, width in sorted(needed_saes):
#         if (sae_layer, width) in activations:
#             pg_top_features(pdf, activations, y, class_names, exp_cfg,
#                            sae_layer, width, title_extra)

#     pg_summary(pdf, results, exp_cfg, class_names, title_extra)

#     # Print highlights
#     sig = [r for r in results if r['p_val'] < 0.05 and not r['is_control']]
#     print(f"    Significant features: {len(sig)}/{len([r for r in results if not r['is_control']])}")
#     for r in sorted(sig, key=lambda x: x['p_val'])[:5]:
#         print(f"      L{r['sae_layer']}_{r['width']} #{r['feat_id']} "
#               f'"{r["label"]}": d={r["cohen_d"]:+.3f} p={r["p_val"]:.2e}')


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--exp', nargs='+', default=['exp2', 'exp1a'])
#     parser.add_argument('--iter', nargs='+', default=['v7', 'v8'])
#     parser.add_argument('--cfgs', nargs='+', type=int, default=None)
#     args = parser.parse_args()

#     # Default cfgs per experiment
#     default_cfgs = {
#         'exp2': [5, 6],
#         'exp1a': [3, 4],
#     }

#     pdf_path = os.path.join(OUT_DIR, 'sae_feature_analysis.pdf')
#     print(f"{'='*60}")
#     print(f"SAE Feature Analysis — GemmaScope 2")
#     print(f"SAE dir: {SAE_DIR}")
#     print(f"PDF: {pdf_path}")
#     print(f"{'='*60}")

#     with PdfPages(pdf_path) as pdf:
#         for exp in args.exp:
#             cfgs = args.cfgs if args.cfgs else default_cfgs.get(exp, [5, 6])
#             for iter_name in args.iter:
#                 for cfg_id in cfgs:
#                     run_one(pdf, iter_name, exp, cfg_id)

#     print(f"\n{'='*60}")
#     print(f"Done! → {pdf_path}")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAE Feature Analysis — Do GPS-derived hidden states activate employment/weekday features?

Loads pre-trained GemmaScope 2 SAEs (JumpReLU), encodes Phase 1 hidden states,
and checks if NeuronPedia-labeled features differentially activate between classes.

Two experiments:
  Exp2 (employment):  employed vs unemployed → check employment SAE features
  Exp1a (weekday):    weekday vs weekend → check weekday/holiday SAE features

Output: sae/output/sae_feature_analysis.pdf

Usage:
    python sae/sae_analysis.py                         # all exps, V7 + V8
    python sae/sae_analysis.py --exp exp2 --iter v7    # specific
    python sae/sae_analysis.py --exp exp1a --cfgs 3 4  # specific cfgs
"""
import sys, os, argparse, json
import numpy as np
import torch
from safetensors.torch import load_file
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_REGISTRY, EXP_CONFIG, config_label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_DIR = os.path.join(BASE_DIR, 'sae', 'resid_post')
OUT_DIR = os.path.join(BASE_DIR, 'sae', 'output')
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Target Features from NeuronPedia
# ══════════════════════════════════════════════════════════════════════

# Exp2: Employment concept
EXP2_FEATURES = {
    # Existing features
    # Layer 41, 65k
    (41, '65k', 22394): 'employment',
    (41, '65k', 10086): 'workplace or office',
    (41, '65k', 45212): 'job losses',

    # Layer 31, 65k
    (31, '65k', 36004): 'doing or working',
    (31, '65k', 41583): 'free time',

    # Layer 41, 262k
    (41, '262k', 171963): 'employment',
    (41, '262k', 19373): 'employment',
    (41, '262k', 202142): 'employment',
    (41, '262k', 28067): 'jobs and employment',
    (41, '262k', 187898): 'employment',
    (41, '262k', 97122): 'employment',
    (41, '262k', 247538): 'business and employment',
    (41, '262k', 40558): 'business, employment',
    (41, '262k', 2949): 'employment quantity',
    (41, '262k', 22733): 'hiring, employment',

    # Layer 12, 262k
    (12, '262k', 172041): 'employment and law',

    # Newly added features
    # Layer 41, 16k
    (41, '16k', 11298): 'job career',
    (41, '16k', 3831): 'employment status',

    # Layer 31, 262k
    (31, '262k', 9855): 'job creation and employment',
    (31, '262k', 52): 'job loss or employment',
    (31, '262k', 18614): 'unemployment and job opportunities',

    # Layer 41, 262k
    (41, '262k', 205293): 'employment and salary',
    (41, '262k', 17949): 'employment and empowerment',
    (41, '262k', 182966): 'work employment',
    (41, '262k', 169042): 'subsidies, unemployment',
    (41, '262k', 230743): 'unemployment',

    # Layer 31, 16k
    (31, '16k', 6210): 'unemployment and employment situations',

    # Layer 12, 16k
    (12, '16k', 6168): 'employment and unemployment',

    # Layer 12, 65k
    (12, '65k', 58479): 'unemployment rate',

    # Layer 41, 65k
    (41, '65k', 28450): 'employment status',
    (41, '65k', 34730): 'unemployment',
}

# Exp1a: Weekday/Weekend concept
EXP1A_FEATURES = {
    # Layer 41, 65k
    (41, '65k', 47289): 'week',
    (41, '65k', 20474): 'week',
    (41, '65k', 65361): 'holidays',
    (41, '65k', 7422): 'holidays',
    (41, '65k', 27819): 'Week',
    (41, '65k', 42844): 'holiday',
    (41, '65k', 37325): 'days of week',
    (41, '65k', 16397): 'days of week',
    # Layer 31, 65k
    (31, '65k', 38778): 'morning and weekend events',
    # Layer 12, 65k
    (12, '65k', 19244): 'holiday weekends',
}

# Control features (random, should NOT differ between groups)
CONTROL_FEATURES = {
    (41, '65k', 100): 'CONTROL_100',
    (41, '65k', 500): 'CONTROL_500',
    (41, '65k', 1000): 'CONTROL_1000',
    (41, '65k', 5000): 'CONTROL_5000',
    (41, '65k', 10000): 'CONTROL_10000',
}

ALL_FEATURES = {
    'exp2': {**EXP2_FEATURES, **CONTROL_FEATURES},
    'exp1a': {**EXP1A_FEATURES, **CONTROL_FEATURES},
}


# ══════════════════════════════════════════════════════════════════════
# SAE Loading + Encoding
# ══════════════════════════════════════════════════════════════════════

def get_sae_path(layer, width):
    """Map (layer, width) to local SAE directory."""
    w = {'16k': '16k', '65k': '65k', '262k': '262k'}[width]
    dirname = f"layer_{layer}_width_{w}_l0_medium"
    path = os.path.join(SAE_DIR, dirname)
    if not os.path.exists(path):
        return None
    return path


def load_sae(sae_path):
    """Load JumpReLU SAE parameters."""
    params = load_file(os.path.join(sae_path, 'params.safetensors'))
    with open(os.path.join(sae_path, 'config.json')) as f:
        config = json.load(f)

    return {
        'w_enc': params['w_enc'],           # (hidden_dim, n_features)
        'b_enc': params['b_enc'],           # (n_features,)
        'w_dec': params['w_dec'],           # (n_features, hidden_dim)
        'b_dec': params['b_dec'],           # (hidden_dim,)
        'threshold': params['threshold'],    # (n_features,) JumpReLU threshold
        'config': config,
    }


def encode_sae(hidden_states, sae):
    """Encode hidden states through JumpReLU SAE → sparse feature activations."""
    h = torch.from_numpy(hidden_states).float()
    b_dec = sae['b_dec'].float()
    w_enc = sae['w_enc'].float()
    b_enc = sae['b_enc'].float()
    threshold = sae['threshold'].float()

    # JumpReLU: z = max(0, W_enc @ (x - b_dec) + b_enc) where pre_act > threshold
    pre_act = (h - b_dec) @ w_enc + b_enc
    z = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))

    return z.numpy()  # (n_samples, n_features)


# ══════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════

def analyze_features(activations, labels, features, class_names):
    """Compute per-feature statistics: mean activation, t-test, effect size."""
    results = []
    for (sae_layer, width, feat_id), label in features.items():
        col = activations.get((sae_layer, width))
        if col is None:
            continue
        act = col[:, feat_id]
        c0 = act[labels == 0]
        c1 = act[labels == 1]

        # Stats
        m0, m1 = c0.mean(), c1.mean()
        s0, s1 = c0.std(), c1.std()
        pooled_std = np.sqrt((s0**2 + s1**2) / 2) if (s0 + s1) > 0 else 1e-10
        cohen_d = (m1 - m0) / pooled_std
        t_stat, p_val = stats.ttest_ind(c1, c0, equal_var=False)
        frac_active_0 = (c0 > 0).mean()
        frac_active_1 = (c1 > 0).mean()

        results.append({
            'sae_layer': sae_layer, 'width': width, 'feat_id': feat_id,
            'label': label,
            'mean_c0': m0, 'mean_c1': m1,
            'std_c0': s0, 'std_c1': s1,
            'cohen_d': cohen_d, 't_stat': t_stat, 'p_val': p_val,
            'frac_active_c0': frac_active_0, 'frac_active_c1': frac_active_1,
            'is_control': label.startswith('CONTROL'),
        })

    return results


# ══════════════════════════════════════════════════════════════════════
# Visualization Pages
# ══════════════════════════════════════════════════════════════════════

def pg_boxplots(pdf, activations, labels, features, class_names, exp_cfg, title_extra):
    """Box plots: class 0 vs class 1 activation per target feature."""
    target_feats = [(k, v) for k, v in features.items() if not v.startswith('CONTROL')]
    if not target_feats:
        return

    n = len(target_feats)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, ((sae_layer, width, feat_id), label) in enumerate(target_feats):
        ax = axes[idx // ncols, idx % ncols]
        col = activations.get((sae_layer, width))
        if col is None:
            ax.set_visible(False); continue

        act = col[:, feat_id]
        c0 = act[labels == 0]
        c1 = act[labels == 1]

        bp = ax.boxplot([c0, c1], labels=[class_names[0], class_names[1]],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#FF5722'); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#2196F3'); bp['boxes'][1].set_alpha(0.6)

        # Scatter overlay
        for i, data in enumerate([c0, c1]):
            x = np.random.normal(i + 1, 0.04, len(data))
            ax.scatter(x, data, alpha=0.05, s=3, c='black')

        t, p = stats.ttest_ind(c1, c0, equal_var=False)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_title(f'L{sae_layer}_{width} #{feat_id}\n"{label}" (p={p:.2e} {sig})',
                     fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, axis='y')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{exp_cfg["name"]} — SAE Feature Activations\n'
                 f'{title_extra} | Target features (NeuronPedia)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150); plt.close()


def pg_control(pdf, activations, labels, features, class_names, exp_cfg, title_extra):
    """Control features — should show no significant difference."""
    ctrl_feats = [(k, v) for k, v in features.items() if v.startswith('CONTROL')]
    if not ctrl_feats:
        return

    n = len(ctrl_feats)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)

    for idx, ((sae_layer, width, feat_id), label) in enumerate(ctrl_feats):
        ax = axes[0, idx]
        col = activations.get((sae_layer, width))
        if col is None:
            ax.set_visible(False); continue

        act = col[:, feat_id]
        c0, c1 = act[labels == 0], act[labels == 1]
        bp = ax.boxplot([c0, c1], labels=[class_names[0], class_names[1]],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#FF5722'); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#2196F3'); bp['boxes'][1].set_alpha(0.6)

        t, p = stats.ttest_ind(c1, c0, equal_var=False)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_title(f'CONTROL #{feat_id}\np={p:.2e} {sig}', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f'{exp_cfg["name"]} — Control Features (should be ns)\n{title_extra}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=150); plt.close()


def pg_activation_rate(pdf, results, exp_cfg, class_names, title_extra):
    """Bar chart: fraction of samples where feature is active (>0), by class."""
    target = [r for r in results if not r['is_control']]
    if not target:
        return

    fig, ax = plt.subplots(figsize=(max(14, len(target) * 1.2), 7))
    x = np.arange(len(target))
    w = 0.35

    bars0 = [r['frac_active_c0'] for r in target]
    bars1 = [r['frac_active_c1'] for r in target]
    labels = [f"L{r['sae_layer']}_{r['width']}\n#{r['feat_id']}\n{r['label'][:15]}"
              for r in target]

    ax.bar(x - w/2, bars0, w, label=class_names[0], color='#FF5722', alpha=0.7)
    ax.bar(x + w/2, bars1, w, label=class_names[1], color='#2196F3', alpha=0.7)

    # Add significance stars
    for i, r in enumerate(target):
        sig = '***' if r['p_val'] < 0.001 else '**' if r['p_val'] < 0.01 else '*' if r['p_val'] < 0.05 else ''
        if sig:
            ax.text(i, max(bars0[i], bars1[i]) + 0.02, sig,
                    ha='center', fontsize=9, fontweight='bold', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('Fraction of samples with activation > 0')
    ax.set_title(f'{exp_cfg["name"]} — Feature Activation Rate\n{title_extra}',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis='y')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


def pg_effect_size(pdf, results, exp_cfg, class_names, title_extra):
    """Cohen's d effect size bar chart."""
    target = [r for r in results if not r['is_control']]
    ctrl = [r for r in results if r['is_control']]
    all_r = target + ctrl

    fig, ax = plt.subplots(figsize=(max(14, len(all_r) * 1.2), 7))
    x = np.arange(len(all_r))
    colors = ['#4CAF50' if not r['is_control'] else '#9E9E9E' for r in all_r]
    ds = [r['cohen_d'] for r in all_r]
    labels_list = [f"{'CTRL ' if r['is_control'] else ''}L{r['sae_layer']}\n#{r['feat_id']}"
                   for r in all_r]

    bars = ax.bar(x, ds, color=colors, alpha=0.8)

    # Significance markers
    for i, r in enumerate(all_r):
        sig = '***' if r['p_val'] < 0.001 else '**' if r['p_val'] < 0.01 else '*' if r['p_val'] < 0.05 else ''
        if sig:
            y = ds[i] + (0.02 if ds[i] >= 0 else -0.05)
            ax.text(i, y, sig, ha='center', fontsize=8, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel("Cohen's d (positive = higher in class 1)")
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.2, color='gray', ls=':', alpha=0.3)
    ax.axhline(-0.2, color='gray', ls=':', alpha=0.3)
    ax.set_title(f'{exp_cfg["name"]} — Effect Size (Cohen\'s d)\n'
                 f'{title_extra} | Green=target, Gray=control',
                 fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


def pg_summary(pdf, results, exp_cfg, class_names, title_extra):
    """Summary table."""
    fig, ax = plt.subplots(figsize=(20, 12)); ax.axis('off')
    lines = [f'{exp_cfg["name"]} — SAE Feature Summary | {title_extra}\n\n']
    lines.append(f'{"Layer":>6s} {"Width":>6s} {"Feat#":>8s} {"Label":>25s} '
                 f'{"mean_c0":>8s} {"mean_c1":>8s} {"d":>7s} '
                 f'{"p-val":>10s} {"sig":>4s} '
                 f'{"act%_c0":>8s} {"act%_c1":>8s}\n')
    lines.append('─' * 120 + '\n')

    for r in sorted(results, key=lambda x: (x['is_control'], x['p_val'])):
        sig = '***' if r['p_val'] < 0.001 else '**' if r['p_val'] < 0.01 else '*' if r['p_val'] < 0.05 else 'ns'
        lines.append(
            f'L{r["sae_layer"]:>4d} {r["width"]:>6s} {r["feat_id"]:>8d} '
            f'{r["label"]:>25s} '
            f'{r["mean_c0"]:>8.4f} {r["mean_c1"]:>8.4f} {r["cohen_d"]:>+7.3f} '
            f'{r["p_val"]:>10.2e} {sig:>4s} '
            f'{r["frac_active_c0"]:>8.1%} {r["frac_active_c1"]:>8.1%}\n')

    lines.append('─' * 120 + '\n')
    lines.append(f'\nc0 = {class_names[0]}, c1 = {class_names[1]}\n')
    lines.append(f"Cohen's d: |d|>0.2 small, |d|>0.5 medium, |d|>0.8 large\n")
    lines.append(f'act% = fraction of samples where feature activation > 0\n')

    n_sig = sum(1 for r in results if r['p_val'] < 0.05 and not r['is_control'])
    n_target = sum(1 for r in results if not r['is_control'])
    n_ctrl_sig = sum(1 for r in results if r['p_val'] < 0.05 and r['is_control'])
    lines.append(f'\nSignificant target features: {n_sig}/{n_target}\n')
    lines.append(f'Significant control features: {n_ctrl_sig}/{len(CONTROL_FEATURES)} '
                 f'(should be ~0)\n')

    ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace')
    fig.suptitle(f'Summary', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Top-K Discovery (not just target features)
# ══════════════════════════════════════════════════════════════════════

def pg_top_features(pdf, activations, labels, class_names, exp_cfg, sae_layer, width, title_extra):
    """Find and display top-20 most differentially activated features (any feature)."""
    col = activations.get((sae_layer, width))
    if col is None:
        return

    n_features = col.shape[1]
    mean_c0 = col[labels == 0].mean(0)
    mean_c1 = col[labels == 1].mean(0)
    diff = mean_c1 - mean_c0  # positive = higher in class 1

    # Top 10 higher in class 1, top 10 higher in class 0
    top_c1 = np.argsort(diff)[-10:][::-1]
    top_c0 = np.argsort(diff)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, idxs, title, color in [
        (axes[0], top_c1, f'Top 10 higher in {class_names[1]}', '#2196F3'),
        (axes[1], top_c0, f'Top 10 higher in {class_names[0]}', '#FF5722'),
    ]:
        feats = [f'#{i}' for i in idxs]
        vals_c0 = [mean_c0[i] for i in idxs]
        vals_c1 = [mean_c1[i] for i in idxs]
        x = np.arange(len(idxs))
        w = 0.35
        ax.barh(x - w/2, vals_c0, w, label=class_names[0], color='#FF5722', alpha=0.7)
        ax.barh(x + w/2, vals_c1, w, label=class_names[1], color='#2196F3', alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel('Mean Activation')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15, axis='x')

    fig.suptitle(f'{exp_cfg["name"]} — Top-20 Differential Features\n'
                 f'L{sae_layer} {width} | {title_extra}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def run_one(pdf, iter_name, exp, cfg_id):
    """Run SAE analysis for one (iter, exp, cfg) combination."""
    exp_cfg = EXP_CONFIG[exp]
    clabel = config_label(exp, cfg_id)
    tag = MODEL_REGISTRY['12b']['tag']

    # Class names
    if 'employed' in exp_cfg['label_col']:
        class_names = ['Unemployed', 'Employed']
    else:
        class_names = ['Weekend', 'Weekday']

    title_extra = f'{iter_name} / cfg{cfg_id} ({clabel})'

    # Load hidden states
    hidden_dir = os.path.join(BASE_DIR, f'outputs_{iter_name}', 'hidden_states', tag)
    hpath = os.path.join(hidden_dir, f"{exp}_cfg{cfg_id}.npz")
    if not os.path.exists(hpath):
        print(f"  SKIP: {hpath} not found")
        return

    data = np.load(hpath, allow_pickle=True)
    X = data['hidden_states']  # (N, 49, 3840)
    y = data['labels']         # (N,)
    print(f"\n  {title_extra}: N={X.shape[0]}, "
          f"c0={int((y==0).sum())}, c1={int((y==1).sum())}")

    features = ALL_FEATURES[exp]

    # Find needed (layer, width) SAEs
    needed_saes = set()
    for (sae_layer, width, _), _ in features.items():
        needed_saes.add((sae_layer, width))

    # Load SAEs and encode
    activations = {}  # (sae_layer, width) → (N, n_features) numpy
    for sae_layer, width in sorted(needed_saes):
        sae_path = get_sae_path(sae_layer, width)
        if sae_path is None:
            print(f"    SAE L{sae_layer}_{width}: NOT FOUND (skip)")
            continue

        print(f"    Loading SAE L{sae_layer}_{width}...", end=' ')
        sae = load_sae(sae_path)

        # Hidden state index: sae_layer=41 → X[:, 42, :] (index 0 = embedding)
        h_idx = sae_layer + 1
        h = X[:, h_idx, :]  # (N, 3840) float32

        print(f"encoding {h.shape[0]} samples...", end=' ')
        act = encode_sae(h, sae)
        n_active = (act > 0).sum(1).mean()
        print(f"done (avg {n_active:.0f} active features)")

        activations[(sae_layer, width)] = act

        # Free SAE memory
        del sae

    if not activations:
        print("    No SAEs loaded, skipping")
        return

    # Analyze
    results = analyze_features(activations, y, features, class_names)

    # Generate pages
    print(f"    Generating PDF pages...")
    pg_boxplots(pdf, activations, y, features, class_names, exp_cfg, title_extra)
    pg_control(pdf, activations, y, features, class_names, exp_cfg, title_extra)
    pg_activation_rate(pdf, results, exp_cfg, class_names, title_extra)
    pg_effect_size(pdf, results, exp_cfg, class_names, title_extra)

    # Top-K discovery for main SAE layers
    for sae_layer, width in sorted(needed_saes):
        if (sae_layer, width) in activations:
            pg_top_features(pdf, activations, y, class_names, exp_cfg,
                           sae_layer, width, title_extra)

    pg_summary(pdf, results, exp_cfg, class_names, title_extra)

    # Print highlights
    sig = [r for r in results if r['p_val'] < 0.05 and not r['is_control']]
    print(f"    Significant features: {len(sig)}/{len([r for r in results if not r['is_control']])}")
    for r in sorted(sig, key=lambda x: x['p_val'])[:5]:
        print(f"      L{r['sae_layer']}_{r['width']} #{r['feat_id']} "
              f'"{r["label"]}": d={r["cohen_d"]:+.3f} p={r["p_val"]:.2e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp2', 'exp1a'])
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'])
    parser.add_argument('--cfgs', nargs='+', type=int, default=None)
    args = parser.parse_args()

    # Default cfgs per experiment
    default_cfgs = {
        'exp2': [5, 6],
        'exp1a': [3, 4],
    }

    pdf_path = os.path.join(OUT_DIR, 'sae_feature_analysis.pdf')
    print(f"{'='*60}")
    print(f"SAE Feature Analysis — GemmaScope 2")
    print(f"SAE dir: {SAE_DIR}")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}")

    with PdfPages(pdf_path) as pdf:
        for exp in args.exp:
            cfgs = args.cfgs if args.cfgs else default_cfgs.get(exp, [5, 6])
            for iter_name in args.iter:
                for cfg_id in cfgs:
                    run_one(pdf, iter_name, exp, cfg_id)

    print(f"\n{'='*60}")
    print(f"Done! → {pdf_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
