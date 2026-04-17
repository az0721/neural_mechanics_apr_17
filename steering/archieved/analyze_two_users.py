# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Quick steering analysis for first 2 completed users.
# Compare employed vs unemployed across all prompts, layers, coefficients.

# Usage:
#     python steering/analyze_two_users.py
# """
# import json, os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from config import OUTPUT_DIR

# PERUSER_DIR = os.path.join(OUTPUT_DIR, 'steering', 'per_user')
# RESULTS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'results')
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # Two completed users
# FILES = {
#     'emp':   '5db7e19202fefe287fd8ad03bb8a64aa.json',
#     'unemp': '608e0e7ae296cc575fc2226d5d882726.json',
# }

# PROMPTS = {
#     'binary':     {'target': 'Yes', 'label': 'P(Yes = employed)'},
#     'behavioral': {'target': 'B',   'label': 'P(B = workplace)'},
#     'routine':    {'target': 'A',   'label': 'P(A = 9-to-5 job)'},
# }

# COEFFS = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
# LAYERS = list(range(23, 48))


# def load_users():
#     users = {}
#     for tag, fname in FILES.items():
#         path = os.path.join(PERUSER_DIR, fname)
#         with open(path) as f:
#             users[tag] = json.load(f)
#         print(f"Loaded {tag}: {users[tag]['meta']['user'][:16]} "
#               f"(employed={users[tag]['meta']['is_employed']})")
#     return users


# def get_probs(d, cfg, qkey, target):
#     """Extract probability matrix (n_layers, n_coeffs) for one user."""
#     mat = np.full((len(LAYERS), len(COEFFS)), np.nan)
#     for li, layer in enumerate(LAYERS):
#         for ci, coeff in enumerate(COEFFS):
#             key = str((cfg, qkey, layer, coeff))
#             p = d['logits'].get(key, {})
#             mat[li, ci] = p.get(target, np.nan)
#     return mat


# def fig1_side_by_side_heatmaps(users):
#     """Heatmap: layers × coeffs, emp vs unemp, per prompt."""
#     for cfg in [5, 6]:
#         for qkey, info in PROMPTS.items():
#             fig, axes = plt.subplots(1, 3, figsize=(22, 8))

#             mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
#             mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

#             zero_ci = COEFFS.index(0)

#             # Raw probabilities
#             for ax, mat, title in [(axes[0], mat_e, 'Employed (raw P)'),
#                                    (axes[1], mat_u, 'Unemployed (raw P)')]:
#                 im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
#                                origin='lower')
#                 ax.set_xticks(range(len(COEFFS)))
#                 ax.set_xticklabels(COEFFS, fontsize=8)
#                 ax.set_yticks(range(0, len(LAYERS), 2))
#                 ax.set_yticklabels([LAYERS[i] for i in range(0, len(LAYERS), 2)], fontsize=8)
#                 ax.set_xlabel('Coefficient')
#                 ax.set_ylabel('Layer')
#                 ax.set_title(title, fontsize=11)
#                 ax.axvline(zero_ci, color='white', ls='--', lw=0.8)
#                 plt.colorbar(im, ax=ax, shrink=0.7)

#             # Difference: emp - unemp
#             diff = mat_e - mat_u
#             vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.01)
#             norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
#             im = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', norm=norm,
#                                 origin='lower')
#             axes[2].set_xticks(range(len(COEFFS)))
#             axes[2].set_xticklabels(COEFFS, fontsize=8)
#             axes[2].set_yticks(range(0, len(LAYERS), 2))
#             axes[2].set_yticklabels([LAYERS[i] for i in range(0, len(LAYERS), 2)], fontsize=8)
#             axes[2].set_xlabel('Coefficient')
#             axes[2].set_title('Emp − Unemp (separation)', fontsize=11)
#             axes[2].axvline(zero_ci, color='black', ls='--', lw=0.8)
#             plt.colorbar(im, ax=axes[2], shrink=0.7)

#             fig.suptitle(f"Steering: {qkey} | cfg{cfg} | {info['label']}\n"
#                          f"Emp: {users['emp']['meta']['user'][:12]}  "
#                          f"Unemp: {users['unemp']['meta']['user'][:12]}",
#                          fontweight='bold', fontsize=13)
#             plt.tight_layout(rect=[0, 0, 1, 0.93])
#             out = os.path.join(RESULTS_DIR, f"two_user_heatmap_{qkey}_cfg{cfg}.png")
#             plt.savefig(out, dpi=200, bbox_inches='tight')
#             print(f"  Saved: {out}")
#             plt.close()


# def fig2_layer_curves(users):
#     """Per-layer curves: P vs coeff, emp and unemp on same plot."""
#     highlight = [24, 25, 26, 27, 28, 31, 33, 40]

#     for cfg in [5, 6]:
#         for qkey, info in PROMPTS.items():
#             mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
#             mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

#             fig, axes = plt.subplots(4, 2, figsize=(16, 20))
#             axes = axes.flatten()

#             for i, layer in enumerate(highlight):
#                 ax = axes[i]
#                 li = LAYERS.index(layer)
#                 ax.plot(COEFFS, mat_e[li], 'b-o', ms=5, lw=2, label='Employed')
#                 ax.plot(COEFFS, mat_u[li], 'r-s', ms=5, lw=2, label='Unemployed')
#                 ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
#                 ax.axvline(0, color='gray', ls='--', alpha=0.3)
#                 ax.set_ylim(-0.05, 1.05)
#                 ax.set_xlabel('Coefficient')
#                 ax.set_ylabel(info['label'])
#                 ax.set_title(f'Layer {layer}', fontweight='bold', fontsize=12)
#                 ax.legend(fontsize=9)
#                 ax.grid(True, alpha=0.15)

#             fig.suptitle(f"Steering Curves: {qkey} | cfg{cfg}\n"
#                          f"Blue=Employed, Red=Unemployed",
#                          fontweight='bold', fontsize=14)
#             plt.tight_layout(rect=[0, 0, 1, 0.95])
#             out = os.path.join(RESULTS_DIR, f"two_user_curves_{qkey}_cfg{cfg}.png")
#             plt.savefig(out, dpi=200, bbox_inches='tight')
#             print(f"  Saved: {out}")
#             plt.close()


# def fig3_best_layers_summary(users):
#     """Bar chart: delta at coeff=-10 and +10 per layer, both users."""
#     for cfg in [5, 6]:
#         fig, axes = plt.subplots(3, 1, figsize=(14, 15))

#         for ax, (qkey, info) in zip(axes, PROMPTS.items()):
#             mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
#             mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])
#             zero_ci = COEFFS.index(0)

#             # Delta from baseline at most extreme coefficients
#             delta_e_neg = mat_e[:, 0] - mat_e[:, zero_ci]   # coeff=-10
#             delta_e_pos = mat_e[:, -1] - mat_e[:, zero_ci]  # coeff=+10
#             delta_u_neg = mat_u[:, 0] - mat_u[:, zero_ci]
#             delta_u_pos = mat_u[:, -1] - mat_u[:, zero_ci]

#             x = np.arange(len(LAYERS))
#             w = 0.2
#             ax.bar(x - 1.5*w, delta_e_neg, w, label='Emp coeff=-10', color='#1565C0')
#             ax.bar(x - 0.5*w, delta_e_pos, w, label='Emp coeff=+10', color='#64B5F6')
#             ax.bar(x + 0.5*w, delta_u_neg, w, label='Unemp coeff=-10', color='#C62828')
#             ax.bar(x + 1.5*w, delta_u_pos, w, label='Unemp coeff=+10', color='#EF9A9A')

#             ax.set_xticks(x)
#             ax.set_xticklabels([f'L{l}' for l in LAYERS], fontsize=7, rotation=45)
#             ax.set_ylabel(f'Δ{info["label"]}')
#             ax.set_title(f'{qkey}: Change from baseline at extreme coefficients',
#                          fontweight='bold')
#             ax.legend(fontsize=7, ncol=4, loc='lower left')
#             ax.axhline(0, color='black', lw=0.5)
#             ax.grid(True, alpha=0.15, axis='y')

#         fig.suptitle(f"Steering Effectiveness per Layer | cfg{cfg}\n"
#                      f"Bars = ΔP from coeff=0 baseline",
#                      fontweight='bold', fontsize=13)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         out = os.path.join(RESULTS_DIR, f"two_user_delta_bars_cfg{cfg}.png")
#         plt.savefig(out, dpi=200, bbox_inches='tight')
#         print(f"  Saved: {out}")
#         plt.close()


# def print_summary(users):
#     """Print text summary of strongest steering effects."""
#     print(f"\n{'='*80}")
#     print("STEERING SUMMARY — 2 USERS")
#     print(f"{'='*80}")

#     for cfg in [5, 6]:
#         print(f"\n  cfg{cfg}:")
#         for qkey, info in PROMPTS.items():
#             mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
#             mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])
#             zero_ci = COEFFS.index(0)

#             print(f"\n    [{qkey}] {info['label']}")
#             print(f"    {'':>6s}  {'Employed':>30s}  {'Unemployed':>30s}")
#             print(f"    {'':>6s}  {'baseline':>10s} {'c=-10':>10s} {'c=+10':>10s}  "
#                   f"{'baseline':>10s} {'c=-10':>10s} {'c=+10':>10s}")

#             for layer in [24, 25, 26, 27, 28, 31, 33, 40]:
#                 li = LAYERS.index(layer)
#                 be = mat_e[li, zero_ci]
#                 ne = mat_e[li, 0]
#                 pe = mat_e[li, -1]
#                 bu = mat_u[li, zero_ci]
#                 nu = mat_u[li, 0]
#                 pu = mat_u[li, -1]
#                 flag_e = ' ***' if abs(ne - be) > 0.1 or abs(pe - be) > 0.1 else ''
#                 flag_u = ' ***' if abs(nu - bu) > 0.1 or abs(pu - bu) > 0.1 else ''
#                 print(f"    L{layer:>2d}:  {be:>10.4f} {ne:>10.4f} {pe:>10.4f}{flag_e}  "
#                       f"{bu:>10.4f} {nu:>10.4f} {pu:>10.4f}{flag_u}")


# def greedy_comparison(users):
#     """Print greedy text side by side."""
#     print(f"\n{'='*80}")
#     print("GREEDY TEXT COMPARISON")
#     print(f"{'='*80}")

#     for cfg in [5]:
#         for qkey in ['binary', 'behavioral', 'routine']:
#             print(f"\n  --- cfg{cfg} / {qkey} ---")
#             for layer in [25, 31, 33, 40, 46]:
#                 for coeff in [-5, 0, 5]:
#                     key = str((cfg, qkey, layer, coeff))
#                     te = users['emp']['greedy'].get(key, 'N/A')[:60]
#                     tu = users['unemp']['greedy'].get(key, 'N/A')[:60]
#                     print(f"  L{layer} c={coeff:>3d}: EMP={te:>30s}  UNEMP={tu}")


# def main():
#     users = load_users()

#     print("\n── Figure 1: Heatmaps ──")
#     fig1_side_by_side_heatmaps(users)

#     print("\n── Figure 2: Layer Curves ──")
#     fig2_layer_curves(users)

#     print("\n── Figure 3: Delta Bars ──")
#     fig3_best_layers_summary(users)

#     print_summary(users)
#     greedy_comparison(users)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steering analysis for first 2 completed users → single PDF report.
Each page: both cfg5 and cfg6 side by side for one prompt type.

Usage:
    python steering/analyze_two_users.py
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR

PERUSER_DIR = os.path.join(OUTPUT_DIR, 'steering', 'per_user')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = {
    'emp':   '5db7e19202fefe287fd8ad03bb8a64aa.json',
    'unemp': '608e0e7ae296cc575fc2226d5d882726.json',
}

PROMPTS = {
    'binary':     {'target': 'Yes', 'label': 'P(Yes = employed)'},
    'behavioral': {'target': 'B',   'label': 'P(B = workplace)'},
    'routine':    {'target': 'A',   'label': 'P(A = 9-to-5 job)'},
}

COEFFS = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
LAYERS = list(range(23, 48))
CFGS = [5, 6]
CFG_NAMES = {5: 'cfg5 (hash/orig/none)', 6: 'cfg6 (hash/orig/cot)'}
HIGHLIGHT_LAYERS = [24, 25, 26, 27, 28, 31, 33, 40]


def load_users():
    users = {}
    for tag, fname in FILES.items():
        with open(os.path.join(PERUSER_DIR, fname)) as f:
            users[tag] = json.load(f)
        print(f"  {tag}: {users[tag]['meta']['user'][:16]} "
              f"(employed={users[tag]['meta']['is_employed']})")
    return users


def get_probs(d, cfg, qkey, target):
    mat = np.full((len(LAYERS), len(COEFFS)), np.nan)
    for li, layer in enumerate(LAYERS):
        for ci, coeff in enumerate(COEFFS):
            key = str((cfg, qkey, layer, coeff))
            p = d['logits'].get(key, {})
            mat[li, ci] = p.get(target, np.nan)
    return mat


# ══════════════════════════════════════════════════════════════════
# Page Type 1: Steering Curves — 8 key layers, emp vs unemp
# ══════════════════════════════════════════════════════════════════

def page_curves(pdf, users, qkey, info):
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(f'Steering Curves: {qkey.upper()} — {info["label"]}\n'
                 f'Blue = Employed user | Red = Unemployed user | '
                 f'Gray dashed = 0.5 baseline',
                 fontsize=13, fontweight='bold')

    for col, cfg in enumerate(CFGS):
        mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
        mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

        for row, layer in enumerate(HIGHLIGHT_LAYERS[:4]):
            ax = axes[row, col]
            li = LAYERS.index(layer)

            ax.plot(COEFFS, mat_e[li], 'b-o', ms=6, lw=2.5,
                    label='Employed', zorder=5)
            ax.plot(COEFFS, mat_u[li], 'r-s', ms=6, lw=2.5,
                    label='Unemployed', zorder=5)
            ax.fill_between(COEFFS, mat_e[li], mat_u[li],
                            alpha=0.15, color='purple')
            ax.axhline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
            ax.axvline(0, color='gray', ls='--', alpha=0.5, lw=1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(info['label'], fontsize=9)
            ax.set_title(f'Layer {layer} — {CFG_NAMES[cfg]}',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.15)
            if row == 3:
                ax.set_xlabel('Steering Coefficient', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    if len(HIGHLIGHT_LAYERS) > 4:
        fig, axes = plt.subplots(4, 2, figsize=(14, 18))
        fig.suptitle(f'Steering Curves: {qkey.upper()} — {info["label"]} (cont.)\n'
                     f'Blue = Employed | Red = Unemployed',
                     fontsize=13, fontweight='bold')

        for col, cfg in enumerate(CFGS):
            mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
            mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

            for row, layer in enumerate(HIGHLIGHT_LAYERS[4:8]):
                ax = axes[row, col]
                li = LAYERS.index(layer)
                ax.plot(COEFFS, mat_e[li], 'b-o', ms=6, lw=2.5, label='Employed', zorder=5)
                ax.plot(COEFFS, mat_u[li], 'r-s', ms=6, lw=2.5, label='Unemployed', zorder=5)
                ax.fill_between(COEFFS, mat_e[li], mat_u[li], alpha=0.15, color='purple')
                ax.axhline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
                ax.axvline(0, color='gray', ls='--', alpha=0.5, lw=1)
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(info['label'], fontsize=9)
                ax.set_title(f'Layer {layer} — {CFG_NAMES[cfg]}',
                             fontsize=10, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.15)
                if row == 3:
                    ax.set_xlabel('Steering Coefficient', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════
# Page Type 2: Heatmaps — raw P, cfg5 and cfg6 on same page
# ══════════════════════════════════════════════════════════════════

def page_heatmaps(pdf, users, qkey, info):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Steering Heatmap: {qkey.upper()} — {info["label"]}\n'
                 f'Color = probability of employed-indicator token',
                 fontsize=13, fontweight='bold')

    zero_ci = COEFFS.index(0)

    for col, cfg in enumerate(CFGS):
        mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
        mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

        ax = axes[0, col]
        im = ax.imshow(mat_e, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                        origin='lower', interpolation='nearest')
        ax.set_title(f'Employed — {CFG_NAMES[cfg]}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(COEFFS)))
        ax.set_xticklabels(COEFFS, fontsize=8)
        ax.set_yticks(range(0, len(LAYERS), 2))
        ax.set_yticklabels([LAYERS[i] for i in range(0, len(LAYERS), 2)], fontsize=8)
        ax.set_ylabel('Layer')
        ax.axvline(zero_ci, color='white', ls='--', lw=1)
        plt.colorbar(im, ax=ax, shrink=0.8, label=info['label'])

        ax = axes[1, col]
        im = ax.imshow(mat_u, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                        origin='lower', interpolation='nearest')
        ax.set_title(f'Unemployed — {CFG_NAMES[cfg]}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(COEFFS)))
        ax.set_xticklabels(COEFFS, fontsize=8)
        ax.set_yticks(range(0, len(LAYERS), 2))
        ax.set_yticklabels([LAYERS[i] for i in range(0, len(LAYERS), 2)], fontsize=8)
        ax.set_ylabel('Layer')
        ax.set_xlabel('Coefficient')
        ax.axvline(zero_ci, color='white', ls='--', lw=1)
        plt.colorbar(im, ax=ax, shrink=0.8, label=info['label'])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Page Type 3: Delta bar chart — all 25 layers, both cfgs
# ══════════════════════════════════════════════════════════════════

def page_delta_bars(pdf, users, qkey, info):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'Steering Effectiveness: {qkey.upper()} — {info["label"]}\n'
                 f'ΔP from coeff=0 baseline at extreme coefficients (±10)',
                 fontsize=13, fontweight='bold')

    zero_ci = COEFFS.index(0)

    for row, cfg in enumerate(CFGS):
        ax = axes[row]
        mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
        mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

        delta_e_neg = mat_e[:, 0] - mat_e[:, zero_ci]
        delta_e_pos = mat_e[:, -1] - mat_e[:, zero_ci]
        delta_u_neg = mat_u[:, 0] - mat_u[:, zero_ci]
        delta_u_pos = mat_u[:, -1] - mat_u[:, zero_ci]

        x = np.arange(len(LAYERS))
        w = 0.2
        ax.bar(x - 1.5*w, delta_e_neg, w, label='Emp c=−10 (→unemp)', color='#1565C0')
        ax.bar(x - 0.5*w, delta_e_pos, w, label='Emp c=+10 (→emp)', color='#64B5F6')
        ax.bar(x + 0.5*w, delta_u_neg, w, label='Unemp c=−10 (→unemp)', color='#C62828')
        ax.bar(x + 1.5*w, delta_u_pos, w, label='Unemp c=+10 (→emp)', color='#EF9A9A')

        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in LAYERS], fontsize=7, rotation=45)
        ax.set_ylabel(f'Δ {info["label"]}', fontsize=10)
        ax.set_title(f'{CFG_NAMES[cfg]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, ncol=4, loc='lower left')
        ax.axhline(0, color='black', lw=0.8)
        ax.grid(True, alpha=0.15, axis='y')

        for arr, color in [(delta_e_neg, '#1565C0'), (delta_u_pos, '#EF9A9A')]:
            idx = np.argmax(np.abs(arr))
            if abs(arr[idx]) > 0.1:
                ax.annotate(f'L{LAYERS[idx]}\nΔ={arr[idx]:+.2f}',
                           xy=(idx, arr[idx]),
                           fontsize=7, fontweight='bold', color=color,
                           ha='center', va='bottom' if arr[idx] > 0 else 'top')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Page Type 4: Summary table
# ══════════════════════════════════════════════════════════════════

def page_summary_table(pdf, users):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')

    lines = []
    lines.append('STEERING SUMMARY — 2 USERS')
    lines.append(f'Employed: {users["emp"]["meta"]["user"][:16]}')
    lines.append(f'Unemployed: {users["unemp"]["meta"]["user"][:16]}')
    lines.append('')

    zero_ci = COEFFS.index(0)

    for cfg in CFGS:
        lines.append(f'{"="*90}')
        lines.append(f'{CFG_NAMES[cfg]}')
        lines.append(f'{"="*90}')

        for qkey, info in PROMPTS.items():
            mat_e = get_probs(users['emp'], cfg, qkey, info['target'])
            mat_u = get_probs(users['unemp'], cfg, qkey, info['target'])

            lines.append(f'\n  {qkey.upper()} — {info["label"]}')
            lines.append(f'  {"Layer":>6s}  {"--- Employed ---":>30s}  {"--- Unemployed ---":>30s}')
            lines.append(f'  {"":>6s}  {"base":>8s} {"c=-10":>8s} {"c=+10":>8s}  '
                         f'{"base":>8s} {"c=-10":>8s} {"c=+10":>8s}')

            for layer in HIGHLIGHT_LAYERS:
                li = LAYERS.index(layer)
                be, ne, pe = mat_e[li, zero_ci], mat_e[li, 0], mat_e[li, -1]
                bu, nu, pu = mat_u[li, zero_ci], mat_u[li, 0], mat_u[li, -1]
                fe = ' !!!' if abs(ne-be) > 0.3 or abs(pe-be) > 0.3 else \
                     ' **' if abs(ne-be) > 0.1 or abs(pe-be) > 0.1 else ''
                fu = ' !!!' if abs(nu-bu) > 0.3 or abs(pu-bu) > 0.3 else \
                     ' **' if abs(nu-bu) > 0.1 or abs(pu-bu) > 0.1 else ''
                lines.append(f'  L{layer:>2d}:   {be:>8.4f} {ne:>8.4f} {pe:>8.4f}{fe:>4s}  '
                             f'{bu:>8.4f} {nu:>8.4f} {pu:>8.4f}{fu}')

    text = '\n'.join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
            fontfamily='monospace', verticalalignment='top')

    fig.suptitle('Numerical Summary', fontsize=14, fontweight='bold')
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Page Type 5: Greedy text comparison
# ══════════════════════════════════════════════════════════════════

def page_greedy_text(pdf, users):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')

    lines = ['GREEDY TEXT COMPARISON (cfg5)', '']

    for qkey in ['binary', 'behavioral', 'routine']:
        lines.append(f'--- {qkey.upper()} ---')
        for layer in [25, 31, 33, 40, 46]:
            for coeff in [-5, 0, 5]:
                key = str((5, qkey, layer, coeff))
                te = users['emp']['greedy'].get(key, 'N/A')[:50]
                tu = users['unemp']['greedy'].get(key, 'N/A')[:50]
                lines.append(f'  L{layer} c={coeff:>3d}:  EMP={te:<30s}  UNEMP={tu}')
            lines.append('')
        lines.append('')

    text = '\n'.join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=6.5,
            fontfamily='monospace', verticalalignment='top')

    fig.suptitle('Greedy Text Comparison', fontsize=14, fontweight='bold')
    pdf.savefig(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("Loading users...")
    users = load_users()

    out_path = os.path.join(RESULTS_DIR,
                            'steering_results_for_two_specific_users.pdf')
    print(f"\nGenerating PDF: {out_path}")

    with PdfPages(out_path) as pdf:
        for qkey, info in PROMPTS.items():
            print(f"  {qkey}: curves...")
            page_curves(pdf, users, qkey, info)
            print(f"  {qkey}: heatmaps...")
            page_heatmaps(pdf, users, qkey, info)
            print(f"  {qkey}: delta bars...")
            page_delta_bars(pdf, users, qkey, info)

        print("  summary table...")
        page_summary_table(pdf, users)
        print("  greedy text...")
        page_greedy_text(pdf, users)

    print(f"\nDone! {out_path}")
    print(f"  Pages: 3 prompts × 4 pages + 2 = {3*4+2} pages total")


if __name__ == "__main__":
    main()