#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steering analysis and visualization.
Figure 1: Effect curves — P(employed) vs alpha, per layer
Figure 2: Layer heatmap — delta_P across layers × coeffs
Figure 3: Text comparison table — greedy outputs at key conditions

Usage:
    python steering/analyze_steering.py
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR
from steering.prompts import STEERING_PROMPTS

RESULTS_DIR = os.path.join(OUTPUT_DIR, 'steering', 'results')
LAYER_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 30))


def load_combined():
    path = os.path.join(RESULTS_DIR, 'steering_combined.json')
    with open(path) as f:
        return json.load(f)


def fig1_effect_curves(data):
    """P(employed_target) vs coeff, one subplot per (prompt, cfg), lines per layer."""
    cfgs = data['cfgs']
    coeffs = data['coeffs']
    layers = data['layers']
    prompts = [p for p in data['prompts'] if STEERING_PROMPTS[p]['targets']]

    highlight_layers = [25, 31, 33, 40, 47]

    for qkey in prompts:
        for cfg_id in cfgs:
            fig, (ax_e, ax_u) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

            for ax, tag, title in [(ax_e, 'emp', 'Employed Users'),
                                   (ax_u, 'unemp', 'Unemployed Users')]:
                key = f"mean_{qkey}_cfg{cfg_id}_{tag}"
                if key not in data:
                    continue
                mean = np.array(data[key])  # (n_layers, n_coeffs)

                for li, layer in enumerate(layers):
                    alpha = 0.8 if layer in highlight_layers else 0.1
                    lw = 2.5 if layer in highlight_layers else 0.5
                    label = f"L{layer}" if layer in highlight_layers else None
                    color_idx = li % len(LAYER_COLORS)
                    ax.plot(coeffs, mean[li], color=LAYER_COLORS[color_idx],
                            alpha=alpha, lw=lw, label=label)

                ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
                ax.axvline(0, color='gray', ls='--', alpha=0.3)
                ax.set_xlabel('Steering Coefficient', fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.grid(True, alpha=0.1)
                if label:
                    ax.legend(fontsize=8, loc='lower right')

            ax_e.set_ylabel(f"P({STEERING_PROMPTS[qkey]['employed_target']})", fontsize=11)

            fig.suptitle(f"Steering Effect: {qkey} | cfg{cfg_id}\n"
                         f"P({STEERING_PROMPTS[qkey]['employed_target']}) vs Coefficient",
                         fontweight='bold', fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            out = os.path.join(RESULTS_DIR, f"steering_curve_{qkey}_cfg{cfg_id}.png")
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"  Saved: {out}")
            plt.close()


def fig2_heatmap(data):
    """Delta_P heatmap: layers × coeffs. Separate for emp/unemp."""
    cfgs = data['cfgs']
    coeffs = data['coeffs']
    layers = data['layers']
    prompts = [p for p in data['prompts'] if STEERING_PROMPTS[p]['targets']]

    zero_idx = coeffs.index(0)

    for qkey in prompts:
        for cfg_id in cfgs:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            for ax, tag, title in [(axes[0], 'emp', 'Employed'),
                                   (axes[1], 'unemp', 'Unemployed')]:
                key = f"mean_{qkey}_cfg{cfg_id}_{tag}"
                if key not in data:
                    continue
                mean = np.array(data[key])  # (n_layers, n_coeffs)
                baseline = mean[:, zero_idx:zero_idx+1]
                delta = mean - baseline

                vmax = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 0.01)
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                im = ax.imshow(delta, aspect='auto', cmap='RdBu_r', norm=norm,
                               origin='lower')
                ax.set_xticks(range(len(coeffs)))
                ax.set_xticklabels(coeffs, fontsize=8)
                ax.set_yticks(range(0, len(layers), 2))
                ax.set_yticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
                ax.set_xlabel('Coefficient')
                ax.set_ylabel('Layer')
                ax.set_title(f'{title} (ΔP from baseline)')
                plt.colorbar(im, ax=ax, shrink=0.8)

            fig.suptitle(f"Steering ΔP Heatmap: {qkey} | cfg{cfg_id}\n"
                         f"Change in P({STEERING_PROMPTS[qkey]['employed_target']}) "
                         f"from coeff=0 baseline",
                         fontweight='bold', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            out = os.path.join(RESULTS_DIR, f"steering_heatmap_{qkey}_cfg{cfg_id}.png")
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"  Saved: {out}")
            plt.close()


def fig3_text_table(data):
    """Save greedy text comparisons as readable text file."""
    texts = data.get('greedy_texts', {})
    if not texts:
        print("  No greedy texts found")
        return

    out = os.path.join(RESULTS_DIR, 'steering_text_comparison.txt')
    with open(out, 'w') as f:
        f.write("STEERING TEXT COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        # Group by condition
        conditions = {}
        for key_str, text in texts.items():
            parts = key_str.split('_', 2)
            uid_tag = f"{parts[0]}_{parts[1]}"
            cond = parts[2]
            if cond not in conditions:
                conditions[cond] = {}
            conditions[cond][uid_tag] = text

        for cond in sorted(conditions.keys()):
            f.write(f"\n{'─'*80}\n")
            f.write(f"Condition: {cond}\n")
            f.write(f"{'─'*80}\n")
            for uid_tag in sorted(conditions[cond].keys()):
                text = conditions[cond][uid_tag][:300]
                f.write(f"  [{uid_tag}]: {text}\n")

    print(f"  Saved: {out}")


def summary_stats(data):
    """Print summary statistics."""
    cfgs = data['cfgs']
    coeffs = data['coeffs']
    prompts = [p for p in data['prompts'] if STEERING_PROMPTS[p]['targets']]
    zero_idx = coeffs.index(0)

    print(f"\n{'='*60}")
    print("STEERING SUMMARY")
    print(f"{'='*60}")

    for qkey in prompts:
        target = STEERING_PROMPTS[qkey]['employed_target']
        print(f"\n  [{qkey}] P({target})")

        for cfg_id in cfgs:
            print(f"    cfg{cfg_id}:")
            for tag in ['emp', 'unemp']:
                key = f"mean_{qkey}_cfg{cfg_id}_{tag}"
                if key not in data:
                    continue
                mean = np.array(data[key])
                baseline = mean[:, zero_idx]  # (n_layers,)

                # Best steering layer: largest delta at strongest positive coeff
                pos_idx = -1  # last coeff (most positive)
                neg_idx = 0   # first coeff (most negative)
                delta_pos = mean[:, pos_idx] - baseline
                delta_neg = mean[:, neg_idx] - baseline
                best_pos_layer = data['layers'][np.argmax(np.abs(delta_pos))]
                best_neg_layer = data['layers'][np.argmax(np.abs(delta_neg))]

                print(f"      {tag:6s}: baseline={baseline.mean():.3f} | "
                      f"+coeff: best_layer=L{best_pos_layer} "
                      f"(Δ={delta_pos[np.argmax(np.abs(delta_pos))]:.3f}) | "
                      f"-coeff: best_layer=L{best_neg_layer} "
                      f"(Δ={delta_neg[np.argmax(np.abs(delta_neg))]:.3f})")


def main():
    data = load_combined()
    print(f"Loaded: {data['n_emp']} emp + {data['n_unemp']} unemp users")

    print("\n── Figure 1: Effect Curves ──")
    fig1_effect_curves(data)

    print("\n── Figure 2: Layer Heatmaps ──")
    fig2_heatmap(data)

    print("\n── Figure 3: Text Comparison ──")
    fig3_text_table(data)

    summary_stats(data)


if __name__ == "__main__":
    main()
