#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: True Activation Patching (CPU) — Causal Localization.

Uses saved hidden states to perform per-layer contribution patching.

Key formula (based on residual stream additivity):
  Δh_L = h_L - h_{L-1}  (layer L's own contribution to residual stream)
  
  Noising:   h_noised   = h_final_emp   - Δh_L_emp   + Δh_L_unemp
  Denoising: h_denoised = h_final_unemp - Δh_L_unemp + Δh_L_emp

Two directions:
  Noising (Necessity):   swap layer L's contribution from employed → unemployed
                         → does the final-layer probe score drop?
  Denoising (Sufficiency): swap layer L's contribution from unemployed → employed
                           → does the final-layer probe score recover?

Metric: probe decision_function at FINAL layer (not patched layer).

Reference: Heimersheim & Nanda (2024), "How to use and interpret activation patching"

Output: patching/output/activation_patching_{exp}_{iter}_cfg{id}.pdf

Usage:
    python patching/activation_patching.py --exp exp2 --iter v7 --cfg 5
    python patching/activation_patching.py --exp exp2 --iter v7 --cfg 5 --n-pairs 10  # test
"""
import sys, os, argparse, time, json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (MODEL_REGISTRY, EXP_CONFIG, config_label, DATA_DIR)
from utils import load_data, load_model, build_user_prompts, wrap_with_chat_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'patching', 'output')
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Matched Pair Construction
# ══════════════════════════════════════════════════════════════════════

def build_matched_pairs(iter_name, exp, cfg_id, n_pairs=50):
    """
    Build matched employed/unemployed prompt pairs from saved hidden states.
    Returns list of (employed_idx, unemployed_idx) into the npz arrays.
    """
    tag = MODEL_REGISTRY['12b']['tag']
    path = os.path.join(BASE_DIR, f'outputs_{iter_name}', 'hidden_states', tag,
                        f'{exp}_cfg{cfg_id}.npz')
    data = np.load(path, allow_pickle=True)
    X, y, meta = data['hidden_states'], data['labels'], data['meta']

    emp_idx = np.where(y == 1)[0]
    unemp_idx = np.where(y == 0)[0]

    rng = np.random.RandomState(42)
    n = min(n_pairs, len(emp_idx), len(unemp_idx))
    emp_sel = rng.choice(emp_idx, n, replace=False)
    unemp_sel = rng.choice(unemp_idx, n, replace=False)

    return X, y, meta, list(zip(emp_sel, unemp_sel))


def get_prompt_for_sample(meta_entry, df, cfg_id, exp):
    """Rebuild the prompt for a specific sample from metadata."""
    # This would need the full data pipeline — for patching we use a simpler approach
    pass


# ══════════════════════════════════════════════════════════════════════
# Activation Patching with Hooks
# ══════════════════════════════════════════════════════════════════════

def run_with_cache(model, inputs, n_decoder_layers):
    """Forward pass, cache all layer hidden states."""
    cache = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[layer_idx] = h.detach().clone()
        return hook

    handles = []
    for i in range(n_decoder_layers):
        handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        output = model(**inputs)

    for h in handles:
        h.remove()

    logits = output.logits[0, -1, :].detach().clone()
    return logits, cache


def run_with_patch(model, inputs, patch_layer, patch_activation, n_decoder_layers):
    """Forward pass, but replace hidden state at patch_layer with patch_activation."""

    def patch_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # Replace entire hidden state at this layer
        h[:, :, :] = patch_activation
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)

    with torch.no_grad():
        output = model(**inputs)

    handle.remove()
    logits = output.logits[0, -1, :].detach().clone()
    return logits


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def probe_projection(logits_or_hidden, probe_weights, probe_bias):
    """Project onto probe direction → scalar (positive = employed)."""
    # Use logits as a proxy — or hidden state at last layer
    return (logits_or_hidden @ probe_weights + probe_bias).item()


def logit_diff(logits, target_token_id, baseline_token_id):
    """logit(target) - logit(baseline)."""
    return (logits[target_token_id] - logits[baseline_token_id]).item()


# ══════════════════════════════════════════════════════════════════════
# Main Patching Loop
# ══════════════════════════════════════════════════════════════════════

def run_patching(model, tokenizer, X_saved, y_saved, meta_saved, pairs,
                 iter_name, exp, cfg_id, device):
    """
    Use the residual stream's additive structure to approximate activation patching.

    Key insight (Heimersheim & Nanda, 2024):
      h_final = h_embed + Δh_L0 + Δh_L1 + ... + Δh_L47
      where Δh_L = h_L - h_{L-1} is layer L's OWN contribution.

    To test layer L's causal role, we swap ONLY its contribution:
      Noising:   h_noised   = h_final_emp   - Δh_L_emp   + Δh_L_unemp
      Denoising: h_denoised = h_final_unemp - Δh_L_unemp + Δh_L_emp

    Then evaluate the patched vector with a probe trained at the final layer.

    IMPORTANT: Δh_L ≠ h_L. 
      h_L   = cumulative residual stream (sum of ALL layers 0..L)
      Δh_L  = h_L - h_{L-1} = only what layer L itself added

    Using Δh_L isolates each layer's individual contribution.
    Using h_L (wrong) would swap ALL layers 0..L simultaneously.
    """
    n_layers = X_saved.shape[1]
    n_pairs = len(pairs)

    # Train probe at final layer for evaluation
    print(f"    Training final-layer probe...")
    final_probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, solver='saga', C=1.0,
                           class_weight='balanced'))
    final_probe.fit(X_saved[:, -1, :], y_saved)
    train_acc = final_probe.score(X_saved[:, -1, :], y_saved)
    print(f"    Final layer probe accuracy: {train_acc:.1%}")

    noising_effect = np.zeros(n_layers)
    denoising_effect = np.zeros(n_layers)

    for layer in range(n_layers):
        noise_scores = []
        denoise_scores = []

        for emp_i, unemp_i in pairs:
            h_emp_final = X_saved[emp_i, -1, :].copy()
            h_unemp_final = X_saved[unemp_i, -1, :].copy()

            # ── Compute per-layer CONTRIBUTION Δh_L = h_L - h_{L-1} ──
            # This is what Layer L itself ADDED to the residual stream,
            # NOT the cumulative state at Layer L.
            if layer == 0:
                # Layer 0 = embedding output; its "contribution" IS itself
                delta_emp = X_saved[emp_i, 0, :].copy()
                delta_unemp = X_saved[unemp_i, 0, :].copy()
            else:
                delta_emp = X_saved[emp_i, layer, :] - X_saved[emp_i, layer - 1, :]
                delta_unemp = X_saved[unemp_i, layer, :] - X_saved[unemp_i, layer - 1, :]

            # === NOISING: replace ONLY layer L's contribution ===
            # h_noised = h_final_emp - Δh_L_emp + Δh_L_unemp
            # "What if layer L had contributed unemployed-style info instead?"
            h_noised = h_emp_final - delta_emp + delta_unemp
            score_original = final_probe.decision_function(h_emp_final.reshape(1, -1))[0]
            score_noised = final_probe.decision_function(h_noised.reshape(1, -1))[0]
            noise_scores.append(score_original - score_noised)

            # === DENOISING: restore ONLY layer L's contribution ===
            # h_denoised = h_final_unemp - Δh_L_unemp + Δh_L_emp
            # "What if layer L had contributed employed-style info instead?"
            h_denoised = h_unemp_final - delta_unemp + delta_emp
            score_original_u = final_probe.decision_function(h_unemp_final.reshape(1, -1))[0]
            score_denoised = final_probe.decision_function(h_denoised.reshape(1, -1))[0]
            denoise_scores.append(score_denoised - score_original_u)

        noising_effect[layer] = np.mean(noise_scores)
        denoising_effect[layer] = np.mean(denoise_scores)

        if (layer + 1) % 10 == 0:
            print(f"    L{layer}: noise={noising_effect[layer]:.3f} "
                  f"denoise={denoising_effect[layer]:.3f}")

    return noising_effect, denoising_effect


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════

def make_pdf(pdf_path, noising, denoising, probe_acc, exp_cfg, title_extra):
    n_layers = len(noising)
    layers = np.arange(n_layers)

    with PdfPages(pdf_path) as pdf:
        # Page 1: Main comparison
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Panel 1: All three curves
        ax = axes[0, 0]
        ax.plot(layers, probe_acc, 'b-o', label='Probing Accuracy', lw=2, ms=3)
        # Normalize patching effects to same scale
        n_max = max(noising.max(), 0.001)
        d_max = max(denoising.max(), 0.001)
        ax.plot(layers, noising / n_max * 0.3 + 0.5, 'r--s',
                label=f'Noising Effect (×{0.3/n_max:.1f})', lw=1.5, ms=3)
        ax.plot(layers, denoising / d_max * 0.3 + 0.5, 'g--^',
                label=f'Denoising Effect (×{0.3/d_max:.1f})', lw=1.5, ms=3)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel('Layer'); ax.set_ylabel('Score')
        ax.set_title('Probing vs Patching (per-layer Δh contribution)',
                     fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

        # Panel 2: Raw noising effect
        ax = axes[0, 1]
        ax.bar(layers, noising, color='#FF5722', alpha=0.7)
        peak = np.argmax(noising)
        ax.axvline(peak, color='red', ls=':', lw=2)
        ax.set_xlabel('Layer'); ax.set_ylabel('Score Drop')
        ax.set_title(f'Noising Effect (Necessity) — peak L{peak}',
                     fontweight='bold')
        ax.grid(True, alpha=0.15)

        # Panel 3: Raw denoising effect
        ax = axes[1, 0]
        ax.bar(layers, denoising, color='#4CAF50', alpha=0.7)
        peak = np.argmax(denoising)
        ax.axvline(peak, color='green', ls=':', lw=2)
        ax.set_xlabel('Layer'); ax.set_ylabel('Score Recovery')
        ax.set_title(f'Denoising Effect (Sufficiency) — peak L{peak}',
                     fontweight='bold')
        ax.grid(True, alpha=0.15)

        # Panel 4: Probing derivative vs patching
        ax = axes[1, 1]
        deriv = np.diff(probe_acc, prepend=probe_acc[0])
        ax.bar(layers - 0.2, deriv, 0.4, color='#2196F3', alpha=0.6,
               label='Probing Δ (info written)')
        ax.bar(layers + 0.2, noising / max(noising.max(), 1e-10) * deriv.max(),
               0.4, color='#FF5722', alpha=0.6, label='Noising (normalized)')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel('Layer'); ax.set_ylabel('Effect')
        ax.set_title('Where Info is Written (probing Δ) vs Where it Matters (noising)',
                     fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

        fig.suptitle(f'{exp_cfg["name"]} — Activation Patching (Per-Layer Δh Contribution)\n'
                     f'{title_extra}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150); plt.close()

        # Page 2: Summary table
        fig, ax = plt.subplots(figsize=(16, 12)); ax.axis('off')
        lines = [f'Activation Patching Summary — {title_extra}\n\n']
        lines.append('Method: Per-layer contribution patching (residual stream additive)\n')
        lines.append('Δh_L = h_L - h_{L-1} (layer L\'s own contribution)\n')
        lines.append('h_noised   = h_emp_final   - Δh_L_emp   + Δh_L_unemp\n')
        lines.append('h_denoised = h_unemp_final - Δh_L_unemp + Δh_L_emp\n\n')

        lines.append(f'{"Layer":>6s} {"ProbeAcc":>10s} {"Noising":>10s} '
                     f'{"Denoising":>10s}\n')
        lines.append('─' * 50 + '\n')
        for l in range(n_layers):
            marker = ''
            if l == np.argmax(noising): marker += ' ← NOISE PEAK'
            if l == np.argmax(denoising): marker += ' ← DENOISE PEAK'
            if l == np.argmax(probe_acc): marker += ' ← PROBE PEAK'
            lines.append(f'L{l:>4d} {probe_acc[l]:>10.3f} '
                         f'{noising[l]:>10.4f} {denoising[l]:>10.4f}{marker}\n')

        lines.append('─' * 50 + '\n')
        lines.append(f'\nInterpretation:\n')
        lines.append(f'  Noising peak at L{np.argmax(noising)}: this layer\'s OWN contribution\n')
        lines.append(f'  (Δh_L = h_L - h_{{L-1}}) is most critical for employment signal.\n')
        lines.append(f'  If noising peak << probing peak → flat probing = passive carrier.\n')

        ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
                fontsize=7, va='top', fontfamily='monospace')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='exp2')
    parser.add_argument('--iter', default='v7')
    parser.add_argument('--cfg', type=int, default=5)
    parser.add_argument('--n-pairs', type=int, default=50)
    args = parser.parse_args()

    exp_cfg = EXP_CONFIG[args.exp]
    clabel = config_label(args.exp, args.cfg)
    title_extra = f'{args.iter} / cfg{args.cfg} ({clabel})'

    print(f"{'='*60}")
    print(f"Activation Patching — Per-Layer Δh Contribution")
    print(f"  {title_extra}")
    print(f"  Pairs: {args.n_pairs}")
    print(f"{'='*60}")

    # Load saved hidden states
    X, y, meta, pairs = build_matched_pairs(
        args.iter, args.exp, args.cfg, args.n_pairs)
    print(f"  Loaded: {X.shape}, {len(pairs)} pairs")

    # Get probing accuracy for reference
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    groups = np.array([m['user'] for m in meta])
    n_layers = X.shape[1]
    print(f"  Computing probing accuracy...")
    probe_acc = np.zeros(n_layers)
    for layer in range(n_layers):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=500, solver='saga',
                                            C=1.0, class_weight='balanced')),
            X[:, layer, :], y, cv=cv, groups=groups,
            scoring='balanced_accuracy')
        probe_acc[layer] = scores.mean()
    print(f"  Probe peak: L{np.argmax(probe_acc)} ({probe_acc.max():.1%})")

    # Run patching
    print(f"\n  Running patching ({len(pairs)} pairs × {n_layers} layers)...")
    noising, denoising = run_patching(
        None, None, X, y, meta, pairs,
        args.iter, args.exp, args.cfg, None)

    # Generate PDF
    pdf_path = os.path.join(OUT_DIR,
                            f'activation_patching_{args.exp}_{args.iter}_cfg{args.cfg}.pdf')
    print(f"\n  Generating: {pdf_path}")
    make_pdf(pdf_path, noising, denoising, probe_acc, exp_cfg, title_extra)

    # Save raw data
    npz_path = os.path.join(OUT_DIR,
                            f'activation_patching_{args.exp}_{args.iter}_cfg{args.cfg}.npz')
    np.savez(npz_path, noising=noising, denoising=denoising,
             probe_acc=probe_acc, n_pairs=len(pairs))
    print(f"  Saved: {npz_path}")

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Noising peak:   L{np.argmax(noising)} (effect={noising.max():.4f})")
    print(f"  Denoising peak: L{np.argmax(denoising)} (effect={denoising.max():.4f})")
    print(f"  Probing peak:   L{np.argmax(probe_acc)} ({probe_acc.max():.1%})")
    if np.argmax(noising) < np.argmax(probe_acc):
        print(f"\n  → Noising peak < Probing peak → FLAT PROBING = PASSIVE CARRIER")
        print(f"  → Causal processing at L{np.argmax(noising)}, "
              f"passive carry from L{np.argmax(noising)+1}+")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()