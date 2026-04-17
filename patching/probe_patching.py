#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: Probe-Based Patching (CPU) — Concept Sensitivity Analysis.

This is an APPROXIMATION of true activation patching. It uses saved
hidden states + trained probes to estimate how sensitive each layer's
representation is to concept swaps. No GPU or model forward pass needed.

Two directions:
  Noising:   Take employed hidden state → replace with unemployed → probe score drops?
  Denoising: Take unemployed hidden state → replace with employed → probe score rises?

Limitation: This does NOT capture cross-layer effects (replacing L20 doesn't
affect L21+ in saved data). It measures per-layer concept signal strength.
True causal patching (Step 2) requires GPU forward passes.

Output: patching/output/probe_patching_{exp}_{iter}.pdf

Usage:
    python patching/probe_patching.py                          # all defaults
    python patching/probe_patching.py --exp exp2 --iter v7     # specific
    python patching/probe_patching.py --cfgs 5 6               # specific cfgs
"""
import sys, os, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_REGISTRY, EXP_CONFIG, CONFIG_MATRIX, config_label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'patching', 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(iter_name, exp, cfg_id):
    tag = MODEL_REGISTRY['12b']['tag']
    path = os.path.join(BASE_DIR, f'outputs_{iter_name}', 'hidden_states', tag,
                        f'{exp}_cfg{cfg_id}.npz')
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True)
    return data['hidden_states'], data['labels'], data['meta']


def train_probes(X, y, meta, n_layers):
    """Train a linear probe at each layer. Returns fitted pipelines."""
    groups = np.array([m['user'] for m in meta])
    probes = []
    accuracies = []

    for layer in range(n_layers):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, solver='saga', C=1.0,
                               class_weight='balanced'))
        # Fit on all data (for patching analysis, not evaluation)
        pipe.fit(X[:, layer, :], y)

        # Also get CV accuracy for reference
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in cv.split(X[:, layer, :], y, groups):
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=500, solver='saga',
                                                 C=1.0, class_weight='balanced'))
            p.fit(X[train_idx, layer, :], y[train_idx])
            pred = p.predict(X[test_idx, layer, :])
            scores.append(balanced_accuracy_score(y[test_idx], pred))
        accuracies.append(np.mean(scores))
        probes.append(pipe)

    return probes, np.array(accuracies)


def compute_patching_effects(X, y, probes, n_layers):
    """
    For each layer, compute noising and denoising effects.

    Noising:   employed sample → swap layer with unemployed → probe score change
    Denoising: unemployed sample → swap layer with employed → probe score change
    """
    emp_idx = np.where(y == 1)[0]
    unemp_idx = np.where(y == 0)[0]
    n_pairs = min(len(emp_idx), len(unemp_idx))

    # Shuffle for random pairing
    rng = np.random.RandomState(42)
    emp_sel = rng.choice(emp_idx, n_pairs, replace=False)
    unemp_sel = rng.choice(unemp_idx, n_pairs, replace=False)

    noising_effect = np.zeros(n_layers)       # drop when employed→unemployed
    denoising_effect = np.zeros(n_layers)     # recovery when unemployed→employed
    noising_acc = np.zeros(n_layers)
    denoising_acc = np.zeros(n_layers)

    for layer in range(n_layers):
        probe = probes[layer]
        lr = probe.named_steps['logisticregression']
        scaler = probe.named_steps['standardscaler']

        # Get decision function scores (continuous, more informative than predict)
        h_emp = X[emp_sel, layer, :]
        h_unemp = X[unemp_sel, layer, :]

        # Original scores
        score_emp_original = probe.decision_function(h_emp)       # should be positive
        score_unemp_original = probe.decision_function(h_unemp)   # should be negative

        # === NOISING: employed → replace with unemployed ===
        # "If I destroy the employment signal at this layer..."
        score_emp_noised = probe.decision_function(h_unemp)  # as if employed became unemployed
        # Effect = how much the score dropped
        noising_effect[layer] = np.mean(score_emp_original) - np.mean(score_emp_noised)
        # Accuracy after noising
        noising_acc[layer] = np.mean(score_emp_noised > 0)  # should drop from ~80% to ~50%

        # === DENOISING: unemployed → replace with employed ===
        # "If I restore the employment signal at this layer..."
        score_unemp_denoised = probe.decision_function(h_emp)  # as if unemployed became employed
        # Effect = how much the score recovered
        denoising_effect[layer] = np.mean(score_unemp_denoised) - np.mean(score_unemp_original)
        # Accuracy after denoising
        denoising_acc[layer] = np.mean(score_unemp_denoised > 0)

        if (layer + 1) % 10 == 0:
            print(f"    L{layer}: noise_eff={noising_effect[layer]:.3f} "
                  f"denoise_eff={denoising_effect[layer]:.3f}")

    return {
        'noising_effect': noising_effect,
        'denoising_effect': denoising_effect,
        'noising_acc': noising_acc,
        'denoising_acc': denoising_acc,
        'n_pairs': n_pairs,
    }


def compute_concept_direction_strength(X, y, n_layers):
    """Per-layer: how far apart are employed vs unemployed along concept direction."""
    separation = np.zeros(n_layers)
    for layer in range(n_layers):
        h_emp = X[y == 1, layer, :].mean(0)
        h_unemp = X[y == 0, layer, :].mean(0)
        v = h_emp - h_unemp
        norm = np.linalg.norm(v)
        separation[layer] = norm
    return separation


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════

def pg_main(pdf, probe_acc, effects, separation, clabel, exp_cfg, title_extra):
    """Main page: probing acc + noising/denoising effect overlaid."""
    n_layers = len(probe_acc)
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1: Probing accuracy vs Patching effect (normalized)
    ax = axes[0, 0]
    ax.plot(layers, probe_acc, 'b-o', label='Probing Accuracy (CV)', lw=2, ms=3)
    # Normalize effects to [0, 1] for overlay
    ne = effects['noising_effect']
    de = effects['denoising_effect']
    ne_norm = ne / max(ne.max(), 1e-10)
    de_norm = de / max(de.max(), 1e-10)
    ax.plot(layers, ne_norm * 0.5 + 0.5, 'r--s', label='Noising Effect (normalized)',
            lw=1.5, ms=3, alpha=0.8)
    ax.plot(layers, de_norm * 0.5 + 0.5, 'g--^', label='Denoising Effect (normalized)',
            lw=1.5, ms=3, alpha=0.8)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Score')
    ax.set_title('Probing Accuracy vs Patching Effects', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
    ax.set_ylim(0.4, 1.0)

    # Panel 2: Raw noising and denoising effects
    ax = axes[0, 1]
    ax.plot(layers, ne, 'r-o', label='Noising (emp→unemp swap)', lw=2, ms=3)
    ax.plot(layers, de, 'g-s', label='Denoising (unemp→emp swap)', lw=2, ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Probe Score Change')
    ax.set_title('Raw Patching Effects (decision_function Δ)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
    # Mark peak layers
    peak_n = np.argmax(ne)
    peak_d = np.argmax(de)
    ax.axvline(peak_n, color='red', ls=':', alpha=0.5, label=f'Noise peak: L{peak_n}')
    ax.axvline(peak_d, color='green', ls=':', alpha=0.5, label=f'Denoise peak: L{peak_d}')
    ax.legend(fontsize=7)

    # Panel 3: Accuracy after patching
    ax = axes[1, 0]
    ax.plot(layers, probe_acc, 'b-o', label='Original Accuracy', lw=2, ms=3)
    ax.plot(layers, effects['noising_acc'], 'r--s',
            label='After Noising (should drop)', lw=1.5, ms=3)
    ax.plot(layers, effects['denoising_acc'], 'g--^',
            label='After Denoising (should rise)', lw=1.5, ms=3)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3, label='Chance')
    ax.set_xlabel('Layer'); ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Accuracy Before/After Patching', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

    # Panel 4: Concept direction separation
    ax = axes[1, 1]
    ax.plot(layers, separation, 'purple', lw=2)
    ax.fill_between(layers, 0, separation, alpha=0.2, color='purple')
    ax.set_xlabel('Layer'); ax.set_ylabel('||mean(emp) - mean(unemp)||')
    ax.set_title('Concept Direction Magnitude', fontweight='bold')
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'{exp_cfg["name"]} — Probe-Based Patching (CPU Approximation)\n'
                 f'{title_extra} | {effects["n_pairs"]} matched pairs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150); plt.close()


def pg_derivative(pdf, probe_acc, effects, clabel, exp_cfg, title_extra):
    """Derivative page: where is concept being ACTIVELY processed?"""
    n_layers = len(probe_acc)
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Derivative of probing accuracy (where is info being WRITTEN?)
    ax = axes[0]
    deriv = np.diff(probe_acc, prepend=probe_acc[0])
    ax.bar(layers, deriv, color=['#4CAF50' if d > 0 else '#FF5722' for d in deriv],
           alpha=0.7)
    ax.set_xlabel('Layer'); ax.set_ylabel('Δ Probing Accuracy')
    ax.set_title('Probing Accuracy Derivative\n'
                 '(Green = info being WRITTEN at this layer)', fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.grid(True, alpha=0.15)

    # Noising effect derivative
    ax = axes[1]
    ne = effects['noising_effect']
    ne_deriv = np.diff(ne, prepend=ne[0])
    ax.bar(layers, ne_deriv,
           color=['#2196F3' if d > 0 else '#FF9800' for d in ne_deriv], alpha=0.7)
    ax.set_xlabel('Layer'); ax.set_ylabel('Δ Noising Effect')
    ax.set_title('Noising Effect Derivative\n'
                 '(Blue = concept signal INCREASING)', fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'{exp_cfg["name"]} — Where Is Employment Being Processed?\n'
                 f'{title_extra}', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=150); plt.close()


def pg_summary(pdf, probe_acc, effects, separation, clabel, exp_cfg, title_extra):
    """Summary table."""
    fig, ax = plt.subplots(figsize=(16, 10)); ax.axis('off')
    n_layers = len(probe_acc)
    ne = effects['noising_effect']
    de = effects['denoising_effect']

    lines = [f'{exp_cfg["name"]} — Probe-Based Patching Summary\n'
             f'{title_extra}\n\n']
    lines.append(f'{"Layer":>6s} {"ProbeAcc":>10s} {"NoiseEff":>10s} '
                 f'{"DenoiseEff":>10s} {"NoiseAcc":>10s} {"DenoiseAcc":>10s} '
                 f'{"||V||":>10s}\n')
    lines.append('─' * 80 + '\n')

    for l in range(n_layers):
        marker = ''
        if l == np.argmax(ne): marker += ' ← noise peak'
        if l == np.argmax(probe_acc): marker += ' ← probe peak'
        lines.append(f'L{l:>4d} {probe_acc[l]:>10.3f} {ne[l]:>10.3f} '
                     f'{de[l]:>10.3f} {effects["noising_acc"][l]:>10.3f} '
                     f'{effects["denoising_acc"][l]:>10.3f} '
                     f'{separation[l]:>10.1f}{marker}\n')

    lines.append('─' * 80 + '\n')
    lines.append(f'\nProbe peak: L{np.argmax(probe_acc)} ({probe_acc.max():.1%})\n')
    lines.append(f'Noise effect peak: L{np.argmax(ne)} ({ne.max():.3f})\n')
    lines.append(f'Denoise effect peak: L{np.argmax(de)} ({de.max():.3f})\n')
    lines.append(f'||V|| peak: L{np.argmax(separation)} ({separation.max():.1f})\n')

    ax.text(0.01, 0.99, ''.join(lines), transform=ax.transAxes,
            fontsize=6.5, va='top', fontfamily='monospace')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def run_one(pdf, iter_name, exp, cfg_id):
    exp_cfg = EXP_CONFIG[exp]
    clabel = config_label(exp, cfg_id)
    title_extra = f'{iter_name} / cfg{cfg_id} ({clabel})'

    X, y, meta = load_data(iter_name, exp, cfg_id)
    if X is None:
        print(f"  SKIP: {iter_name}/{exp}/cfg{cfg_id} not found")
        return

    n_layers = X.shape[1]
    print(f"\n  {title_extra}: N={X.shape[0]}, layers={n_layers}")

    print(f"    Training probes...")
    probes, probe_acc = train_probes(X, y, meta, n_layers)

    print(f"    Computing patching effects...")
    effects = compute_patching_effects(X, y, probes, n_layers)

    print(f"    Computing concept direction...")
    separation = compute_concept_direction_strength(X, y, n_layers)

    print(f"    Generating pages...")
    pg_main(pdf, probe_acc, effects, separation, clabel, exp_cfg, title_extra)
    pg_derivative(pdf, probe_acc, effects, clabel, exp_cfg, title_extra)
    pg_summary(pdf, probe_acc, effects, separation, clabel, exp_cfg, title_extra)
    print(f"    Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['exp2'])
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'])
    parser.add_argument('--cfgs', nargs='+', type=int, default=None)
    args = parser.parse_args()

    default_cfgs = {'exp2': [5, 6], 'exp1a': [3, 4]}

    pdf_path = os.path.join(OUT_DIR, 'probe_patching.pdf')
    print(f"{'='*60}")
    print(f"Probe-Based Patching (CPU Approximation)")
    print(f"Output: {pdf_path}")
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
