#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAE Feature Analysis: Check if GemmaScope 2 SAE features (e.g., "employment")
activate on mobility trajectory hidden states.

Tests cross-modal concept reuse: do GPS trajectory activations light up
the same SAE features as natural language about employment?

Usage:
    python sae_feature_check.py
    python sae_feature_check.py --exp exp2 --cfgs 5 6
    python sae_feature_check.py --top-k 100        # also show top differing features
    python sae_feature_check.py --layers 12 24 31 41  # check multiple SAE layers

Requires: pip install sae-lens --break-system-packages
"""
import sys, os, argparse, json
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths (adjust to your cluster) ──────────────────────────────────
BASE_DIR = "/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7"
HIDDEN_DIR = os.path.join(BASE_DIR, "outputs_v7/hidden_states/gemma3_12b_it")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs_v7/results/gemma3_12b_it")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Layer mapping ───────────────────────────────────────────────────
# Your .npz: index 0 = embedding (L0), index i = model.layers[i-1]
# GemmaScope: layer_N = model.layers[N] = your array index N+1
#
# Subset SAE layers (wide hyperparameter sweep): 12, 24, 31, 41
# All-layer SAEs also available via resid_post_all release
GEMMASCOPE_TO_NPZ = lambda gs_layer: gs_layer + 1  # GemmaScope layer_N → npz index N+1

# ── Known features of interest ──────────────────────────────────────
# Format: (gemmascope_layer, width_str, feature_index, label)
# Add more as you discover them on Neuronpedia!
FEATURES_OF_INTEREST = [
    (41, "65k", 22394, "employment"),
    # Examples — uncomment / add as you find them:
    # (41, "65k", XXXXX, "working_day"),
    # (41, "65k", XXXXX, "weekend"),
    # (41, "65k", XXXXX, "commute"),
    # (24, "65k", XXXXX, "routine"),
    # (31, "65k", XXXXX, "schedule"),
]

# ── Experiment / config definitions ─────────────────────────────────
EXP_CONFIGS = {
    "exp1a": {"cfgs": [1, 2, 3, 4], "label_names": {0: "weekend", 1: "weekday"}},
    "exp2":  {"cfgs": [1, 2, 3, 4, 5, 6, 7, 8], "label_names": {0: "unemployed", 1: "employed"}},
}


def load_sae(gs_layer, width="65k", device="cuda"):
    """Load GemmaScope 2 SAE for a given layer."""
    from sae_lens import SAE

    release = "google/gemma-scope-2-12b-it"
    sae_id = f"resid_post/layer_{gs_layer}_width_{width}_l0_medium"


    print(f"  Loading SAE: {release} / {sae_id} ...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device,
    )
    sae = sae.to(torch.bfloat16)  # match model dtype, negligible quality loss
    sae.eval()
    print(f"  SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, dtype={next(sae.parameters()).dtype}")
    return sae


def analyze_config(sae, gs_layer, feature_idx, feature_label,
                   exp_name, cfg_id, label_names, top_k=0):
    """Run SAE on one config's hidden states, report feature activation."""
    tag = f"{exp_name}_cfg{cfg_id}"
    npz_path = os.path.join(HIDDEN_DIR, f"{tag}.npz")
    if not os.path.exists(npz_path):
        return None

    # Load hidden states
    data = np.load(npz_path, allow_pickle=True)
    hidden = data["hidden_states"]  # (n_samples, 49, 3840), float32
    labels = data["labels"]         # (n_samples,), int32

    # Extract the layer matching the SAE
    npz_idx = GEMMASCOPE_TO_NPZ(gs_layer)
    layer_acts = torch.tensor(hidden[:, npz_idx, :], dtype=torch.bfloat16, device="cuda")

    # Encode through SAE
    with torch.no_grad():
        feat_acts = sae.encode(layer_acts)  # (n_samples, d_sae)

    # Target feature
    target_act = feat_acts[:, feature_idx].cpu().numpy()
    labels_np = labels

    result = {"tag": tag, "n_samples": len(labels_np)}

    # Per-class stats
    for cls_val, cls_name in label_names.items():
        mask = labels_np == cls_val
        acts = target_act[mask]
        n = mask.sum()
        n_active = (acts > 0).sum()
        result[cls_name] = {
            "n": int(n),
            "n_active": int(n_active),
            "pct_active": float(n_active / n * 100) if n > 0 else 0.0,
            "mean_act": float(acts.mean()) if n > 0 else 0.0,
            "max_act": float(acts.max()) if n > 0 else 0.0,
            "mean_when_active": float(acts[acts > 0].mean()) if n_active > 0 else 0.0,
        }

    # Overall
    n_active_total = (target_act > 0).sum()
    result["overall"] = {
        "n_active": int(n_active_total),
        "pct_active": float(n_active_total / len(target_act) * 100),
        "mean_act": float(target_act.mean()),
    }

    # Top-K differentially activated features (discovery mode)
    if top_k > 0 and len(label_names) == 2:
        cls_vals = sorted(label_names.keys())
        mask0 = labels_np == cls_vals[0]
        mask1 = labels_np == cls_vals[1]
        mean0 = feat_acts[mask0].mean(dim=0).cpu().numpy()
        mean1 = feat_acts[mask1].mean(dim=0).cpu().numpy()
        diff = mean1 - mean0  # positive = more active for class 1

        top_class1 = np.argsort(diff)[-top_k:][::-1]
        top_class0 = np.argsort(diff)[:top_k]

        result["top_features_class1"] = [
            {"idx": int(i), "diff": float(diff[i]),
             "mean_c0": float(mean0[i]), "mean_c1": float(mean1[i])}
            for i in top_class1 if diff[i] > 0
        ][:top_k]
        result["top_features_class0"] = [
            {"idx": int(i), "diff": float(diff[i]),
             "mean_c0": float(mean0[i]), "mean_c1": float(mean1[i])}
            for i in top_class0 if diff[i] < 0
        ][:top_k]

    return result


def print_result(result, feature_label, label_names):
    """Pretty-print one config's results."""
    tag = result["tag"]
    cls_names = list(label_names.values())

    print(f"\n  {tag} ({result['n_samples']} samples)")
    print(f"  {'':>16s}  {'N':>6s}  {'Active':>7s}  {'%Act':>7s}  "
          f"{'MeanAct':>9s}  {'MaxAct':>9s}  {'MeanIfAct':>10s}")
    print(f"  {'-'*70}")

    for cls_name in cls_names:
        s = result[cls_name]
        print(f"  {cls_name:>16s}  {s['n']:6d}  {s['n_active']:7d}  "
              f"{s['pct_active']:6.2f}%  {s['mean_act']:9.4f}  "
              f"{s['max_act']:9.4f}  {s['mean_when_active']:10.4f}")

    o = result["overall"]
    print(f"  {'OVERALL':>16s}  {result['n_samples']:6d}  {o['n_active']:7d}  "
          f"{o['pct_active']:6.2f}%  {o['mean_act']:9.4f}")

    # Top differential features if available
    for key, direction in [("top_features_class1", cls_names[1]),
                           ("top_features_class0", cls_names[0])]:
        if key in result and result[key]:
            print(f"\n  Top features more active for [{direction}]:")
            for i, f in enumerate(result[key][:10]):
                print(f"    #{f['idx']:6d}  diff={f['diff']:+.4f}  "
                      f"({cls_names[0]}={f['mean_c0']:.4f}, {cls_names[1]}={f['mean_c1']:.4f})")
                # Look up on Neuronpedia:
                # https://www.neuronpedia.org/gemma-3-12b-it/{layer}-gemmascope-2-res-{width}/{idx}


def main():
    parser = argparse.ArgumentParser(description="SAE Feature Analysis on Mobility Hidden States")
    parser.add_argument("--exp", nargs="+", default=["exp2"],
                        help="Experiments to analyze (default: exp2)")
    parser.add_argument("--cfgs", nargs="+", type=int, default=None,
                        help="Config IDs (default: all)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Also report top-K differentially activated features")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("SAE Feature Analysis: Cross-Modal Concept Detection")
    print("=" * 70)

    # Group features by (layer, width) to minimize SAE loads
    sae_groups = defaultdict(list)
    for gs_layer, width, feat_idx, label in FEATURES_OF_INTEREST:
        sae_groups[(gs_layer, width)].append((feat_idx, label))

    all_results = {}

    for (gs_layer, width), features in sae_groups.items():
        print(f"\n{'─'*70}")
        print(f"SAE: GemmaScope layer_{gs_layer}, width={width}")
        print(f"  → Your .npz index: {GEMMASCOPE_TO_NPZ(gs_layer)}")
        print(f"  → Neuronpedia: neuronpedia.org/gemma-3-12b-it/"
              f"{gs_layer}-gemmascope-2-res-{width}/")
        print(f"{'─'*70}")

        sae = load_sae(gs_layer, width, args.device)

        for feat_idx, feat_label in features:
            print(f"\n>>> Feature #{feat_idx}: \"{feat_label}\"")

            for exp_name in args.exp:
                if exp_name not in EXP_CONFIGS:
                    print(f"  Unknown experiment: {exp_name}")
                    continue

                exp_info = EXP_CONFIGS[exp_name]
                cfgs = args.cfgs or exp_info["cfgs"]
                label_names = exp_info["label_names"]

                print(f"\n  Experiment: {exp_name}")

                for cfg_id in cfgs:
                    result = analyze_config(
                        sae, gs_layer, feat_idx, feat_label,
                        exp_name, cfg_id, label_names,
                        top_k=args.top_k,
                    )
                    if result is None:
                        print(f"  {exp_name}_cfg{cfg_id}: file not found, skipping")
                        continue

                    print_result(result, feat_label, label_names)
                    all_results[f"{exp_name}_cfg{cfg_id}_L{gs_layer}_{feat_label}"] = result

        # Free SAE memory before loading next
        del sae
        torch.cuda.empty_cache()

    # Save JSON report
    out_path = os.path.join(RESULTS_DIR, "sae_feature_analysis.json")
    with open(out_path, "w") as f:
        # Remove non-serializable numpy types
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for key, res in all_results.items():
        labels = [k for k in res if k not in ("tag", "n_samples", "overall",
                                                "top_features_class1", "top_features_class0")]
        parts = [f"{name}: {res[name]['pct_active']:.1f}% active" for name in labels]
        print(f"  {key}: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
