#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge per-user steering JSONs into combined analysis-ready data.

Usage:
    python steering/merge_steering.py
"""
import sys, os, json, ast
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR
from steering.prompts import STEERING_PROMPTS

STEER_DIR = os.path.join(OUTPUT_DIR, 'steering')
PERUSER_DIR = os.path.join(STEER_DIR, 'per_user')
RESULTS_DIR = os.path.join(STEER_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_key(key_str):
    """Parse string key like '(5, \"binary\", 25, 200)' back to tuple."""
    return ast.literal_eval(key_str)


def main():
    files = sorted(f for f in os.listdir(PERUSER_DIR) if f.endswith('.json'))
    print(f"Found {len(files)} user files")
    if not files:
        return

    # Collect all data
    all_data = []
    for fname in files:
        with open(os.path.join(PERUSER_DIR, fname)) as f:
            d = json.load(f)
        all_data.append(d)

    # Extract config from first file
    meta0 = all_data[0]['meta']
    cfgs = meta0['cfgs']
    layers = meta0['layers']
    coeffs = meta0['coeffs']
    prompts = list(STEERING_PROMPTS.keys())

    print(f"  Configs: {cfgs}")
    print(f"  Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"  Coeffs: {coeffs}")
    print(f"  Prompts: {prompts}")

    # ── Build logits arrays ──
    # Shape: (n_users, n_cfgs, n_prompts, n_layers, n_coeffs, n_targets)
    emp_users = [d for d in all_data if d['meta']['is_employed']]
    unemp_users = [d for d in all_data if not d['meta']['is_employed']]
    print(f"  Employed: {len(emp_users)}, Unemployed: {len(unemp_users)}")

    combined = {
        'cfgs': cfgs, 'layers': layers, 'coeffs': coeffs,
        'prompts': prompts, 'n_emp': len(emp_users), 'n_unemp': len(unemp_users),
    }

    for qkey in prompts:
        targets = STEERING_PROMPTS[qkey]['targets']
        if not targets:
            continue  # skip 'location' for now

        for cfg_id in cfgs:
            # Arrays: (n_users, n_layers, n_coeffs)
            # Store probability of "employed indicator" target
            emp_indicator = STEERING_PROMPTS[qkey]['employed_target']

            for label, group, tag in [(1, emp_users, 'emp'), (0, unemp_users, 'unemp')]:
                arr = np.full((len(group), len(layers), len(coeffs)), np.nan)
                for ui, d in enumerate(group):
                    for li, layer in enumerate(layers):
                        for ci, coeff in enumerate(coeffs):
                            key_str = str((cfg_id, qkey, layer, coeff))
                            probs = d['logits'].get(key_str, {})
                            if emp_indicator and emp_indicator in probs:
                                arr[ui, li, ci] = probs[emp_indicator]

                key = f"logits_{qkey}_cfg{cfg_id}_{tag}"
                combined[key] = arr.tolist()
                mean = np.nanmean(arr, axis=0)  # (n_layers, n_coeffs)
                combined[f"mean_{qkey}_cfg{cfg_id}_{tag}"] = mean.tolist()

    # ── Collect greedy texts ──
    greedy = {}
    for d in all_data:
        uid = d['meta']['user'][:12]
        tag = 'emp' if d['meta']['is_employed'] else 'unemp'
        for key_str, text in d.get('greedy', {}).items():
            greedy[f"{uid}_{tag}_{key_str}"] = text

    combined['greedy_texts'] = greedy

    # ── Save ──
    out_path = os.path.join(RESULTS_DIR, 'steering_combined.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f)
    print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
