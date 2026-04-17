#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export all raw answers to structured JSON for review.
Groups by exp/cfg with full generated_text, parsed_answer, top_tokens.

Usage:
    python export_raw_answers.py
Output:
    outputs_v7/raw_answers_export.json
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PER_USER_DIR, OUTPUT_DIR
from utils import parse_answer

files = sorted(f for f in os.listdir(PER_USER_DIR) if f.endswith('.npz'))
print(f"Loading {len(files)} user files...")

export = {}
for f in files:
    d = np.load(os.path.join(PER_USER_DIR, f), allow_pickle=True)
    for a, m, label in zip(d['answers'], d['meta'], d['labels']):
        exp = m['exp_name']
        cfg = int(m['config_id'])
        key = f"{exp}_cfg{cfg}"
        if key not in export:
            export[key] = []

        reparsed = parse_answer(a.get('generated_text', ''))
        export[key].append({
            'user': str(m['user'])[:16],
            'date': m['date'],
            'pred_time': m['pred_time'],
            'label': int(label),
            'gt_geo': m.get('gt_geo_id', ''),
            'is_employed': m.get('is_employed', -1),
            'n_tokens': a.get('n_tokens', 0),
            'status': a.get('status', ''),
            'top_tokens': a.get('top_tokens', []),
            'top_probs': a.get('top_probs', []),
            'generated_text': a.get('generated_text', ''),
            'parsed_answer': reparsed,
            'parse_correct': len(reparsed) < 50,
        })

# Summary
print(f"\n{'='*60}")
print(f"Export Summary")
print(f"{'='*60}")
total = 0
for key in sorted(export.keys()):
    samples = export[key]
    n = len(samples)
    n_bad = sum(1 for s in samples if not s['parse_correct'])
    n_match = sum(1 for s in samples if s['parsed_answer'] == s['gt_geo'])
    total += n
    print(f"  {key:20s}: {n:4d} samples, {n_bad} bad parse, "
          f"{n_match} exact match ({n_match/n*100:.1f}%)")

out_path = os.path.join(OUTPUT_DIR, 'raw_answers_export.json')
with open(out_path, 'w') as f:
    json.dump(export, f, indent=2)

print(f"\nTotal: {total} samples")
print(f"Saved: {out_path}")
print(f"Size: {os.path.getsize(out_path)/1e6:.1f} MB")
