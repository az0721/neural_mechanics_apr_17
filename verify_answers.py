#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, re, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PER_USER_DIR
from utils import parse_answer

files = sorted(f for f in os.listdir(PER_USER_DIR) if f.endswith('.npz'))
print(f"Checking {len(files)} user files...\n")

stats = {'total': 0, 'code_block': 0, 'final_answer': 0,
         'backtick': 0, 'direct': 0, 'bad_parse': 0}
bad = []

for f in files:
    d = np.load(os.path.join(PER_USER_DIR, f), allow_pickle=True)
    for i, (a, m) in enumerate(zip(d['answers'], d['meta'])):
        stats['total'] += 1
        gt = a.get('generated_text', '')
        cleaned = gt.replace('<end_of_turn>', '').strip()
        parsed = parse_answer(gt)

        # Classify which parse path was used
        if re.findall(r'```(?:text)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL):
            stats['code_block'] += 1
        elif re.search(r'Final Answer:\s*(\S+)', cleaned):
            stats['final_answer'] += 1
        elif re.findall(r'`([a-f0-9]{6,15})`', cleaned):
            stats['backtick'] += 1
        else:
            stats['direct'] += 1

        # Check: parsed should be short (geo_id) for non-CoT,
        # or extracted geo_id for CoT
        is_cot = m.get('config_id', 0) in [2, 4, 6, 8]
        if is_cot and len(parsed) > 50:
            stats['bad_parse'] += 1
            bad.append({
                'user': str(m['user'])[:12],
                'exp': m['exp_name'],
                'cfg': m['config_id'],
                'date': m['date'],
                'time': m['pred_time'],
                'parsed_len': len(parsed),
                'parsed_preview': parsed[:80],
                'generated_preview': gt[:120],
            })

print(f"=== Parse Stats ===")
print(f"Total samples:  {stats['total']}")
print(f"Code block:     {stats['code_block']}")
print(f"Final Answer:   {stats['final_answer']}")
print(f"Backtick:       {stats['backtick']}")
print(f"Direct (short): {stats['direct']}")
print(f"Bad parse (CoT but >50 chars): {stats['bad_parse']}")

if bad:
    print(f"\n=== Bad Parses ({len(bad)}) ===")
    for b in bad[:20]:
        print(f"  {b['user']} {b['exp']}_cfg{b['cfg']} {b['date']} {b['time']}")
        print(f"    parsed ({b['parsed_len']} chars): {b['parsed_preview']}")
        print(f"    generated: {b['generated_preview']}")
        print()

    # Save full report
    with open('outputs_v7/bad_parses.json', 'w') as f:
        json.dump(bad, f, indent=2)
    print(f"Full report: outputs_v7/bad_parses.json")
else:
    print(f"\nALL GOOD — every CoT answer parsed to a short geo_id")
