#!/bin/bash
#SBATCH --job-name=nl_finegrain2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/nl_finegrain2_%j.out
#SBATCH --error=logs/nl_finegrain2_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source activate gemma4_env

python3 << 'PYEOF'
import torch, numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = 'gemma-4-31B-it-unsloth-bnb-4bit'
print('Loading model...')
proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH, dtype='auto', device_map='auto', local_files_only=True).eval()
print('Model loaded.\n')

# ── Build code pool ──
SPLIT = {'gq','lq','qf','qg','qj','tq','vj','wz','yq','yv'}
pair_to_id = {}
for a in 'abcdefghijklmnopqrstuvwxyz':
    for b in 'abcdefghijklmnopqrstuvwxyz':
        p = a + b
        if p in SPLIT or 'q' in p:
            continue
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) == 1:
            pair_to_id[p] = ids[0]
all_codes = sorted(pair_to_id.keys())
print(f'Available single-token codes: {len(all_codes)}')


def make_labels(start_h, start_m, count, step=5, pm=False):
    labels = []
    for i in range(count):
        total = start_h * 60 + start_m + i * step
        h = total // 60; m = total % 60
        suffix = 'PM' if pm else 'AM'
        disp_h = h - 12 if pm and h >= 13 else h
        labels.append(f'{disp_h}:{m:02d} {suffix}')
    return labels


def build_opts(labels, codes):
    return ', '.join(f'{c}={l}' for c, l in zip(codes, labels))


def run_prompt(name, text, codes, labels, pm=False):
    messages = [{'role': 'user', 'content': text}]
    text_enc = proc.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = tok(text_enc, return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_tok = inputs['input_ids'].shape[1]

    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    code_probs = {c: probs[pair_to_id[c]].item() for c in codes}
    total = sum(code_probs.values())
    ranked = sorted(code_probs.items(), key=lambda x: -x[1])

    topk = torch.topk(probs, 5)
    global_top = [(tok.decode([topk.indices[i].item()]).strip(),
                   topk.values[i].item(),
                   topk.indices[i].item() in pair_to_id.values())
                  for i in range(5)]

    print(f'\n{"="*70}')
    print(f'PROMPT: {name}')
    print(f'{"="*70}')
    print(f'Input tokens: {n_tok}, Options: {len(codes)}, Sum P: {total:.6f}')

    print(f'\nGlobal top-5:')
    for d, p, is_c in global_top:
        print(f'  {repr(d):>12s}  P={p:.6f}{"  <-- CODE" if is_c else ""}')

    print(f'\n{"Code":>5s}  {"Label":>10s}  {"P":>10s}  {"cumP":>8s}  bar')
    print('-' * 65)
    cum = 0
    for c, p in ranked[:25]:
        cum += p
        idx = codes.index(c) if c in codes else -1
        lbl = labels[idx] if idx >= 0 else '?'
        bar = '#' * int(p * 200)
        print(f'{c:>5s}  {lbl:>10s}  {p:>10.6f}  {cum:>8.4f}  {bar}')

    # E[time]
    try:
        nums = []
        for l in labels:
            clean = l.replace(' AM','').replace(' PM','')
            parts = clean.split(':')
            h = int(parts[0]); m = int(parts[1])
            if pm and h != 12: h += 12
            nums.append(h + m / 60.0)
        ev = sum(nums[i] * code_probs[codes[i]] for i in range(len(codes)))
        ev_h = int(ev); ev_m = int((ev - ev_h) * 60)
        print(f'\nE[time] = {ev:.3f} = {ev_h}:{ev_m:02d}')
    except Exception as e:
        print(f'\nE[time] error: {e}')

    spread_1 = sum(1 for _, p in code_probs.items() if p > 0.01)
    spread_half = sum(1 for _, p in code_probs.items() if p > 0.005)
    print(f'Spread: {spread_1} codes > 1%, {spread_half} codes > 0.5%')
    print(f'Top-1 P = {ranked[0][1]:.4f}')

    del out; torch.cuda.empty_cache()
    return code_probs


# ══════════════════════════════════════════════════════════════════════
# Labels: 5-minute intervals
# ══════════════════════════════════════════════════════════════════════

# Morning: 5:00 AM to 9:55 AM = 60 options
morn_labels = make_labels(5, 0, 60, step=5, pm=False)
morn_codes = all_codes[:60]

# Evening: 5:00 PM to 9:55 PM = 60 options
eve_labels = make_labels(17, 0, 60, step=5, pm=True)
eve_codes = all_codes[:60]

print(f'Morning: {morn_labels[0]} .. {morn_labels[-1]} ({len(morn_labels)} opts)')
print(f'Evening: {eve_labels[0]} .. {eve_labels[-1]} ({len(eve_labels)} opts)')


# ══════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════

# ── P1: Wake-up, day NOT specified (weekday/weekend target) ──
# Neutral prompt. Weekday steering → earlier. Weekend steering → later.
P1 = (
    "At what time does a typical adult wake up in the morning? "
    "Pick one: " + build_opts(morn_labels, morn_codes) + ". "
    "Reply with ONLY the two-letter code: "
)

# ── P2: Leave home, day NOT specified (weekday/weekend target) ──
# Weekday: leave ~7-8 AM for work. Weekend: ~10 AM or later.
P2 = (
    "At what time does a typical adult first leave their home? "
    "Pick one: " + build_opts(morn_labels, morn_codes) + ". "
    "Reply with ONLY the two-letter code: "
)

# ── P3: Wake-up on Tuesday (emp/unemp target) ──
# Day fixed → controls weekday/weekend. Employed: earlier. Unemployed: later.
P3 = (
    "At what time does a typical adult wake up on a Tuesday morning? "
    "Pick one: " + build_opts(morn_labels, morn_codes) + ". "
    "Reply with ONLY the two-letter code: "
)

# ── P4: Evening meal on Tuesday (emp/unemp target) ──
# REPLACES old "return home" which maxed out at 6:00 PM (99.6%).
# Dinner time has natural spread:
#   Employed: 7:00-8:00 PM (cook after commute + getting home)
#   Unemployed: 5:30-6:30 PM (already home, eat earlier)
# Range 5:00-9:55 PM captures full dinner window.
P4 = (
    "At what time does a typical adult sit down for their evening meal "
    "on a Tuesday? "
    "Pick one: " + build_opts(eve_labels, eve_codes) + ". "
    "Reply with ONLY the two-letter code: "
)


# ══════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════

prompts = [
    ('P1: Wake-up (no day — wkday/wkend)', P1, morn_codes, morn_labels, False),
    ('P2: Leave home (no day — wkday/wkend)', P2, morn_codes, morn_labels, False),
    ('P3: Wake-up Tue (emp/unemp)', P3, morn_codes, morn_labels, False),
    ('P4: Dinner Tue (emp/unemp)', P4, eve_codes, eve_labels, True),
]

results = {}
for name, text, codes, labels, pm in prompts:
    results[name] = run_prompt(name, text, codes, labels, pm)

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print('SUMMARY')
print(f'{"="*70}')
print(f'{"Prompt":>42s}  {"Peak":>8s}  {"PeakP":>7s}  {"E[t]":>6s}  {"SumP":>6s}  {">1%":>4s}  {">0.5%":>5s}')
print('-' * 85)

for name, text, codes, labels, pm in prompts:
    r = results[name]
    total = sum(r.values())
    ranked = sorted(r.items(), key=lambda x: -x[1])
    pc = ranked[0][0]; pp = ranked[0][1]
    idx = codes.index(pc)
    pl = labels[idx]
    nums = []
    for l in labels:
        clean = l.replace(' AM','').replace(' PM','')
        parts = clean.split(':')
        h = int(parts[0]); m = int(parts[1])
        if pm and h != 12: h += 12
        nums.append(h + m/60.0)
    ev = sum(nums[i] * r[codes[i]] for i in range(len(codes)))
    eh = int(ev); em = int((ev - eh)*60)
    s1 = sum(1 for _,p in r.items() if p > 0.01)
    s5 = sum(1 for _,p in r.items() if p > 0.005)
    print(f'{name:>42s}  {pl:>8s}  {pp:>7.4f}  {eh}:{em:02d}   {total:>6.4f}  {s1:>4d}  {s5:>5d}')

print(f'\nResolution: 5-minute intervals, 60 options per prompt')
print(f'DONE')
PYEOF
