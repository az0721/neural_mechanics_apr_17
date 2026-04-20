#!/bin/bash
#SBATCH --job-name=nl_final4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/nl_final4_%j.out
#SBATCH --error=logs/nl_final4_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source activate gemma4_env

python3 << 'PYEOF'
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = 'gemma-4-31B-it-unsloth-bnb-4bit'
print('Loading model...')
proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH, dtype='auto', device_map='auto', local_files_only=True).eval()
print('Model loaded.\n')

# ══════════════════════════════════════════════════════════════════════
# Build code pool (exclude q-containing and known split pairs)
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# Option set 1: TIME — 5:00 AM to 9:55 AM, 5-min intervals, 60 options
# ══════════════════════════════════════════════════════════════════════
time_labels = []
for h in range(5, 10):
    for m in range(0, 60, 5):
        time_labels.append(f'{h}:{m:02d}AM')
time_codes = all_codes[:60]
time_c2l = dict(zip(time_codes, time_labels))
time_opts = ', '.join(f'{c}={l}' for c, l in zip(time_codes, time_labels))
print(f'TIME options: {len(time_labels)} ({time_labels[0]} to {time_labels[-1]})')

# ══════════════════════════════════════════════════════════════════════
# Option set 2: PERCENTAGE — 0% to 100%, 2% steps, 51 options
# ══════════════════════════════════════════════════════════════════════
pct_labels = [f'{i}%' for i in range(0, 102, 2)]
pct_codes = all_codes[:51]
pct_c2l = dict(zip(pct_codes, pct_labels))
pct_opts = ', '.join(f'{c}={l}' for c, l in zip(pct_codes, pct_labels))
print(f'PCT  options: {len(pct_labels)} ({pct_labels[0]} to {pct_labels[-1]})')

# ══════════════════════════════════════════════════════════════════════
# The 4 Final Prompts
# ══════════════════════════════════════════════════════════════════════

PROMPTS = [
    {
        'name': 'EXP1A-Q1: Wake-up time (no day specified)',
        'exp': 'exp1a',
        'type': 'time',
        'codes': time_codes,
        'c2l': time_c2l,
        'prompt': (
            'At what time does a typical adult wake up in the morning? '
            f'Pick one: {time_opts}. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    {
        'name': 'EXP1A-Q2: P(awake at 6:30 AM, no day specified)',
        'exp': 'exp1a',
        'type': 'pct',
        'codes': pct_codes,
        'c2l': pct_c2l,
        'prompt': (
            'What is the probability that a typical adult is awake at 6:30 AM? '
            f'Pick one: {pct_opts}. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    {
        'name': 'EXP2-Q1: P(awake before 7 AM on Tuesday)',
        'exp': 'exp2',
        'type': 'pct',
        'codes': pct_codes,
        'c2l': pct_c2l,
        'prompt': (
            'What is the probability that a typical adult is already awake '
            'before 7:00 AM on a Tuesday? '
            f'Pick one: {pct_opts}. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    {
        'name': 'EXP2-Q2: P(eaten breakfast by 8 AM on Tuesday)',
        'exp': 'exp2',
        'type': 'pct',
        'codes': pct_codes,
        'c2l': pct_c2l,
        'prompt': (
            'What is the probability that a typical adult has eaten breakfast '
            'by 8:00 AM on a Tuesday? '
            f'Pick one: {pct_opts}. '
            'Reply with ONLY the two-letter code: '
        ),
    },
]

# ══════════════════════════════════════════════════════════════════════
# Run each prompt
# ══════════════════════════════════════════════════════════════════════

results = []

for P in PROMPTS:
    name = P['name']
    codes = P['codes']
    c2l = P['c2l']

    messages = [{'role': 'user', 'content': P['prompt']}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_in = inputs['input_ids'].shape[1]

    # ── Logit extraction ──
    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    cp = {c: probs[pair_to_id[c]].item() for c in codes}
    total = sum(cp.values())
    ranked = sorted(cp.items(), key=lambda x: -x[1])

    # Global top-5
    topk = torch.topk(probs, 5)
    gtop = [(tok.decode([topk.indices[i].item()]).strip(),
             topk.values[i].item(),
             topk.indices[i].item() in pair_to_id.values())
            for i in range(5)]

    # ── Greedy generation ──
    with torch.inference_mode():
        gen = model.generate(input_ids=inputs['input_ids'],
                              attention_mask=inputs['attention_mask'],
                              max_new_tokens=10, do_sample=False)
    greedy = tok.decode(gen[0, n_in:], skip_special_tokens=True).strip()
    greedy_code = greedy.lower()[:2]
    greedy_valid = greedy_code in c2l
    del gen

    # ── Stochastic generation x5 ──
    stoch = []
    for _ in range(5):
        with torch.inference_mode():
            gen = model.generate(input_ids=inputs['input_ids'],
                                  attention_mask=inputs['attention_mask'],
                                  max_new_tokens=10, do_sample=True,
                                  temperature=1.0, top_p=0.95, top_k=64)
        g = tok.decode(gen[0, n_in:], skip_special_tokens=True).strip()
        gc = g.lower()[:2]
        stoch.append((gc, c2l.get(gc, '???'), gc in c2l))
        del gen

    # ── Stats ──
    s1 = sum(1 for _, p in cp.items() if p > 0.01)
    s5 = sum(1 for _, p in cp.items() if p > 0.005)
    top1 = ranked[0][1]

    # ── Print ──
    print(f'\n{"="*70}')
    print(f'{name}')
    print(f'Experiment: {P["exp"]}  |  Type: {P["type"]}  |  Options: {len(codes)}')
    print(f'{"="*70}')
    print(f'Input tokens: {n_in}')
    print(f'Sum P(codes): {total:.6f}')
    print()

    print('Global top-5 tokens:')
    for d, p, ic in gtop:
        print(f'  {repr(d):>12s}  P={p:.6f}{"  <-- VALID CODE" if ic else ""}')
    print()

    print(f'{"Rank":>4s}  {"Code":>5s}  {"Label":>8s}  {"P":>10s}  {"CumP":>8s}  bar')
    print('-' * 70)
    cum = 0
    for i, (c, p) in enumerate(ranked[:15]):
        cum += p
        bar = '#' * int(p * 200)
        print(f'{i+1:>4d}  {c:>5s}  {c2l[c]:>8s}  {p:>10.6f}  {cum:>8.4f}  {bar}')
    print()

    print(f'Peak P:       {top1:.4f}')
    print(f'Spread >1%:   {s1}')
    print(f'Spread >0.5%: {s5}')
    print()

    print(f'Greedy generation: "{greedy}" → {greedy_code} = {c2l.get(greedy_code, "???")} '
          f'{"✓ VALID" if greedy_valid else "✗ INVALID"}')
    print()

    print('Stochastic generation (temp=1.0, 5 runs):')
    for i, (gc, lbl, valid) in enumerate(stoch):
        print(f'  Run {i+1}: {gc} = {lbl} {"✓" if valid else "✗"}')

    del out
    torch.cuda.empty_cache()

    results.append({
        'name': name, 'exp': P['exp'], 'type': P['type'],
        'peak': top1, 'spread1': s1, 'spread5': s5,
        'total': total, 'greedy': greedy_code,
        'greedy_label': c2l.get(greedy_code, '???'),
        'greedy_valid': greedy_valid,
        'stoch_valid': sum(1 for _, _, v in stoch if v),
    })

# ══════════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print('FINAL SUMMARY — 4 Selected Prompts')
print(f'{"="*70}')
print(f'{"Prompt":>50s}  {"Exp":>5s}  {"Type":>4s}  {"Peak":>6s}  {">1%":>4s}  {">0.5%":>5s}  {"Gen":>6s}')
print('-' * 90)
for r in results:
    print(f'{r["name"]:>50s}  {r["exp"]:>5s}  {r["type"]:>4s}  '
          f'{r["peak"]:>6.3f}  {r["spread1"]:>4d}  {r["spread5"]:>5d}  '
          f'{r["greedy_label"]:>6s} {"✓" if r["greedy_valid"] else "✗"} '
          f'({r["stoch_valid"]}/5)')

all_valid = all(r['greedy_valid'] for r in results)
all_spread = all(r['spread1'] >= 3 for r in results)
print(f'\nAll greedy outputs valid: {"YES ✓" if all_valid else "NO ✗"}')
print(f'All spread >= 3:         {"YES ✓" if all_spread else "NO ✗"}')
print('\nDONE')
PYEOF
