#!/bin/bash
#SBATCH --job-name=twolet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/twolet_%j.out
#SBATCH --error=logs/twolet_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source activate gemma4_env

python3 -c "
import torch, numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = 'gemma-4-31B-it-unsloth-bnb-4bit'
print('Loading model...')
proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH, dtype='auto', device_map='auto', local_files_only=True).eval()

# All 666 valid two-letter tokens (exclude 10 split pairs)
SPLIT_PAIRS = {'gq','lq','qf','qg','qj','tq','vj','wz','yq','yv'}
all_pairs = []
for a in 'abcdefghijklmnopqrstuvwxyz':
    for b in 'abcdefghijklmnopqrstuvwxyz':
        p = a+b
        if p not in SPLIT_PAIRS:
            all_pairs.append(p)
print(f'Valid two-letter tokens: {len(all_pairs)}')

# Verify they are single tokens
pair_to_id = {}
for p in all_pairs:
    ids = tok.encode(p, add_special_tokens=False)
    if len(ids) == 1:
        pair_to_id[p] = ids[0]
    else:
        print(f'  WARNING: {p} -> {len(ids)} tokens, skipping')
print(f'Confirmed single-token pairs: {len(pair_to_id)}')

# Build mapping: 60 options, 5-min intervals from 5:00 to 9:55
times = []
for h in range(5, 10):
    for m in range(0, 60, 5):
        times.append(f'{h}:{m:02d} AM')
times = times[:60]

# Pick 60 codes from pair_to_id (skip q-containing ones)
codes = [p for p in all_pairs if 'q' not in p][:60]
code_to_time = dict(zip(codes, times))
time_to_code = dict(zip(times, codes))

# Build prompt
options = '\n'.join(f'- {c} = {t}' for c, t in zip(codes, times))

PROMPT = f'''A typical employed adult wakes up at approximately what time on a Tuesday morning?

Below are 60 options representing times from 5:00 AM to 9:55 AM in 5-minute intervals. Pick the single most likely time by responding with ONLY the two-letter code.

{options}

Please respond with ONLY the two-letter code above: '''

messages = [{'role': 'user', 'content': PROMPT}]
text = proc.apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=True,
                                 enable_thinking=False)
inputs = tok(text, return_tensors='pt', add_special_tokens=False)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    out = model(**inputs)
logits = out.logits[0, -1, :].float()
probs = torch.softmax(logits, dim=-1)

# Extract probabilities for all 60 codes
code_probs = {}
for c in codes:
    tid = pair_to_id[c]
    code_probs[c] = probs[tid].item()

total = sum(code_probs.values())
print(f'\nSum of P over 60 codes: {total:.6f}')

# Sort by probability
ranked = sorted(code_probs.items(), key=lambda x: -x[1])

print(f'\n{\"Code\":>5s}  {\"Time\":>10s}  {\"P\":>10s}  {\"cumP\":>10s}  bar')
print('-' * 65)
cum = 0
for c, p in ranked[:30]:
    cum += p
    bar = '#' * int(p * 200)
    print(f'{c:>5s}  {code_to_time[c]:>10s}  {p:>10.6f}  {cum:>10.4f}  {bar}')

print(f'\n--- Top-10 global tokens (any token) ---')
topk = torch.topk(probs, 20)
for i in range(20):
    tid = topk.indices[i].item()
    p = topk.values[i].item()
    decoded = repr(tok.decode([tid]))
    is_code = tid in pair_to_id.values()
    flag = ' <-- TWO-LETTER CODE' if is_code else ''
    print(f'  {decoded:>15s} (id={tid:>6d})  P={p:.6f}{flag}')
"
