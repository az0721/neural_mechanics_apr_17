#!/bin/bash
#SBATCH --job-name=nl_broad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/nl_broad_%j.out
#SBATCH --error=logs/nl_broad_%j.err

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

SPLIT = {'gq','lq','qf','qg','qj','tq','vj','wz','yq','yv'}
pair_to_id = {}
for a in 'abcdefghijklmnopqrstuvwxyz':
    for b in 'abcdefghijklmnopqrstuvwxyz':
        p = a + b
        if p in SPLIT or 'q' in p: continue
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) == 1: pair_to_id[p] = ids[0]
all_codes = sorted(pair_to_id.keys())

# ── 3 option formats ──

# TIME: 5:00-9:55 AM, 5-min = 60 opts
time_labels = []
for h in range(5, 10):
    for m in range(0, 60, 5):
        time_labels.append(f'{h}:{m:02d}AM')
time_codes = all_codes[:60]
time_c2l = dict(zip(time_codes, time_labels))
time_opts = ', '.join(f'{c}={l}' for c, l in zip(time_codes, time_labels))

# PERCENTAGE: 0%-100% in 2% steps = 51 opts
pct_labels = [f'{i}%' for i in range(0, 102, 2)]
pct_codes = all_codes[:51]
pct_c2l = dict(zip(pct_codes, pct_labels))
pct_opts = ', '.join(f'{c}={l}' for c, l in zip(pct_codes, pct_labels))

# DURATION: 0-120 min in 5-min steps = 25 opts
dur_labels = [f'{i}min' for i in range(0, 125, 5)]
dur_codes = all_codes[:25]
dur_c2l = dict(zip(dur_codes, dur_labels))
dur_opts = ', '.join(f'{c}={l}' for c, l in zip(dur_codes, dur_labels))

print(f'Time: {len(time_labels)} opts  Pct: {len(pct_labels)} opts  Dur: {len(dur_labels)} opts')


def test(name, prompt, codes_list, c2l_map):
    messages = [{'role': 'user', 'content': prompt}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_in = inputs['input_ids'].shape[1]

    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    cp = {c: probs[pair_to_id[c]].item() for c in codes_list}
    total = sum(cp.values())
    ranked = sorted(cp.items(), key=lambda x: -x[1])

    topk = torch.topk(probs, 3)
    gtop = [(tok.decode([topk.indices[i].item()]).strip(),
             topk.values[i].item(),
             topk.indices[i].item() in pair_to_id.values())
            for i in range(3)]

    print(f'\n{"="*70}')
    print(f'{name}')
    print(f'{"="*70}')
    print(f'Tok={n_in} Sum={total:.4f} Top3: '
          + ', '.join(f'{repr(d)}={p:.4f}{"*" if ic else ""}' for d,p,ic in gtop))

    cum = 0
    for c, p in ranked[:10]:
        cum += p
        bar = '#' * int(p * 200)
        print(f'  {c:>4s} {c2l_map[c]:>8s}  {p:.6f}  cum={cum:.4f}  {bar}')

    s1 = sum(1 for _, p in cp.items() if p > 0.01)
    s5 = sum(1 for _, p in cp.items() if p > 0.005)
    top1 = ranked[0][1]

    with torch.inference_mode():
        gen = model.generate(input_ids=inputs['input_ids'],
                              attention_mask=inputs['attention_mask'],
                              max_new_tokens=10, do_sample=False)
    g = tok.decode(gen[0, n_in:], skip_special_tokens=True).strip()
    gc = g.lower()[:2]
    valid = gc in c2l_map
    print(f'  >1%={s1} >0.5%={s5} Peak={top1:.4f} Gen="{g[:20]}" {"V" if valid else "X"}')

    del out, gen; torch.cuda.empty_cache()
    return top1, s1, s5, total


# ══════════════════════════════════════════════════════════════════════
# PROMPTS: 3 confirmed + 8 probability + 3 duration + 3 time = 17 total
# ══════════════════════════════════════════════════════════════════════

tests = []

# ── Confirmed good ──
tests.append(('T', '*P1 Wake-up noday [1a]',
    f'At what time does a typical adult wake up in the morning? Pick one: {time_opts}. Reply with ONLY the two-letter code: ',
    time_codes, time_c2l))
tests.append(('T', '*P3 Wake-up Tue [2]',
    f'At what time does a typical adult wake up on a Tuesday morning? Pick one: {time_opts}. Reply with ONLY the two-letter code: ',
    time_codes, time_c2l))
tests.append(('T', '*CandD Wake-up weekday [2]',
    f'At what time does a typical adult wake up on a weekday morning? Pick one: {time_opts}. Reply with ONLY the two-letter code: ',
    time_codes, time_c2l))

# ── Probability: exp2 (emp/unemp) ──
tests.append(('P', 'Prob1 P(home noon Tue) [2]',
    f'What is the probability that a typical adult is at home at 12:00 noon on a Tuesday? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))
tests.append(('P', 'Prob2 P(outside 3PM Tue) [2]',
    f'What is the probability that a typical adult is outside their home at 3:00 PM on a Tuesday? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))
tests.append(('P', 'Prob3 P(awake<7AM Tue) [2]',
    f'What is the probability that a typical adult is already awake before 7:00 AM on a Tuesday? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))
tests.append(('P', 'Prob4 P(brkfst by 8AM Tue) [2]',
    f'What is the probability that a typical adult has eaten breakfast by 8:00 AM on a Tuesday? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))
tests.append(('P', 'Prob5 P(transit 8AM Tue) [2]',
    f'What is the probability that a typical adult is commuting or in transit at 8:00 AM on a Tuesday? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))

# ── Probability: exp1a (weekday/weekend) ──
tests.append(('P', 'Prob6 P(home 10AM noday) [1a]',
    f'What is the probability that a typical adult is at home at 10:00 AM? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))
tests.append(('P', 'Prob7 P(bed 8AM noday) [1a]',
    f'What is the probability that a typical adult is still in bed at 8:00 AM? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))
tests.append(('P', 'Prob8 P(awake 630AM noday) [1a]',
    f'What is the probability that a typical adult is awake at 6:30 AM? Pick one: {pct_opts}. Reply with ONLY the two-letter code: ',
    pct_codes, pct_c2l))

# ── Duration: exp2 ──
tests.append(('D', 'Dur1 Commute mins Tue [2]',
    f'How many minutes does a typical adult spend commuting on a Tuesday morning? Pick one: {dur_opts}. Reply with ONLY the two-letter code: ',
    dur_codes, dur_c2l))
tests.append(('D', 'Dur2 MornRoutine mins Tue [2]',
    f'How many minutes does a typical adult spend on their morning routine on a Tuesday? Pick one: {dur_opts}. Reply with ONLY the two-letter code: ',
    dur_codes, dur_c2l))

# ── Duration: exp1a ──
tests.append(('D', 'Dur3 MornRoutine noday [1a]',
    f'How many minutes does a typical adult spend on their morning routine? Pick one: {dur_opts}. Reply with ONLY the two-letter code: ',
    dur_codes, dur_c2l))

# ── More time: exp1a ──
tests.append(('T', 'Alarm noday [1a]',
    f'At what time does a typical adult set their alarm clock? Pick one: {time_opts}. Reply with ONLY the two-letter code: ',
    time_codes, time_c2l))
tests.append(('T', 'GetUp noday [1a]',
    f'At what time does a typical adult get out of bed? Pick one: {time_opts}. Reply with ONLY the two-letter code: ',
    time_codes, time_c2l))
tests.append(('T', 'Shower noday [1a]',
    f'At what time does a typical adult take a shower in the morning? Pick one: {time_opts}. Reply with ONLY the two-letter code: ',
    time_codes, time_c2l))


# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

results = {}
for typ, name, prompt, cl, cm in tests:
    t1, s1, s5, total = test(name, prompt, cl, cm)
    results[name] = (typ, t1, s1, s5, total)

# ══════════════════════════════════════════════════════════════════════
# SUMMARY sorted by peak
# ══════════════════════════════════════════════════════════════════════

print(f'\n{"="*70}')
print('FULL RANKING (sorted by Peak, lower = better)')
print(f'Target: Peak < 0.70, Spread(>1%) >= 3')
print(f'{"="*70}')
print(f'{"Ty":>2s}  {"Prompt":>40s}  {"Peak":>6s}  {">1%":>4s}  {">0.5%":>5s}  {"Sum":>6s}  {"Grade":>8s}')
print('-' * 80)

sorted_n = sorted(results.keys(), key=lambda n: results[n][1])
for name in sorted_n:
    typ, t1, s1, s5, total = results[name]
    if t1 < 0.45:     v = 'A+'
    elif t1 < 0.55:   v = 'A'
    elif t1 < 0.65:   v = 'B+'
    elif t1 < 0.75:   v = 'B'
    elif t1 < 0.85:   v = 'C'
    elif t1 < 0.95:   v = 'D'
    else:             v = 'F'
    print(f'{typ:>2s}  {name:>40s}  {t1:>6.3f}  {s1:>4d}  {s5:>5d}  {total:>6.4f}  {v:>8s}')

print(f'\nGrading: A+=<0.45 A=<0.55 B+=<0.65 B=<0.75 C=<0.85 D=<0.95 F=>=0.95')
print('DONE')
PYEOF
