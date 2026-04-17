#!/bin/bash
#SBATCH --job-name=w19_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/w19_diag_%j.out
#SBATCH --error=logs/w19_diag_%j.err

cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
source activate gemma4_env

python3 -c "
import torch, json
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = 'gemma-4-31B-it-unsloth-bnb-4bit'
print('Loading model...')
proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH, dtype='auto', device_map='auto', local_files_only=True).eval()
print('Model loaded.')

# Exact W19 prompt from prompts_nl.py
W19_PROMPT = (
    'Is an individual at the same non-home location '
    'at noon today as they were at noon last week?\n'
    '(A) No, they are at home today\n'
    '(B) They are out, but at a different location\n'
    '(C) Yes, at the same location\n'
    'Answer with only the letter A, B, or C: '
)

# Also test a reworded version for comparison
W19_ALT = (
    'Does an individual tend to be at the same non-home location '
    'at noon on the same day each week?\n'
    '(A) No, they are usually at home at noon\n'
    '(B) They are out, but at a different location each week\n'
    '(C) Yes, almost always at the same location\n'
    'Answer with only the letter A, B, or C: '
)

# And E20 (same topic but works fine) for reference
E20_PROMPT = (
    'Is a typical adult at the same non-home location '
    'at noon every Tuesday?\n'
    '(A) No, they are usually at home at noon\n'
    '(B) Sometimes the same place, sometimes not\n'
    '(C) Yes, almost always at the same location\n'
    'Answer with only the letter A, B, or C: '
)

tests = [
    ('W19 (original — broken)', W19_PROMPT),
    ('W19 (reworded — proposed fix)', W19_ALT),
    ('E20 (similar topic — works fine)', E20_PROMPT),
]

for name, prompt in tests:
    print(f'\n{\"=\"*70}')
    print(f'{name}')
    print(f'{\"=\"*70}')
    print(f'Prompt:\n{prompt}')

    for thinking in [False, True]:
        mode = 'thinking OFF' if not thinking else 'thinking ON'
        messages = [{'role': 'user', 'content': prompt}]
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking)

        inputs = tok(text, return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        n_in = inputs['input_ids'].shape[1]

        # 1. Logit check
        with torch.inference_mode():
            out = model(**inputs)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        abc = {}
        for L in 'ABC':
            best = 0
            for s in [L, f' {L}']:
                ids = tok.encode(s, add_special_tokens=False)
                for i in ids:
                    best = max(best, probs[i].item())
            abc[L] = best
        total = sum(abc.values())

        topk = torch.topk(probs, 10)
        top10 = [(topk.indices[i].item(), topk.values[i].item()) for i in range(10)]
        top_str = '  '.join(f'{repr(tok.decode([t])):>12s}={p:.6f}' for t, p in top10[:5])

        print(f'\n  [{mode}]')
        print(f'  P(A)={abc[\"A\"]:.6f}  P(B)={abc[\"B\"]:.6f}  P(C)={abc[\"C\"]:.6f}  sum={total:.6f}')
        print(f'  Top5: {top_str}')

        # 2. Actual generation
        gen = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=200,
            do_sample=False,
        )
        gen_text = tok.decode(gen[0, n_in:], skip_special_tokens=True).strip()
        print(f'  Generated: \"{gen_text[:200]}\"')

        del out, gen
        torch.cuda.empty_cache()

print(f'\n{\"=\"*70}')
print('DONE')
print(f'{\"=\"*70}')
"
