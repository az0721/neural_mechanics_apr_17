#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steering Test — Verify model answers employment questions correctly.
Tests 10 users × 3 prompts, no steering, just check behavior.

Usage:
    sbatch -p gpu --gres=gpu:h200:1 -t 1:00:00 --mem=64G \
        --wrap="cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7 && \
                source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh && \
                conda activate unsloth_env && module purge && module load cuda/12.1.1 && \
                export HF_HOME=/scratch/\$USER/.cache/huggingface && \
                export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && \
                python -u steering_test.py"
"""
import sys, os, random, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_DIR, EXP_CONFIG, SAMPLING_CONFIG, RANDOM_SEED,
                    OUTPUT_DIR, GENERATION_CONFIG)
from utils import (load_data, load_model, build_prompt, wrap_with_chat_template,
                   _targets_exp2, _filter_window, _fmt_geo, _ts,
                   CONFIG_MATRIX_CACHE)

# ── 3 Prompt endings ──
PROMPTS = {
    'binary': (
        "=== Question ===\n"
        "Based on the mobility pattern shown above, is this person "
        "currently employed?\n"
        "Answer with only Yes or No:"
    ),
    'behavioral': (
        "=== Question ===\n"
        "Based on this person's typical daily mobility pattern, at 12:00 "
        "on a typical weekday, where is this person most likely?\n"
        "(A) At home or near home\n"
        "(B) At a workplace or office area\n"
        "(C) Traveling between locations\n"
        "Answer with only the letter:"
    ),
    'routine': (
        "=== Question ===\n"
        "This person's daily mobility pattern most closely resembles:\n"
        "(A) Someone with a regular 9-to-5 job who commutes daily\n"
        "(B) Someone who stays home most of the day without regular commute\n"
        "(C) Someone with an irregular or part-time schedule\n"
        "Answer with only the letter:"
    ),
}

# Target tokens for logit extraction
TARGET_TOKENS = {
    'binary': ['Yes', 'No'],
    'behavioral': ['A', 'B', 'C'],
    'routine': ['A', 'B', 'C'],
}


def build_steering_prompt(user_df, target_date, predict_time, cfg_dict, question_key):
    """Build prompt with original history but custom question ending."""
    exp_cfg = EXP_CONFIG['exp2']
    gh, dh = cfg_dict['geo_hash'], cfg_dict['day_hash']

    from utils import _history_exp2
    hist = _history_exp2(user_df, target_date, exp_cfg['hist_window'], gh, dh)
    if hist is None:
        return None

    ctx = _filter_window(user_df[user_df['date'] == target_date], exp_cfg['time_window'])
    if len(ctx) == 0:
        return None
    ctx_lines = []
    for _, r in ctx.iterrows():
        geo = _fmt_geo(r['geo_id'], gh)
        ts = _ts(r)
        from utils import _fmt_dow
        ctx_lines.append(f"{_fmt_dow(r['dow'], dh)}, {ts}, {geo}")

    prompt = (
        f"Your task is to analyze an individual's mobility pattern "
        f"based on their location history.\n"
        f"Each record: {exp_cfg['prompt_fmt']}\n\n"
        f"=== Mobility History ===\n"
        + "\n".join(hist) + "\n\n"
        f"=== Current Day Context ===\n"
        + "\n".join(ctx_lines) + "\n\n"
        + PROMPTS[question_key]
    )
    return prompt


def get_target_logits(logits, tokenizer, token_names):
    """Extract logits for specific tokens."""
    results = {}
    probs = torch.softmax(logits, dim=-1)
    for name in token_names:
        tid = tokenizer.encode(name, add_special_tokens=False)
        if len(tid) == 1:
            results[name] = {
                'logit': logits[tid[0]].item(),
                'prob': probs[tid[0]].item(),
                'token_id': tid[0],
            }
        else:
            # Try with space prefix
            tid2 = tokenizer.encode(f" {name}", add_special_tokens=False)
            if len(tid2) >= 1:
                results[name] = {
                    'logit': logits[tid2[-1]].item(),
                    'prob': probs[tid2[-1]].item(),
                    'token_id': tid2[-1],
                }
    return results


def main():
    print("="*60)
    print("Steering Test: Verify model can answer employment questions")
    print("="*60)

    # Load data
    df = load_data()
    import pandas as pd
    users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))

    # Pick 5 employed + 5 unemployed
    random.seed(RANDOM_SEED)
    emp_ids = users_df[users_df['is_employed'] == 1]['cuebiq_id'].tolist()
    unemp_ids = users_df[users_df['is_employed'] == 0]['cuebiq_id'].tolist()
    test_emp = random.sample(emp_ids, 5)
    test_unemp = random.sample(unemp_ids, 5)
    test_users = [(uid, True) for uid in test_emp] + [(uid, False) for uid in test_unemp]

    # Use cfg1 (orig_orig_none) — no hash, no CoT
    cfg_dict = CONFIG_MATRIX_CACHE['exp2'][0]  # cfg1

    # Load model
    model, tokenizer = load_model('12b')
    device = next(model.parameters()).device

    results = []

    for uid, is_emp in test_users:
        user_df = df[df['cuebiq_id'] == uid]
        exp_df = user_df.query(EXP_CONFIG['exp2']['filter_query'])

        random.seed(RANDOM_SEED + hash(uid) % 10000)
        targets = _targets_exp2(exp_df)
        if targets is None:
            continue
        tdate = targets[0]

        tag = "EMP" if is_emp else "UNEMP"
        print(f"\n{'='*60}")
        print(f"User: {uid[:12]}... ({tag})")

        for qkey in ['binary', 'behavioral', 'routine']:
            random.seed(RANDOM_SEED + hash(f"{uid}_{tdate}_12:00") % 10000)
            prompt = build_steering_prompt(exp_df, tdate, '12:00', cfg_dict, qkey)
            if prompt is None:
                continue

            full = wrap_with_chat_template(tokenizer, prompt, False)
            inputs = tokenizer(full, return_tensors="pt", truncation=True,
                               max_length=128000)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            n_tok = inputs['input_ids'].shape[1]

            # Forward pass → logits
            with torch.no_grad():
                out = model(**inputs)
            logits_last = out.logits[0, -1, :]
            target_logits = get_target_logits(logits_last, tokenizer,
                                              TARGET_TOKENS[qkey])

            # Generate answer
            gen_cfg = {**GENERATION_CONFIG, 'max_new_tokens': 128}
            gen = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id=tokenizer.eos_token_id,
                **gen_cfg)
            gen_text = tokenizer.decode(
                gen[0, inputs['input_ids'].shape[1]:],
                skip_special_tokens=False).strip()

            del out, gen, inputs
            torch.cuda.empty_cache()

            print(f"\n  [{qkey}] tokens={n_tok}")
            for tok, info in target_logits.items():
                print(f"    P({tok:3s}) = {info['prob']:.4f}  "
                      f"(logit={info['logit']:.2f})")
            print(f"    Generated: {gen_text[:150]}")

            results.append({
                'user': uid[:16], 'is_employed': is_emp,
                'question': qkey, 'n_tokens': n_tok,
                'target_logits': {k: {'prob': v['prob'], 'logit': v['logit']}
                                  for k, v in target_logits.items()},
                'generated_text': gen_text,
            })

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'steering_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved: {out_path}")
    print(f"Total: {len(results)} tests")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for qkey in ['binary', 'behavioral', 'routine']:
        print(f"\n  [{qkey}]")
        for label, tag in [(True, 'EMP'), (False, 'UNEMP')]:
            subset = [r for r in results
                      if r['question'] == qkey and r['is_employed'] == label]
            if not subset:
                continue
            if qkey == 'binary':
                mean_yes = np.mean([r['target_logits'].get('Yes', {}).get('prob', 0)
                                    for r in subset])
                mean_no = np.mean([r['target_logits'].get('No', {}).get('prob', 0)
                                   for r in subset])
                print(f"    {tag}: P(Yes)={mean_yes:.3f}  P(No)={mean_no:.3f}")
            else:
                mean_a = np.mean([r['target_logits'].get('A', {}).get('prob', 0)
                                  for r in subset])
                mean_b = np.mean([r['target_logits'].get('B', {}).get('prob', 0)
                                  for r in subset])
                mean_c = np.mean([r['target_logits'].get('C', {}).get('prob', 0)
                                  for r in subset])
                print(f"    {tag}: P(A)={mean_a:.3f}  P(B)={mean_b:.3f}  "
                      f"P(C)={mean_c:.3f}")


if __name__ == "__main__":
    main()