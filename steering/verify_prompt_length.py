#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify steering v8 prompt lengths vs context window.
Run on login node (no GPU needed, only tokenizer).

Usage:
    python steering/verify_prompt_length.py
"""
import sys, os, json, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR

PROMPT_DIR = os.path.join(DATA_DIR, 'steering_prompts_v8')
MODEL_CONTEXT = 128_000
GEN_TOKENS = 2_048  # max_new_tokens for trajectory generation


def main():
    # Try loading tokenizer (offline mode)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'google/gemma-3-12b-it', local_files_only=True)
        has_tok = True
        print("Tokenizer loaded (exact token counts)")
    except Exception as e:
        has_tok = False
        print(f"Tokenizer not available ({e})")
        print("Using char/4 approximation\n")

    files = sorted(glob.glob(os.path.join(PROMPT_DIR, '*.json')))
    if not files:
        print(f"No prompts found in {PROMPT_DIR}")
        print("Run: python steering/prebuild_steering_prompts_v8.py first")
        return

    print(f"\n{'='*70}")
    print(f"Steering v8 Prompt Length Verification")
    print(f"Context window: {MODEL_CONTEXT:,} tokens")
    print(f"Generation budget: {GEN_TOKENS:,} tokens")
    print(f"Available for prompt: {MODEL_CONTEXT - GEN_TOKENS:,} tokens")
    print(f"{'='*70}\n")

    results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        prompt = data['prompt']
        meta = data['meta']
        uid = meta['user'][:16]
        n_lines = meta['n_hist_lines']
        n_chars = len(prompt)

        if has_tok:
            # Wrap with chat template for accurate count
            from utils import wrap_with_chat_template
            wrapped = wrap_with_chat_template(tokenizer, prompt, False)
            toks = tokenizer(wrapped, return_tensors='pt',
                             truncation=False, max_length=200_000)
            n_tokens = toks['input_ids'].shape[1]
        else:
            n_tokens = n_chars // 4  # rough approximation

        headroom = MODEL_CONTEXT - n_tokens - GEN_TOKENS
        pct = n_tokens / MODEL_CONTEXT * 100

        results.append({
            'uid': uid, 'n_lines': n_lines, 'n_chars': n_chars,
            'n_tokens': n_tokens, 'headroom': headroom, 'pct': pct,
        })

        status = '✅' if headroom > 0 else '❌ OVERFLOW'
        print(f"  {uid}: {n_lines:,} lines | {n_tokens:,} tokens | "
              f"{pct:.1f}% of 128K | headroom={headroom:,} {status}")

    # Summary
    toks = [r['n_tokens'] for r in results]
    heads = [r['headroom'] for r in results]
    lines = [r['n_lines'] for r in results]

    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(results)} users)")
    print(f"{'='*70}")
    print(f"  Lines:  min={min(lines):,}  max={max(lines):,}  mean={np.mean(lines):,.0f}")
    print(f"  Tokens: min={min(toks):,}  max={max(toks):,}  mean={np.mean(toks):,.0f}")
    print(f"  Headroom: min={min(heads):,}  max={max(heads):,}")
    print(f"  Context usage: {np.mean(toks)/MODEL_CONTEXT*100:.1f}% avg, "
          f"{max(toks)/MODEL_CONTEXT*100:.1f}% max")

    if min(heads) < 0:
        print(f"\n  ❌ WARNING: {sum(1 for h in heads if h < 0)} users EXCEED context window!")
        print(f"     Consider filtering to 5:00-23:45 instead of 0:00-23:45")
    elif min(heads) < 10_000:
        print(f"\n  ⚠️  Tight: min headroom is only {min(heads):,} tokens")
        print(f"     Consider filtering to 5:00-23:45 to save ~20% tokens")
    else:
        print(f"\n  ✅ All prompts fit comfortably within 128K context")
        print(f"     Headroom allows for generation + safety margin")

    # Optimization suggestion
    midnight_slots = 20  # 0:00-4:45 at 15-min = 20 slots/day
    avg_days = np.mean(lines) / 96  # rough
    wasted = midnight_slots * avg_days * 14  # ~14 tokens/line
    print(f"\n  💡 Optimization: filtering 0:00-4:45 (all 'at home') would save "
          f"~{wasted:,.0f} tokens/prompt ({wasted/np.mean(toks)*100:.1f}%)")


if __name__ == "__main__":
    main()