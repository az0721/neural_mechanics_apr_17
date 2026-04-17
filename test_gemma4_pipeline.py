#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test gemma-4-31B-it-unsloth-bnb-4bit using the SAME loading/extraction/generation
logic as the updated utils.py pipeline. Validates on stops-format prompt.

Tests:
  0. Short sanity check (2+2)
  1. Stops prompt — full pipeline: hooks + generation + parse

Usage:
    # Submit to H200
    sbatch test_gemma4_pipeline.sbatch
    # Or interactively on V100
    srun --partition=gpu --gres=gpu:v100-sxm2:1 --time=1:00:00 --mem=64G --pty bash
    conda activate gemma4_env && python -u test_gemma4_pipeline.py
"""
import os
import sys
import time
import json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

STOPS_PROMPT_FILE = "example_prompt_stops.txt"


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import numpy as np

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram:.1f} GB")

    # ── Import from our pipeline (same code that will run at scale) ──
    from config import (
        MODEL_REGISTRY, GENERATION_CONFIG_GEMMA4,
        MAX_NEW_TOKENS_COT_GEMMA4, MAX_NEW_TOKENS_NO_COT,
        SYSTEM_PROMPT_COT_GEMMA4, get_system_prompt_cot,
    )
    from utils import (
        load_model, get_model_layers, get_model_config,
        wrap_with_chat_template, parse_answer, process_sample,
    )

    MODEL_KEY = 'gemma4_31b'
    print(f"Model key: {MODEL_KEY}")
    print(f"Model path: {MODEL_REGISTRY[MODEL_KEY]['hf_name']}")
    print(f"Use stops: {MODEL_REGISTRY[MODEL_KEY].get('use_stops', False)}")

    # ── Load model (same as pipeline) ──
    print("\n" + "=" * 60)
    print("Loading model via utils.load_model()")
    print("=" * 60)
    t0 = time.time()
    model, processor = load_model(MODEL_KEY)
    device = next(model.parameters()).device
    print(f"Loaded in {time.time()-t0:.0f}s")
    print(f"Device: {device}")
    print(f"GPU mem (weights): {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    # ── Verify layer access (same as pipeline) ──
    embed, layers = get_model_layers(model, MODEL_KEY)
    n_layers_cfg, hidden_dim = get_model_config(model, MODEL_KEY)
    print(f"Layers: {n_layers_cfg}, Hidden: {hidden_dim}")
    print(f"Embed type: {type(embed).__name__}")
    print(f"Layer 0 type: {type(layers[0]).__name__}")

    # ══════════════════════════════════════════════════════════════
    # TEST 0: Short sanity check
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 0: SHORT SANITY CHECK (2+2)")
    print("=" * 60)
    try:
        short_prompt = "What is 2+2? Answer with just the number."
        hidden, answer = process_sample(
            model, processor, short_prompt, False, device, MODEL_KEY)
        print(f"  Hidden shape: {hidden.shape}")
        print(f"  Tokens: {answer['n_tokens']}")
        print(f"  Top tokens: {answer['top_tokens'][:5]}")
        print(f"  Generated: {answer['generated_text'][:200]}")
        print(f"  Parsed: {answer['parsed_answer']}")
        print(f"  GPU peak: {torch.cuda.max_memory_allocated(0)/1e9:.1f} GB")
        print("  ✅ SHORT: SUCCESS")
    except Exception as e:
        print(f"  ❌ SHORT: FAILED — {e}")
        import traceback; traceback.print_exc()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Stops prompt — NO CoT (use_sys=False)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 1: STOPS PROMPT — NO CoT")
    print("=" * 60)
    stops_prompt = open(STOPS_PROMPT_FILE).read().rstrip()
    stops_tok = len(processor.tokenizer.encode(stops_prompt)) if hasattr(processor, 'tokenizer') else 'N/A'
    print(f"  Prompt: {len(stops_prompt)} chars, ~{stops_tok} tokens")
    try:
        hidden, answer = process_sample(
            model, processor, stops_prompt, False, device, MODEL_KEY)
        print(f"  Hidden shape: {hidden.shape}")
        print(f"  Expected: ({n_layers_cfg + 1}, {hidden_dim})")
        print(f"  Tokens: {answer['n_tokens']}")
        print(f"  Top tokens: {answer['top_tokens'][:5]}")
        print(f"  Top probs: {[f'{p:.4f}' for p in answer['top_probs'][:5]]}")
        print(f"  Parsed answer: {answer['parsed_answer']}")
        print(f"  GPU peak: {torch.cuda.max_memory_allocated(0)/1e9:.1f} GB")
        print(f"\n  Full generated text ({len(answer['generated_text'])} chars):")
        print(f"  {'─'*50}")
        print(f"  {answer['generated_text'][:800]}")
        if len(answer['generated_text']) > 800:
            print(f"  ... [{len(answer['generated_text']) - 800} more chars]")
        print(f"  {'─'*50}")
        print("  ✅ STOPS NO-COT: SUCCESS")
    except Exception as e:
        print(f"  ❌ STOPS NO-COT: FAILED — {e}")
        import traceback; traceback.print_exc()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Stops prompt — WITH CoT (use_sys=True, thinking mode)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 2: STOPS PROMPT — WITH CoT (thinking mode)")
    print("=" * 60)
    print(f"  System prompt preview: {SYSTEM_PROMPT_COT_GEMMA4[:120]}...")
    try:
        hidden_cot, answer_cot = process_sample(
            model, processor, stops_prompt, True, device, MODEL_KEY)
        print(f"  Hidden shape: {hidden_cot.shape}")
        print(f"  Tokens: {answer_cot['n_tokens']}")
        print(f"  Top tokens: {answer_cot['top_tokens'][:5]}")
        print(f"  Parsed answer: {answer_cot['parsed_answer']}")
        print(f"  GPU peak: {torch.cuda.max_memory_allocated(0)/1e9:.1f} GB")
        print(f"\n  Full generated text ({len(answer_cot['generated_text'])} chars):")
        print(f"  {'─'*50}")
        # Print more for CoT since we want to see the thinking block
        text = answer_cot['generated_text']
        print(f"  {text[:1500]}")
        if len(text) > 1500:
            print(f"  ... [{len(text) - 1500} more chars] ...")
            print(f"  {text[-300:]}")
        print(f"  {'─'*50}")
        print("  ✅ STOPS COT: SUCCESS")

        # ── Verify hidden states are different between CoT and non-CoT ──
        if 'hidden' in dir() and hidden is not None:
            cosine = np.dot(hidden[-1], hidden_cot[-1]) / (
                np.linalg.norm(hidden[-1]) * np.linalg.norm(hidden_cot[-1]) + 1e-8)
            print(f"\n  Last-layer cosine(no-CoT, CoT): {cosine:.4f}")
            print(f"  (Should be <1.0 since system prompt differs)")

    except Exception as e:
        print(f"  ❌ STOPS COT: FAILED — {e}")
        import traceback; traceback.print_exc()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Verify parse_answer with Gemma 4 output formats
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 3: parse_answer() unit tests")
    print("=" * 60)
    test_cases = [
        # (input, expected_output, description)
        ("<|channel>thought\nSome reasoning here...<channel|>\n<answer>250214412041</answer>",
         "250214412041", "Gemma4 thinking + answer tags"),
        ("<answer>440070185002</answer>",
         "440070185002", "answer tags only"),
        ("The person will be at 250214412041",
         "250214412041", "plain text fallback"),
        ("<|channel>thought\nAnalysis...<channel|>\nBased on patterns, the location is `250214412041`.",
         "250214412041", "Gemma4 thinking + backtick"),
    ]
    all_pass = True
    for text, expected, desc in test_cases:
        result = parse_answer(text, MODEL_KEY)
        ok = result == expected
        if not ok:
            all_pass = False
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: got='{result}' expected='{expected}'")
    if all_pass:
        print("  ✅ ALL PARSE TESTS PASSED")
    else:
        print("  ⚠️ SOME PARSE TESTS FAILED — check parse_answer()")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  GPU: {gpu_name} ({vram:.0f} GB)")
    print(f"  Model: gemma-4-31B-it-unsloth-bnb-4bit")
    print(f"  Layers: {n_layers_cfg}, Hidden: {hidden_dim}")
    print(f"  Format: stops (start_time, end_time)")
    print("=" * 60)


if __name__ == "__main__":
    main()