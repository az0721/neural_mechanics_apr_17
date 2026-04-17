#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive chat with Gemma 4 31B — no history, single-turn Q&A.

Runs on srun interactive GPU session. Each message is independent.
Shows both thinking (CoT) and final answer separately.

Usage:
    # First get a GPU session:
    srun --partition=gpu --gres=gpu:h200:1 --time=2:00:00 --mem=64G --cpus-per-task=4 --pty bash
    conda activate gemma4_env
    cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
    python chat_gemma4.py

Commands:
    Type your message and press Enter to chat.
    /think on    — enable thinking mode (default)
    /think off   — disable thinking mode
    /temp 0.7    — set temperature (default 1.0)
    /maxtok 2048 — set max new tokens (default 4096)
    /quit or /q  — exit
    Ctrl+C       — also exits
"""
import os, sys, time, re
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'gemma-4-31B-it-unsloth-bnb-4bit')

# ── Settings ──
SETTINGS = {
    'thinking': True,
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 64,
    'max_new_tokens': 4096,
}


def load_model():
    print("=" * 60)
    print("  Gemma 4 31B Interactive Chat")
    print("  Loading model (takes ~45s)...")
    print("=" * 60)

    proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype='auto', device_map='auto',
        local_files_only=True,
    ).eval()

    print(f"  Model loaded! Vocab: {tok.vocab_size}")
    print(f"  Device: {next(model.parameters()).device}")
    print()
    return proc, tok, model


def print_settings():
    print(f"  thinking={SETTINGS['thinking']}  temp={SETTINGS['temperature']}  "
          f"top_p={SETTINGS['top_p']}  top_k={SETTINGS['top_k']}  "
          f"max_tokens={SETTINGS['max_new_tokens']}")


def print_help():
    print()
    print("  Commands:")
    print("    /think on|off  — toggle thinking mode")
    print("    /temp <float>  — set temperature")
    print("    /maxtok <int>  — set max new tokens")
    print("    /quit or /q    — exit")
    print("    /help or /?    — show this")
    print()
    print_settings()
    print()


def chat(proc, tok, model, user_input):
    messages = [{'role': 'user', 'content': user_input}]

    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=SETTINGS['thinking'],
    )

    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_input = inputs['input_ids'].shape[1]

    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            do_sample=SETTINGS['temperature'] > 0,
            temperature=SETTINGS['temperature'],
            top_p=SETTINGS['top_p'],
            top_k=SETTINGS['top_k'],
            max_new_tokens=SETTINGS['max_new_tokens'],
        )
    elapsed = time.time() - t0
    n_gen = output.shape[1] - n_input

    raw = tok.decode(output[0, n_input:], skip_special_tokens=False)

    # Parse thinking and answer
    thinking_text = None
    answer_text = raw

    # Try to split by channel tags (Gemma 4 thinking format)
    # Format: <|channel>thought\n...<channel|>answer
    channel_match = re.search(
        r'<\|channel>thought\n(.*?)<channel\|>(.*)',
        raw, re.DOTALL)
    if channel_match:
        thinking_text = channel_match.group(1).strip()
        answer_text = channel_match.group(2).strip()
    else:
        # Try <think>...</think> format
        think_match = re.search(
            r'<think>(.*?)</think>(.*)',
            raw, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            answer_text = think_match.group(2).strip()

    # Clean up special tokens from answer
    for tag in ['<turn|>', '<|turn>', '<eos>', '</s>', '<end_of_turn>']:
        answer_text = answer_text.replace(tag, '').strip()
        if thinking_text:
            thinking_text = thinking_text.replace(tag, '').strip()

    # Display
    print()
    if thinking_text:
        print("┌─ THINKING " + "─" * 48)
        # Truncate very long thinking for display
        if len(thinking_text) > 2000:
            print(f"│ {thinking_text[:1000]}")
            print(f"│ ... [{len(thinking_text)} chars, showing first/last 1000] ...")
            print(f"│ {thinking_text[-1000:]}")
        else:
            for line in thinking_text.split('\n'):
                print(f"│ {line}")
        print("└" + "─" * 59)
        print()

    print("┌─ ANSWER " + "─" * 50)
    for line in answer_text.split('\n'):
        print(f"│ {line}")
    print("└" + "─" * 59)
    print(f"  [{n_gen} tokens, {elapsed:.1f}s, {n_gen/elapsed:.1f} tok/s]")
    print()

    del output
    torch.cuda.empty_cache()


def main():
    proc, tok, model = load_model()
    print_help()

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ('/quit', '/q', '/exit'):
            print("Bye!")
            break

        if user_input.lower() in ('/help', '/?'):
            print_help()
            continue

        if user_input.lower().startswith('/think'):
            parts = user_input.split()
            if len(parts) >= 2:
                SETTINGS['thinking'] = parts[1].lower() in ('on', 'true', '1', 'yes')
            print(f"  thinking = {SETTINGS['thinking']}")
            continue

        if user_input.lower().startswith('/temp'):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    SETTINGS['temperature'] = float(parts[1])
                except ValueError:
                    print("  Invalid temperature")
                    continue
            print(f"  temperature = {SETTINGS['temperature']}")
            continue

        if user_input.lower().startswith('/maxtok'):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    SETTINGS['max_new_tokens'] = int(parts[1])
                except ValueError:
                    print("  Invalid max_tokens")
                    continue
            print(f"  max_new_tokens = {SETTINGS['max_new_tokens']}")
            continue

        if user_input.startswith('/'):
            print("  Unknown command. Type /help")
            continue

        chat(proc, tok, model, user_input)


if __name__ == "__main__":
    main()
