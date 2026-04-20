#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive chat with Gemma 4 31B — no history, single-turn Q&A.

Usage:
    srun --partition=gpu --gres=gpu:h200:1 --time=2:00:00 --mem=64G --cpus-per-task=4 --pty bash
    conda activate gemma4_env
    cd /scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
    python chat_gemma4.py

Input:
    Type your message and press Enter twice (empty line) to send.
    This supports multi-line paste — just paste and hit Enter twice.

Commands (single line, no need for double-Enter):
    /think on|off  — toggle thinking mode
    /temp 0.7      — set temperature
    /maxtok 2048   — set max new tokens
    /quit or /q    — exit
    Ctrl+C         — also exits
"""
import os, sys, time, re
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'gemma-4-31B-it-unsloth-bnb-4bit')

SETTINGS = {
    'thinking': False,    # default OFF for NL steering experiments
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 64,
    'max_new_tokens': 4096,
}


def load_model():
    print("=" * 60)
    print("  Gemma 4 31B Interactive Chat")
    print("  Loading model (takes ~2 min)...")
    print("=" * 60)
    proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype='auto', device_map='auto',
        local_files_only=True).eval()
    print(f"  Model loaded! Vocab: {tok.vocab_size}")
    print(f"  Device: {next(model.parameters()).device}")
    return proc, tok, model


def print_settings():
    print(f"  thinking={SETTINGS['thinking']}  temp={SETTINGS['temperature']}  "
          f"top_p={SETTINGS['top_p']}  top_k={SETTINGS['top_k']}  "
          f"max_tokens={SETTINGS['max_new_tokens']}")


def print_help():
    print()
    print("  Type your message, then press Enter TWICE (empty line) to send.")
    print("  Multi-line paste is supported — just paste and hit Enter twice.")
    print()
    print("  Commands (single line):")
    print("    /think on|off  — toggle thinking mode")
    print("    /temp <float>  — set temperature")
    print("    /maxtok <int>  — set max new tokens")
    print("    /quit or /q    — exit")
    print("    /help or /?    — show this")
    print()
    print_settings()
    print()


def read_multiline():
    """Read multi-line input. Empty line (just Enter) submits."""
    lines = []
    while True:
        try:
            line = input("" if lines else "You > ")
        except (KeyboardInterrupt, EOFError):
            return None
        # If this is the first line and it's a command, return immediately
        if not lines and line.strip().startswith('/'):
            return line.strip()
        # Empty line = submit
        if line.strip() == '' and lines:
            break
        lines.append(line)
    return '\n'.join(lines)


def chat(proc, tok, model, user_input):
    messages = [{'role': 'user', 'content': user_input}]
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=SETTINGS['thinking'])

    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_input = inputs['input_ids'].shape[1]
    print(f"  [input: {n_input} tokens, thinking={SETTINGS['thinking']}]")

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

    if SETTINGS['thinking']:
        # Only parse thinking blocks when thinking mode is ON
        channel_match = re.search(
            r'<\|channel>thought\n(.*?)<channel\|>(.*)', raw, re.DOTALL)
        if channel_match:
            thinking_text = channel_match.group(1).strip()
            answer_text = channel_match.group(2).strip()
        else:
            think_match = re.search(
                r'<think>(.*?)</think>(.*)', raw, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
                answer_text = think_match.group(2).strip()

    # Clean special tokens
    for tag in ['<turn|>', '<|turn>', '<eos>', '</s>', '<end_of_turn>',
                '<|channel>thought', '<channel|>']:
        answer_text = answer_text.replace(tag, '').strip()
        if thinking_text:
            thinking_text = thinking_text.replace(tag, '').strip()

    print()
    if thinking_text:
        print("+" + "-" * 59 + " THINKING")
        display = thinking_text
        if len(display) > 2000:
            display = display[:1000] + \
                f'\n... [{len(thinking_text)} chars] ...\n' + display[-1000:]
        for line in display.split('\n'):
            print(f"| {line}")
        print("+" + "-" * 59)
        print()

    print("=" * 60 + " ANSWER")
    for line in answer_text.split('\n'):
        print(f"  {line}")
    print("=" * 60)
    print(f"  [{n_gen} tokens, {elapsed:.1f}s, {n_gen/elapsed:.1f} tok/s]")
    print()

    del output
    torch.cuda.empty_cache()


def main():
    proc, tok, model = load_model()
    print_help()

    while True:
        user_input = read_multiline()

        if user_input is None:
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()

        if cmd in ('/quit', '/q', '/exit'):
            print("Bye!")
            break

        if cmd in ('/help', '/?'):
            print_help()
            continue

        if cmd.startswith('/think'):
            parts = cmd.split()
            if len(parts) >= 2:
                SETTINGS['thinking'] = parts[1] in ('on', 'true', '1', 'yes')
            print(f"  >> thinking = {SETTINGS['thinking']}")
            continue

        if cmd.startswith('/temp'):
            parts = cmd.split()
            if len(parts) >= 2:
                try: SETTINGS['temperature'] = float(parts[1])
                except ValueError: print("  Invalid temperature"); continue
            print(f"  >> temperature = {SETTINGS['temperature']}")
            continue

        if cmd.startswith('/maxtok'):
            parts = cmd.split()
            if len(parts) >= 2:
                try: SETTINGS['max_new_tokens'] = int(parts[1])
                except ValueError: print("  Invalid max_tokens"); continue
            print(f"  >> max_new_tokens = {SETTINGS['max_new_tokens']}")
            continue

        if cmd.startswith('/'):
            print("  Unknown command. Type /help")
            continue

        chat(proc, tok, model, user_input)


if __name__ == "__main__":
    main()