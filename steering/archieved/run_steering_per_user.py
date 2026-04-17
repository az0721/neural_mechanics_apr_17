#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-user steering — Phase 2b (logits only).
Loads pre-built prompts from data/steering_prompts/ (no data loading on GPU).

Pipeline:
  1. Load prebuilt prompt JSON (~0s)
  2. Load model + tokenizer (~60s)
  3. Load steering vectors (~1s)
  4. Wrap with chat template + tokenize all 8 prompts (~10s)
  5. Run logits: 49 layers × 9 coeffs × 8 prompts = 3528 forward passes (~7.5h)

Usage:
    python steering/run_steering_per_user.py --user <uid>
"""
import sys, os, argparse, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, OUTPUT_DIR
from utils import load_model, wrap_with_chat_template
from steering.prompts import STEERING_PROMPTS

# ══════════════════════════════════════════════════════════════════════
# Config — Phase 2b
# ══════════════════════════════════════════════════════════════════════
STEER_DIR = os.path.join(OUTPUT_DIR, 'steering')
VECTORS_DIR = os.path.join(STEER_DIR, 'vectors')
PERUSER_DIR = os.path.join(STEER_DIR, 'per_user_v2')
PROMPT_DIR = os.path.join(DATA_DIR, 'steering_prompts')
os.makedirs(PERUSER_DIR, exist_ok=True)

DEFAULT_CFGS = [5, 6]
DEFAULT_EXP = 'exp2'

LAYERS = list(range(0, 48))
COEFFS = [-50, -20, -10, -5, 0, 5, 10, 20, 50]


# ══════════════════════════════════════════════════════════════════════
# Steering Hook
# ══════════════════════════════════════════════════════════════════════

def make_steering_hook(steer_vec, coeff):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += coeff * steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def get_target_probs(logits, tokenizer, target_names):
    probs = torch.softmax(logits, dim=-1)
    result = {}
    for name in target_names:
        for candidate in [name, f" {name}"]:
            tids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(tids) >= 1:
                result[name] = probs[tids[-1]].item()
                break
    return result


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True)
    parser.add_argument('--model', default='12b')
    parser.add_argument('--exp', default=DEFAULT_EXP)
    parser.add_argument('--cfgs', nargs='+', type=int, default=DEFAULT_CFGS)
    args = parser.parse_args()

    uid = args.user
    json_path = os.path.join(PERUSER_DIR, f"{uid[:32]}.json")
    if os.path.exists(json_path):
        print(f"Already done: {json_path}")
        return

    t0 = time.time()

    # ── Step 1: Load prebuilt prompts (instant) ──
    prebuilt_path = os.path.join(PROMPT_DIR, f"{uid[:32]}.json")
    if not os.path.exists(prebuilt_path):
        print(f"ERROR: prebuilt not found: {prebuilt_path}")
        print(f"Run: python steering/prebuild_steering_prompts.py")
        return

    with open(prebuilt_path) as f:
        prebuilt = json.load(f)

    meta = prebuilt['meta']
    is_emp = meta['is_employed']
    tdate = meta['date']
    raw_prompts = prebuilt['prompts']  # {str((cfg, qkey)): text}

    n_prompts = len(raw_prompts)
    n_layers = len(LAYERS)
    n_coeffs = len(COEFFS)
    total = n_prompts * n_layers * n_coeffs

    print(f"{'='*60}")
    print(f"Steering Phase 2b: user={uid[:16]}...")
    print(f"  Employed: {is_emp}, date: {tdate}")
    print(f"  {n_prompts} prompts × {n_layers} layers × {n_coeffs} coeffs = {total}")
    print(f"  Est: {total * 8 / 3600:.1f}h")
    print(f"{'='*60}")

    # ── Step 2: Load model + tokenizer ──
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device
    t_model = time.time()
    print(f"  Model loaded in {t_model - t0:.0f}s")

    # ── Step 3: Load steering vectors ──
    vectors = {}
    for cfg_id in args.cfgs:
        vpath = os.path.join(VECTORS_DIR, f"{args.exp}_cfg{cfg_id}_vectors.npz")
        vdata = np.load(vpath)
        vectors[cfg_id] = torch.from_numpy(vdata['vectors'])
        print(f"  Vectors cfg{cfg_id}: {vdata['vectors'].shape}")

    # ── Step 4: Wrap + tokenize all prompts upfront ──
    prompts = {}  # (cfg_id, qkey) → {input_ids, attention_mask} on device
    for key_str, raw_text in raw_prompts.items():
        # Parse key: "(5, 'behavioral')" → (5, 'behavioral')
        cfg_id, qkey = eval(key_str)
        if cfg_id not in args.cfgs:
            continue
        wrapped = wrap_with_chat_template(tokenizer, raw_text, False)
        inputs = tokenizer(wrapped, return_tensors="pt", truncation=True,
                           max_length=128000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompts[(cfg_id, qkey)] = inputs

    t_tok = time.time()
    sample_key = next(iter(prompts))
    print(f"  Tokenized {len(prompts)} prompts in {t_tok - t_model:.0f}s "
          f"({prompts[sample_key]['input_ids'].shape[1]} tokens each)")

    # ── Step 5: Logits for ALL conditions ──
    total_conditions = len(prompts) * n_layers * n_coeffs
    print(f"\n── Logits: {total_conditions} conditions ──")
    logits_results = {}
    count = 0

    for (cfg_id, qkey), inputs in prompts.items():
        targets_tok = STEERING_PROMPTS[qkey]['targets']

        for layer in LAYERS:
            steer_vec = vectors[cfg_id][layer + 1]

            for coeff in COEFFS:
                if coeff != 0:
                    hook_fn = make_steering_hook(steer_vec, coeff)
                    handle = model.model.layers[layer].register_forward_hook(hook_fn)

                with torch.no_grad():
                    out = model(**inputs)
                logits_last = out.logits[0, -1, :]

                if coeff != 0:
                    handle.remove()

                key = (cfg_id, qkey, layer, coeff)
                logits_results[key] = get_target_probs(logits_last, tokenizer, targets_tok)

                del out
                count += 1

            torch.cuda.empty_cache()

        # Progress per (cfg, qkey) — 8 total blocks
        elapsed = (time.time() - t0) / 60
        rate = count / max(elapsed, 0.01)
        eta = (total_conditions - count) / max(rate, 0.01)
        print(f"  cfg{cfg_id}/{qkey}: {count}/{total_conditions} "
              f"({count/total_conditions*100:.0f}%) | "
              f"{elapsed:.0f}m elapsed | ETA {eta:.0f}m")

    # ── Save ──
    save_data = {
        'meta': {
            'user': uid, 'is_employed': is_emp, 'date': tdate,
            'cfgs': args.cfgs, 'layers': LAYERS, 'coeffs': COEFFS,
            'prompts': list(STEERING_PROMPTS.keys()),
            'phase': '2b',
        },
        'logits': {str(k): v for k, v in logits_results.items()},
    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f)

    total_time = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Done: {uid[:16]} | {total_time:.1f}min | {count} conditions")
    print(f"Saved: {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()