#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NL-based steering — logits only, no generation.

Two methods per (exp, cfg, iter):
  Old:  V = H_emp - H_unemp (raw), coeffs [-10,-5,-1,0,1,5,10]
  New:  3 conditions: neutral / +V_emp / +V_unemp (raw, no coefficient)

All decoder layers. Measures logit distribution at answer token position.

Output: {model_dir}/outputs_{iter}/steering/nl_results/{exp}_cfg{id}_nl.json

Usage:
    python steering/run_steering_nl.py --model gemma4_31b --iter v7 --exp exp2 --cfg 6
    python steering/run_steering_nl.py --model 12b --iter v8 --exp exp1a --cfg 4
"""
import sys, os, argparse, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_REGISTRY, get_iter_output_dir
from utils import load_model, get_model_layers
from steering.prompts_nl import ALL_QUESTIONS, MCQ_TOKENS, DIGIT_TOKENS

OLD_COEFFS = [-10, -5, -1, 0, 1, 5, 10]


# ══════════════════════════════════════════════════════════════════════
# Steering Hook
# ══════════════════════════════════════════════════════════════════════

def make_hook(steer_vec):
    """Add raw vector to all token positions."""
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def make_hook_coeff(steer_vec, coeff):
    """Add coeff * vector to all token positions."""
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += coeff * steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


# ══════════════════════════════════════════════════════════════════════
# Logit Extraction
# ══════════════════════════════════════════════════════════════════════

def get_token_probs(logits, tokenizer, token_list):
    """Get softmax probabilities for specific tokens from last-position logits."""
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    # Get the base tokenizer (handles both AutoProcessor and AutoTokenizer)
    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    result = {}
    for tok_str in token_list:
        best_p = 0.0
        for candidate in [tok_str, f" {tok_str}"]:
            ids = tok.encode(candidate, add_special_tokens=False)
            if len(ids) >= 1:
                p = probs[ids[0]].item()
                best_p = max(best_p, p)
        result[tok_str] = best_p
    return result


def run_single(model, tokenizer, inputs, decoder_layers, layer, steer_vec,
               coeff, question):
    """Run one forward pass with steering and extract logits."""
    handle = None
    if steer_vec is not None:
        if coeff is not None:
            if coeff != 0:
                handle = decoder_layers[layer].register_forward_hook(
                    make_hook_coeff(steer_vec, coeff))
        else:
            handle = decoder_layers[layer].register_forward_hook(
                make_hook(steer_vec))

    with torch.no_grad():
        output = model(**inputs)

    if handle is not None:
        handle.remove()

    logits = output.logits

    if question['type'] == 'mcq':
        probs = get_token_probs(logits, tokenizer, MCQ_TOKENS)
    else:
        probs = get_token_probs(logits, tokenizer, DIGIT_TOKENS)

    del output
    return probs


# ══════════════════════════════════════════════════════════════════════
# Tokenization — model-aware
# ══════════════════════════════════════════════════════════════════════

def tokenize_prompt(tokenizer, prompt_text, model_key, device):
    """Tokenize a prompt, handling both Gemma 3 and Gemma 4."""
    arch = MODEL_REGISTRY[model_key]['arch']
    if arch == 'gemma4':
        # Gemma 4: AutoProcessor, returns dict directly
        tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
    else:
        # Gemma 3: AutoTokenizer, apply_chat_template returns string
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=8192)
    return {k: v.to(device) for k, v in inputs.items()}


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', required=True, choices=['v7', 'v8'])
    parser.add_argument('--exp', required=True, choices=['exp2', 'exp1a'])
    parser.add_argument('--cfg', required=True, type=int)
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    # ── Model-aware paths ──
    out_base = get_iter_output_dir(args.model, args.iter)
    vec_dir = os.path.join(out_base, 'steering', 'vectors_nl')
    out_dir = os.path.join(out_base, 'steering', 'nl_results')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.exp}_cfg{args.cfg}_nl.json")
    if os.path.exists(out_path):
        print(f"Already done: {out_path}"); return

    t0 = time.time()

    # ── Load vectors ──
    vec_path = os.path.join(vec_dir, f"{args.exp}_cfg{args.cfg}_nl_vectors.npz")
    if not os.path.exists(vec_path):
        print(f"ERROR: {vec_path} not found")
        print(f"Run: python steering/compute_vectors_nl.py --model {args.model} "
              f"--iter {args.iter} --exp {args.exp} --cfgs {args.cfg}")
        return
    vdata = np.load(vec_path)
    v_old = torch.from_numpy(vdata['v_old'])
    v_emp_new = torch.from_numpy(vdata['v_emp_new'])
    v_unemp_new = torch.from_numpy(vdata['v_unemp_new'])
    n_layers_vec = v_old.shape[0]

    print(f"{'='*60}")
    print(f"NL Steering: {args.iter} / {args.exp} / cfg{args.cfg}")
    print(f"  Model: {args.model} ({MODEL_REGISTRY[args.model]['tag']})")
    print(f"  Vectors: {vec_path}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    # ── Load model ──
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # Get decoder layers (handles both Gemma 3 and 4)
    _, decoder_layers = get_model_layers(model, args.model)
    n_decoder_layers = len(decoder_layers)
    print(f"  Model loaded in {time.time()-t0:.0f}s, {n_decoder_layers} decoder layers")

    # ── Get questions ──
    questions = ALL_QUESTIONS[args.exp]

    # ── Tokenize all prompts ──
    prompt_inputs = {}
    for q in questions:
        inputs = tokenize_prompt(tokenizer, q['prompt'], args.model, device)
        prompt_inputs[q['id']] = inputs
        print(f"  {q['id']}: {inputs['input_ids'].shape[1]} tokens | {q['description']}")

    # ── Run all conditions ──
    all_results = []
    count = 0

    n_old = len(questions) * n_decoder_layers * len(OLD_COEFFS)
    n_new = len(questions) * n_decoder_layers * 3
    total = n_old + n_new
    print(f"\n  Total conditions: {total} ({n_old} old + {n_new} new)")
    print(f"  Est time: {total * 0.3 / 60:.1f} min\n")

    for q in questions:
        inputs = prompt_inputs[q['id']]
        q_results = {'question': q, 'old_method': [], 'new_method': []}

        # ── Old method: V_old with coefficients ──
        for layer in range(n_decoder_layers):
            sv = v_old[layer + 1]  # +1: vectors[0]=embedding, vectors[1]=decoder L0
            for coeff in OLD_COEFFS:
                if coeff == 0:
                    probs = run_single(model, tokenizer, inputs, decoder_layers,
                                       layer, None, None, q)
                else:
                    probs = run_single(model, tokenizer, inputs, decoder_layers,
                                       layer, sv, coeff, q)

                q_results['old_method'].append({
                    'layer': layer, 'coeff': coeff, 'probs': probs
                })
                count += 1

            if (layer + 1) % 15 == 0:
                elapsed = time.time() - t0
                eta = (total - count) * elapsed / max(count, 1)
                print(f"  {q['id']} old L{layer}: {count}/{total} "
                      f"({elapsed/60:.0f}m, ETA {eta/60:.0f}m)")

        # ── New method: 3 conditions per layer ──
        for layer in range(n_decoder_layers):
            ve = v_emp_new[layer + 1]
            vu = v_unemp_new[layer + 1]

            probs_n = run_single(model, tokenizer, inputs, decoder_layers,
                                  layer, None, None, q)
            probs_e = run_single(model, tokenizer, inputs, decoder_layers,
                                  layer, ve, None, q)
            probs_u = run_single(model, tokenizer, inputs, decoder_layers,
                                  layer, vu, None, q)

            q_results['new_method'].append({
                'layer': layer,
                'neutral': probs_n,
                'employed': probs_e,
                'unemployed': probs_u,
            })
            count += 3

            if (layer + 1) % 15 == 0:
                print(f"  {q['id']} new L{layer}: {count}/{total}")

        torch.cuda.empty_cache()
        all_results.append(q_results)

    # ── Save ──
    save_data = {
        'meta': {
            'iter': args.iter, 'exp': args.exp, 'cfg': args.cfg,
            'model': args.model,
            'model_tag': MODEL_REGISTRY[args.model]['tag'],
            'old_coeffs': OLD_COEFFS,
            'new_conditions': ['neutral', 'employed', 'unemployed'],
            'n_decoder_layers': n_decoder_layers,
            'n_questions': len(questions),
            'total_passes': count,
            'elapsed_min': (time.time() - t0) / 60,
        },
        'results': all_results,
    }

    with open(out_path, 'w') as f:
        json.dump(save_data, f)

    print(f"\n{'='*60}")
    print(f"Done: {args.iter}/{args.exp}/cfg{args.cfg}")
    print(f"  {count} forward passes in {(time.time()-t0)/60:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()