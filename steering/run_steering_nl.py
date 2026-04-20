#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NL-based steering — logits only, no generation.  (v2)

Two methods per (exp, cfg, iter):
  Old:  V = H_emp - H_unemp (raw), coeffs [-5,-3,-1,0,1,3,5]
  New:  3 conditions: neutral / +V_emp / +V_unemp (raw, no coefficient)

Supports 3 question types:
  'mcq'       → extract P(A), P(B), P(C)
  'numeric'   → extract P(0)..P(9)
  'finegrain' → extract P(code) for all codes in q['fg_codes']

Output: {model_dir}/outputs_{iter}/steering/nl_results/{exp}_cfg{id}_nl.json

Changes from v1:
  - OLD_COEFFS: [-10,-5,-1,0,1,5,10] → [-5,-3,-1,0,1,3,5]
    (G4-31B: all 61 layers safe at c=±5; ±10 caused off-manifold at high layers)
  - Added 'finegrain' question type: two-letter codes (51-84 options)
  - G4 chat template: enable_thinking=False (prevents thinking token logits)
  - Added --force flag to overwrite existing results

Usage:
    python steering/run_steering_nl.py --model gemma4_31b --iter v8 --exp exp2 --cfg 4
"""
import sys, os, argparse, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_REGISTRY, get_iter_output_dir
from utils import load_model, get_model_layers
from steering.prompts_nl import ALL_QUESTIONS

# Backward compat: import MCQ/DIGIT tokens if they exist
try:
    from steering.prompts_nl import MCQ_TOKENS, DIGIT_TOKENS
except ImportError:
    MCQ_TOKENS = ['A', 'B', 'C']
    DIGIT_TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

OLD_COEFFS = [-5, -3, -1, 0, 1, 3, 5]


# ══════════════════════════════════════════════════════════════════════
# Steering Hooks
# ══════════════════════════════════════════════════════════════════════

def make_hook(steer_vec):
    """Add raw vector to all token positions (new method)."""
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def make_hook_coeff(steer_vec, coeff):
    """Add coeff * vector to all token positions (old method)."""
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += coeff * steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


# ══════════════════════════════════════════════════════════════════════
# Token ID Mapping
# ══════════════════════════════════════════════════════════════════════

def build_fg_token_map(tokenizer, fg_codes):
    """Map each two-letter code to its token ID. Verify single-token.
    Returns: dict {code_str: token_id}
    Raises: ValueError if any code is not a single token.
    """
    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    code_to_id = {}
    bad = []
    for code in fg_codes:
        ids = tok.encode(code, add_special_tokens=False)
        if len(ids) == 1:
            code_to_id[code] = ids[0]
        else:
            bad.append((code, len(ids)))
    if bad:
        print(f"  WARNING: {len(bad)} codes are NOT single-token: {bad[:5]}...")
    return code_to_id


def get_mcq_digit_token_ids(tokenizer, token_list):
    """Get token IDs for MCQ (A/B/C) or digit (0-9) tokens.
    Tries with and without leading space, takes the higher-prob variant.
    Returns: dict {token_str: token_id}
    """
    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    result = {}
    for tok_str in token_list:
        best_id = None
        for candidate in [tok_str, f" {tok_str}"]:
            ids = tok.encode(candidate, add_special_tokens=False)
            if len(ids) >= 1:
                if best_id is None:
                    best_id = ids[0]
        result[tok_str] = best_id
    return result


# ══════════════════════════════════════════════════════════════════════
# Logit Extraction
# ══════════════════════════════════════════════════════════════════════

def extract_probs(logits, token_id_map):
    """Extract softmax probabilities for a set of token IDs.
    Args:
        logits: model output logits (batch=1, seq_len, vocab)
        token_id_map: dict {label: token_id}
    Returns: dict {label: probability}
    """
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    return {label: probs[tid].item() for label, tid in token_id_map.items()
            if tid is not None}


def run_single(model, inputs, decoder_layers, layer, steer_vec, coeff):
    """Run one forward pass with optional steering. Returns raw logits."""
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
    # Don't delete output here — caller extracts probs from logits
    return logits


# ══════════════════════════════════════════════════════════════════════
# Tokenization — model-aware
# ══════════════════════════════════════════════════════════════════════

def tokenize_prompt(tokenizer, prompt_text, model_key, device):
    """Tokenize a prompt, handling both Gemma 3 and Gemma 4."""
    arch = MODEL_REGISTRY[model_key]['arch']
    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    messages = [{"role": "user", "content": prompt_text}]

    if arch == 'gemma4':
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False)
    else:
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
    parser.add_argument('--model', default='gemma4_31b',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing results')
    args = parser.parse_args()

    # ── Paths ──
    out_base = get_iter_output_dir(args.model, args.iter)
    vec_dir = os.path.join(out_base, 'steering', 'vectors_nl')
    out_dir = os.path.join(out_base, 'steering', 'nl_results')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.exp}_cfg{args.cfg}_nl.json")
    if os.path.exists(out_path) and not args.force:
        print(f"Already done: {out_path} (use --force to overwrite)")
        return

    t0 = time.time()

    # ── Load vectors ──
    vec_path = os.path.join(vec_dir,
                            f"{args.exp}_cfg{args.cfg}_nl_vectors.npz")
    if not os.path.exists(vec_path):
        print(f"ERROR: {vec_path} not found")
        print(f"Run: python steering/compute_vectors_nl.py --model {args.model} "
              f"--iter {args.iter} --exp {args.exp} --cfgs {args.cfg}")
        return
    vdata = np.load(vec_path)
    v_old = torch.from_numpy(vdata['v_old'])           # (n_layers+1, hdim)
    v_emp_new = torch.from_numpy(vdata['v_emp_new'])    # (n_layers+1, hdim)
    v_unemp_new = torch.from_numpy(vdata['v_unemp_new'])

    print(f"{'='*60}")
    print(f"NL Steering v2: {args.iter} / {args.exp} / cfg{args.cfg}")
    print(f"  Model: {args.model} ({MODEL_REGISTRY[args.model]['tag']})")
    print(f"  Coeffs: {OLD_COEFFS}")
    print(f"  Vectors: {vec_path}")
    print(f"  v_old shape: {list(v_old.shape)}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    # ── Load model ──
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device
    _, decoder_layers = get_model_layers(model, args.model)
    n_decoder_layers = len(decoder_layers)
    n_vec_layers = v_old.shape[0]
    print(f"  Model: {n_decoder_layers} decoder layers, "
          f"vectors: {n_vec_layers} layers")
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # ── Get questions ──
    questions = ALL_QUESTIONS[args.exp]

    # ── Build token ID maps per question ──
    token_maps = {}
    for q in questions:
        if q['type'] == 'finegrain':
            tmap = build_fg_token_map(tokenizer, q['fg_codes'])
            n_tokens = len(tmap)
        elif q['type'] == 'mcq':
            tmap = get_mcq_digit_token_ids(tokenizer, MCQ_TOKENS)
            n_tokens = len(tmap)
        else:  # numeric
            tmap = get_mcq_digit_token_ids(tokenizer, DIGIT_TOKENS)
            n_tokens = len(tmap)
        token_maps[q['id']] = tmap

    # ── Tokenize all prompts ──
    prompt_inputs = {}
    for q in questions:
        inputs = tokenize_prompt(tokenizer, q['prompt'], args.model, device)
        prompt_inputs[q['id']] = inputs
        n_toks = inputs['input_ids'].shape[1]
        n_answer = len(token_maps[q['id']])
        print(f"  {q['id']}: {n_toks} input tokens, "
              f"{n_answer} answer tokens | {q['description']}")

    # ── Determine layer range ──
    # vectors[0] = embedding layer, vectors[1] = decoder layer 0, etc.
    # We steer decoder layers 0..n_decoder_layers-1
    # Vector index for decoder layer L = L + 1
    # But if n_vec_layers == n_decoder_layers (no embedding offset),
    # vector index = L
    if n_vec_layers == n_decoder_layers + 1:
        vec_offset = 1  # vectors[0]=embedding, vectors[1]=decoder L0
    elif n_vec_layers == n_decoder_layers:
        vec_offset = 0  # vectors[0]=decoder L0
    else:
        print(f"  WARNING: vector layers ({n_vec_layers}) != "
              f"decoder layers ({n_decoder_layers}) ± 1. Using offset=1.")
        vec_offset = 1

    print(f"  Vector offset: {vec_offset} "
          f"(v[{vec_offset}] → decoder L0)")

    # ── Count total forward passes ──
    n_old = len(questions) * n_decoder_layers * len(OLD_COEFFS)
    n_new = len(questions) * n_decoder_layers * 3
    total = n_old + n_new
    print(f"\n  Total passes: {total} "
          f"({n_old} old + {n_new} new)")
    print(f"  Est time: {total * 0.1 / 60:.1f} min\n")

    # ── Run all conditions ──
    all_results = []
    count = 0

    for q in questions:
        inputs = prompt_inputs[q['id']]
        tmap = token_maps[q['id']]

        # Store question metadata (without fg_codes/fg_labels lists
        # which are large — store id and type, reconstruct from prompts_nl)
        q_meta = {k: v for k, v in q.items()
                  if k not in ('fg_codes', 'fg_labels', 'prompt')}
        q_meta['prompt_preview'] = q['prompt'][:100] + '...'
        q_meta['n_answer_tokens'] = len(tmap)

        q_results = {'question': q_meta, 'old_method': [], 'new_method': []}

        # ── Old method: V_old with coefficients ──
        for layer in range(n_decoder_layers):
            vi = layer + vec_offset
            if vi >= n_vec_layers:
                continue
            sv = v_old[vi]

            for coeff in OLD_COEFFS:
                logits = run_single(model, inputs, decoder_layers,
                                    layer, None if coeff == 0 else sv,
                                    None if coeff == 0 else coeff)
                probs = extract_probs(logits, tmap)
                del logits

                q_results['old_method'].append({
                    'layer': layer, 'coeff': coeff, 'probs': probs
                })
                count += 1

            if (layer + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = (total - count) * elapsed / max(count, 1)
                print(f"  {q['id']} old L{layer}: {count}/{total} "
                      f"({elapsed/60:.1f}m, ETA {eta/60:.1f}m)")

        # ── New method: 3 conditions per layer ──
        for layer in range(n_decoder_layers):
            vi = layer + vec_offset
            if vi >= n_vec_layers:
                continue
            ve = v_emp_new[vi]
            vu = v_unemp_new[vi]

            logits_n = run_single(model, inputs, decoder_layers,
                                   layer, None, None)
            probs_n = extract_probs(logits_n, tmap)
            del logits_n

            logits_e = run_single(model, inputs, decoder_layers,
                                   layer, ve, None)
            probs_e = extract_probs(logits_e, tmap)
            del logits_e

            logits_u = run_single(model, inputs, decoder_layers,
                                   layer, vu, None)
            probs_u = extract_probs(logits_u, tmap)
            del logits_u

            q_results['new_method'].append({
                'layer': layer,
                'neutral': probs_n,
                'employed': probs_e,
                'unemployed': probs_u,
            })
            count += 3

            if (layer + 1) % 20 == 0:
                print(f"  {q['id']} new L{layer}: {count}/{total}")

        torch.cuda.empty_cache()
        all_results.append(q_results)
        print(f"  {q['id']} done. "
              f"old={len(q_results['old_method'])} "
              f"new={len(q_results['new_method'])}")

    # ── Sanity check: verify sum(P) ≈ 1 for baseline ──
    print(f"\n  Sanity check (baseline sum at L0):")
    for qr in all_results:
        qid = qr['question']['id']
        for e in qr['old_method']:
            if e['layer'] == 0 and e['coeff'] == 0:
                s = sum(e['probs'].values())
                flag = 'OK' if s > 0.9 else 'LOW!'
                print(f"    {qid}: sum(P)={s:.4f} {flag}")
                break

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
            'vec_offset': vec_offset,
            'version': 'v2_finegrain',
        },
        'results': all_results,
    }

    with open(out_path, 'w') as f:
        json.dump(save_data, f)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Done: {args.iter}/{args.exp}/cfg{args.cfg}")
    print(f"  {count} forward passes in {elapsed:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"  Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()