# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Per-user trajectory steering — Phase 2c (Redesigned Mar 30).

# Per job: 1 iter × 1 layer × 1 user × 1 cfg × both methods.

# Method 1 (Old, with coefficients):
#   V = mean(H_emp) - mean(H_unemp)  [RAW, unnormalized]
#   coeffs = [-10, -5, -1, 0, 5]
#   h += coeff × V

# Method 2 (New, Matteo — no coefficients):
#   V_emp   = mean(H_emp) - mean(H_all)   [RAW]
#   V_unemp = mean(H_unemp) - mean(H_all) [RAW]
#   3 conditions: neutral / +V_emp / +V_unemp

# Vectors from vectors_nl/ (RAW unnormalized, shared with NL steering).

# Per job: (5 old + 3 new) × 30 reps = 240 gen × ~1.5 min = ~6h

# Output: outputs_{iter}/steering/per_user_v8/{uid[:32]}_cfg{cfg}_L{layer}.json

# Usage:
#     python steering/run_steering_per_user_v8.py --user <uid> --iter v7 --layer 26 --cfg 5
# """
# import sys, os, argparse, time, json
# import numpy as np
# import torch

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from config import DATA_DIR, MODEL_REGISTRY
# from utils import load_model, wrap_with_chat_template

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OLD_COEFFS = [-10, -5, -1, 0, 5]
# DEFAULT_N_REPS = 30

# GEN_KWARGS = dict(do_sample=True, temperature=1.0,
#                   max_new_tokens=4096, repetition_penalty=1.0)


# # ══════════════════════════════════════════════════════════════════════
# # Hooks
# # ══════════════════════════════════════════════════════════════════════

# def make_hook_raw(steer_vec):
#     def hook(module, input, output):
#         h = output[0] if isinstance(output, tuple) else output
#         h[:, :, :] += steer_vec.to(h.device, h.dtype)
#         if isinstance(output, tuple):
#             return (h,) + output[1:]
#         return h
#     return hook


# def make_hook_coeff(steer_vec, coeff):
#     def hook(module, input, output):
#         h = output[0] if isinstance(output, tuple) else output
#         h[:, :, :] += coeff * steer_vec.to(h.device, h.dtype)
#         if isinstance(output, tuple):
#             return (h,) + output[1:]
#         return h
#     return hook


# # ══════════════════════════════════════════════════════════════════════
# # Parse trajectory
# # ══════════════════════════════════════════════════════════════════════

# def parse_trajectory(text, home_geo, work_geo):
#     lines = text.strip().split('\n')
#     geo_ids = []
#     for line in lines:
#         parts = [p.strip() for p in line.strip().split(',')]
#         if len(parts) >= 2:
#             geo = parts[-1].strip()
#             if geo and len(geo) > 3:
#                 geo_ids.append(geo)
#     if not geo_ids:
#         return {'n_slots': 0, 'frac_home': 0, 'frac_work': 0,
#                 'n_unique': 0, 'geo_ids': [], 'valid': False}
#     n = len(geo_ids)
#     return {
#         'n_slots': n,
#         'frac_home': sum(1 for g in geo_ids if g == home_geo) / n,
#         'frac_work': sum(1 for g in geo_ids if g == work_geo) / n,
#         'n_unique': len(set(geo_ids)),
#         'geo_ids': geo_ids,
#         'valid': n >= 10,
#     }


# def generate_one(model, inputs, n_input, tokenizer, layer, hook_fn, home_geo, work_geo):
#     handle = None
#     if hook_fn is not None:
#         handle = model.model.layers[layer].register_forward_hook(hook_fn)
#     with torch.no_grad():
#         output = model.generate(**inputs, **GEN_KWARGS)
#     if handle is not None:
#         handle.remove()
#     gen_text = tokenizer.decode(output[0, n_input:], skip_special_tokens=True)
#     parsed = parse_trajectory(gen_text, home_geo, work_geo)
#     result = {
#         'frac_home': parsed['frac_home'],
#         'frac_work': parsed['frac_work'],
#         'n_unique': parsed['n_unique'],
#         'n_slots': parsed['n_slots'],
#         'valid': parsed['valid'],
#         'generated_text': gen_text,
#         'geo_ids': parsed['geo_ids'],
#     }
#     del output
#     return result


# def run_condition(model, inputs, n_input, tokenizer, layer, hook_fn,
#                   home_geo, work_geo, label, n_reps, count, total, t0):
#     """Run n_reps generations for one condition."""
#     reps = []
#     for rep in range(n_reps):
#         r = generate_one(model, inputs, n_input, tokenizer,
#                          layer, hook_fn, home_geo, work_geo)
#         r['rep'] = rep
#         reps.append(r)
#         count[0] += 1
#         if (rep + 1) % 10 == 0 or rep == 0:
#             elapsed = (time.time() - t0) / 60
#             eta = (total - count[0]) * elapsed / max(count[0], 1)
#             valid = [x for x in reps if x['valid']]
#             mw = np.mean([x['frac_work'] for x in valid]) if valid else 0
#             mh = np.mean([x['frac_home'] for x in valid]) if valid else 0
#             print(f"  {label} {rep+1}/{n_reps}: "
#                   f"home={mh:.2f} work={mw:.2f} | "
#                   f"{elapsed:.0f}m ETA {eta:.0f}m")
#     torch.cuda.empty_cache()
#     valid = [x for x in reps if x['valid']]
#     return {
#         'label': label, 'layer': layer,
#         'n_valid': len(valid), 'n_total': n_reps,
#         'mean_frac_home': float(np.mean([x['frac_home'] for x in valid])) if valid else 0,
#         'mean_frac_work': float(np.mean([x['frac_work'] for x in valid])) if valid else 0,
#         'std_frac_home': float(np.std([x['frac_home'] for x in valid])) if valid else 0,
#         'std_frac_work': float(np.std([x['frac_work'] for x in valid])) if valid else 0,
#         'reps': reps,
#     }


# # ══════════════════════════════════════════════════════════════════════
# # Main
# # ══════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--user', required=True)
#     parser.add_argument('--iter', required=True, choices=['v7', 'v8'])
#     parser.add_argument('--layer', required=True, type=int)
#     parser.add_argument('--cfg', required=True, type=int)
#     parser.add_argument('--model', default='12b')
#     parser.add_argument('--n-reps', type=int, default=DEFAULT_N_REPS)
#     args = parser.parse_args()

#     uid = args.user
#     layer = args.layer
#     cfg_id = args.cfg

#     out_dir = os.path.join(BASE_DIR, f'outputs_{args.iter}', 'steering', 'per_user_v8')
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"{uid[:32]}_cfg{cfg_id}_L{layer}.json")
#     if os.path.exists(out_path):
#         print(f"Already done: {out_path}"); return

#     t0 = time.time()

#     # ── Load prompt ──
#     prompt_path = os.path.join(DATA_DIR, 'steering_prompts_v8', f"{uid[:32]}.json")
#     if not os.path.exists(prompt_path):
#         print(f"ERROR: {prompt_path} not found"); return
#     with open(prompt_path) as f:
#         prebuilt = json.load(f)
#     home_geo = prebuilt['meta']['home_geo_id']
#     work_geo = prebuilt['meta']['work_geo_id']

#     # ── Load vectors ──
#     vec_dir = os.path.join(BASE_DIR, f'outputs_{args.iter}', 'steering', 'vectors_nl')
#     vpath = os.path.join(vec_dir, f"exp2_cfg{cfg_id}_nl_vectors.npz")
#     if not os.path.exists(vpath):
#         print(f"ERROR: {vpath} not found"); return
#     vdata = np.load(vpath)
#     v_old = torch.from_numpy(vdata['v_old'][layer + 1])          # decoder layer index
#     v_emp = torch.from_numpy(vdata['v_emp_new'][layer + 1])
#     v_unemp = torch.from_numpy(vdata['v_unemp_new'][layer + 1])

#     n_old = len(OLD_COEFFS) * args.n_reps
#     n_new = 3 * args.n_reps
#     total = n_old + n_new

#     print(f"{'='*60}")
#     print(f"Trajectory Steering: {uid[:16]} | {args.iter} | cfg{cfg_id} | L{layer}")
#     print(f"  home={home_geo}, work={work_geo}")
#     print(f"  Old: {len(OLD_COEFFS)} coeffs × {args.n_reps} = {n_old} gen")
#     print(f"  New: 3 conds × {args.n_reps} = {n_new} gen")
#     print(f"  Total: {total} gen × ~1.5 min = ~{total * 1.5 / 60:.1f}h")
#     print(f"  ||V_old||={vdata['norms_old'][layer+1]:.1f}  "
#           f"||V_emp||={vdata['norms_emp'][layer+1]:.1f}  "
#           f"||V_unemp||={vdata['norms_unemp'][layer+1]:.1f}")
#     print(f"{'='*60}")

#     # ── Load model ──
#     model, tokenizer = load_model(args.model)
#     device = next(model.parameters()).device
#     print(f"  Model loaded in {time.time()-t0:.0f}s")

#     # ── Tokenize ──
#     wrapped = wrap_with_chat_template(tokenizer, prebuilt['prompt'], False)
#     inputs = tokenizer(wrapped, return_tensors="pt", truncation=True,
#                        max_length=128000)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     n_input = inputs['input_ids'].shape[1]
#     print(f"  Prompt: {n_input} tokens")

#     # ── Run conditions ──
#     all_results = []
#     count = [0]  # mutable for nested function

#     # Old method
#     for coeff in OLD_COEFFS:
#         label = f"old_c{coeff:+.0f}"
#         hook_fn = None if coeff == 0 else make_hook_coeff(v_old, coeff)
#         result = run_condition(model, inputs, n_input, tokenizer, layer,
#                                hook_fn, home_geo, work_geo, label,
#                                args.n_reps, count, total, t0)
#         result['method'] = 'old'
#         result['coeff'] = coeff
#         result['cfg_id'] = cfg_id
#         all_results.append(result)
#         print(f"  {label} DONE: {result['n_valid']}/{args.n_reps} valid | "
#               f"home={result['mean_frac_home']:.3f} work={result['mean_frac_work']:.3f}")

#     # New method
#     for cond_name, hook_fn in [('neutral', None),
#                                 ('emp', make_hook_raw(v_emp)),
#                                 ('unemp', make_hook_raw(v_unemp))]:
#         label = f"new_{cond_name}"
#         result = run_condition(model, inputs, n_input, tokenizer, layer,
#                                hook_fn, home_geo, work_geo, label,
#                                args.n_reps, count, total, t0)
#         result['method'] = 'new'
#         result['condition'] = cond_name
#         result['cfg_id'] = cfg_id
#         all_results.append(result)
#         print(f"  {label} DONE: {result['n_valid']}/{args.n_reps} valid | "
#               f"home={result['mean_frac_home']:.3f} work={result['mean_frac_work']:.3f}")

#     # ── Save ──
#     save_data = {
#         'meta': {
#             'user': uid, 'home_geo_id': home_geo, 'work_geo_id': work_geo,
#             'iter': args.iter, 'cfg_id': cfg_id, 'layer': layer,
#             'old_coeffs': OLD_COEFFS, 'n_reps': args.n_reps,
#             'gen_kwargs': {k: v for k, v in GEN_KWARGS.items()},
#             'vectors_unnormalized': True,
#         },
#         'results': all_results,
#     }
#     with open(out_path, 'w') as f:
#         json.dump(save_data, f)

#     fsize = os.path.getsize(out_path) / (1024 * 1024)
#     elapsed = (time.time() - t0) / 60
#     print(f"\n{'='*60}")
#     print(f"Done: {uid[:16]} {args.iter} cfg{cfg_id} L{layer}")
#     print(f"  {count[0]} gen in {elapsed:.0f}m ({elapsed/count[0]:.1f}m/gen)")
#     print(f"  Saved: {out_path} ({fsize:.1f}MB)")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-user trajectory steering — Phase 2c (Redesigned Mar 30).

Per job: 1 iter × 1 layer × 1 user × 1 cfg × both methods.

Method 1 (Old, with coefficients):
  V = mean(H_emp) - mean(H_unemp)  [RAW, unnormalized]
  coeffs = [-10, -5, -1, 0, 5]
  h += coeff × V

Method 2 (New, Matteo — no coefficients):
  V_emp   = mean(H_emp) - mean(H_all)   [RAW]
  V_unemp = mean(H_unemp) - mean(H_all) [RAW]
  3 conditions: neutral / +V_emp / +V_unemp

Vectors from vectors_nl/ (RAW unnormalized, shared with NL steering).

Per job: (5 old + 3 new) × 30 reps = 240 gen × ~1.5 min = ~6h

Output: outputs_{iter}/steering/per_user_v8/{uid[:32]}_cfg{cfg}_L{layer}.json

Usage:
    python steering/run_steering_per_user_v8.py --user <uid> --iter v7 --layer 26 --cfg 5
"""
import sys, os, argparse, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, MODEL_REGISTRY
from utils import load_model, wrap_with_chat_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLD_COEFFS = [-10, -5, -1, 0, 5]
DEFAULT_N_REPS = 30

GEN_KWARGS = dict(do_sample=True, temperature=1.0,
                  max_new_tokens=1500, repetition_penalty=1.0)


# ══════════════════════════════════════════════════════════════════════
# Hooks
# ══════════════════════════════════════════════════════════════════════

def make_hook_raw(steer_vec):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def make_hook_coeff(steer_vec, coeff):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, :, :] += coeff * steer_vec.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


# ══════════════════════════════════════════════════════════════════════
# Parse trajectory
# ══════════════════════════════════════════════════════════════════════

def parse_trajectory(text, home_geo, work_geo):
    """Parse generated trajectory, extracting valid geo_ids.
    
    Expects lines like: "Tuesday, Mar 03, 2020, 05:15, 250056441013"
    Validates: geo_id must be numeric and >= 9 digits (census tract format).
    The prompt seeds the first line, so generated text may start with just a geo_id.
    """
    import re
    lines = text.strip().split('\n')
    geo_ids = []
    geo_pattern = re.compile(r'\d{9,15}')  # 9-15 digit census tract ID
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like code, explanation, or markdown
        if any(kw in line.lower() for kw in ['```', 'def ', 'import ', 'print(',
               'answer:', 'disclaimer', 'note:', 'http', '**', 'python']):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        # Try to find a geo_id in the last field
        if len(parts) >= 2:
            candidate = parts[-1].strip()
            if geo_pattern.fullmatch(candidate):
                geo_ids.append(candidate)
                continue
        
        # Fallback: first line might be just a geo_id (completing seeded line)
        stripped = line.strip()
        if geo_pattern.fullmatch(stripped):
            geo_ids.append(stripped)
    
    if not geo_ids:
        return {'n_slots': 0, 'frac_home': 0, 'frac_work': 0,
                'n_unique': 0, 'geo_ids': [], 'valid': False}
    n = len(geo_ids)
    return {
        'n_slots': n,
        'frac_home': sum(1 for g in geo_ids if g == home_geo) / n,
        'frac_work': sum(1 for g in geo_ids if g == work_geo) / n,
        'n_unique': len(set(geo_ids)),
        'geo_ids': geo_ids,
        'valid': n >= 10,
    }


def generate_one(model, inputs, n_input, tokenizer, layer, hook_fn, home_geo, work_geo):
    handle = None
    if hook_fn is not None:
        handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        output = model.generate(**inputs, **GEN_KWARGS)
    if handle is not None:
        handle.remove()
    gen_text = tokenizer.decode(output[0, n_input:], skip_special_tokens=True)
    parsed = parse_trajectory(gen_text, home_geo, work_geo)
    result = {
        'frac_home': parsed['frac_home'],
        'frac_work': parsed['frac_work'],
        'n_unique': parsed['n_unique'],
        'n_slots': parsed['n_slots'],
        'valid': parsed['valid'],
        'generated_text': gen_text,
        'geo_ids': parsed['geo_ids'],
    }
    del output
    return result


def run_condition(model, inputs, n_input, tokenizer, layer, hook_fn,
                  home_geo, work_geo, label, n_reps, count, total, t0):
    """Run n_reps generations for one condition."""
    reps = []
    for rep in range(n_reps):
        r = generate_one(model, inputs, n_input, tokenizer,
                         layer, hook_fn, home_geo, work_geo)
        r['rep'] = rep
        reps.append(r)
        count[0] += 1
        if (rep + 1) % 10 == 0 or rep == 0:
            elapsed = (time.time() - t0) / 60
            eta = (total - count[0]) * elapsed / max(count[0], 1)
            valid = [x for x in reps if x['valid']]
            mw = np.mean([x['frac_work'] for x in valid]) if valid else 0
            mh = np.mean([x['frac_home'] for x in valid]) if valid else 0
            print(f"  {label} {rep+1}/{n_reps}: "
                  f"home={mh:.2f} work={mw:.2f} | "
                  f"{elapsed:.0f}m ETA {eta:.0f}m")
    torch.cuda.empty_cache()
    valid = [x for x in reps if x['valid']]
    return {
        'label': label, 'layer': layer,
        'n_valid': len(valid), 'n_total': n_reps,
        'mean_frac_home': float(np.mean([x['frac_home'] for x in valid])) if valid else 0,
        'mean_frac_work': float(np.mean([x['frac_work'] for x in valid])) if valid else 0,
        'std_frac_home': float(np.std([x['frac_home'] for x in valid])) if valid else 0,
        'std_frac_work': float(np.std([x['frac_work'] for x in valid])) if valid else 0,
        'reps': reps,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True)
    parser.add_argument('--iter', required=True, choices=['v7', 'v8'])
    parser.add_argument('--layer', required=True, type=int)
    parser.add_argument('--cfg', required=True, type=int)
    parser.add_argument('--model', default='12b')
    parser.add_argument('--n-reps', type=int, default=DEFAULT_N_REPS)
    args = parser.parse_args()

    uid = args.user
    layer = args.layer
    cfg_id = args.cfg

    out_dir = os.path.join(BASE_DIR, f'outputs_{args.iter}', 'steering', 'per_user_v8')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uid[:32]}_cfg{cfg_id}_L{layer}.json")
    if os.path.exists(out_path):
        print(f"Already done: {out_path}"); return

    t0 = time.time()

    # ── Load prompt ──
    prompt_path = os.path.join(DATA_DIR, 'steering_prompts_v8', f"{uid[:32]}.json")
    if not os.path.exists(prompt_path):
        print(f"ERROR: {prompt_path} not found"); return
    with open(prompt_path) as f:
        prebuilt = json.load(f)
    home_geo = prebuilt['meta']['home_geo_id']
    work_geo = prebuilt['meta']['work_geo_id']

    # ── Load vectors ──
    vec_dir = os.path.join(BASE_DIR, f'outputs_{args.iter}', 'steering', 'vectors_nl')
    vpath = os.path.join(vec_dir, f"exp2_cfg{cfg_id}_nl_vectors.npz")
    if not os.path.exists(vpath):
        print(f"ERROR: {vpath} not found"); return
    vdata = np.load(vpath)
    v_old = torch.from_numpy(vdata['v_old'][layer + 1])          # decoder layer index
    v_emp = torch.from_numpy(vdata['v_emp_new'][layer + 1])
    v_unemp = torch.from_numpy(vdata['v_unemp_new'][layer + 1])

    n_old = len(OLD_COEFFS) * args.n_reps
    n_new = 3 * args.n_reps
    total = n_old + n_new

    print(f"{'='*60}")
    print(f"Trajectory Steering: {uid[:16]} | {args.iter} | cfg{cfg_id} | L{layer}")
    print(f"  home={home_geo}, work={work_geo}")
    print(f"  Old: {len(OLD_COEFFS)} coeffs × {args.n_reps} = {n_old} gen")
    print(f"  New: 3 conds × {args.n_reps} = {n_new} gen")
    print(f"  Total: {total} gen × ~1.5 min = ~{total * 1.5 / 60:.1f}h")
    print(f"  ||V_old||={vdata['norms_old'][layer+1]:.1f}  "
          f"||V_emp||={vdata['norms_emp'][layer+1]:.1f}  "
          f"||V_unemp||={vdata['norms_unemp'][layer+1]:.1f}")
    print(f"{'='*60}")

    # ── Load model ──
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    # ── Tokenize ──
    wrapped = wrap_with_chat_template(tokenizer, prebuilt['prompt'], False)
    inputs = tokenizer(wrapped, return_tensors="pt", truncation=True,
                       max_length=128000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_input = inputs['input_ids'].shape[1]
    print(f"  Prompt: {n_input} tokens")

    # ── Run conditions ──
    all_results = []
    count = [0]  # mutable for nested function

    # Old method
    for coeff in OLD_COEFFS:
        label = f"old_c{coeff:+.0f}"
        hook_fn = None if coeff == 0 else make_hook_coeff(v_old, coeff)
        result = run_condition(model, inputs, n_input, tokenizer, layer,
                               hook_fn, home_geo, work_geo, label,
                               args.n_reps, count, total, t0)
        result['method'] = 'old'
        result['coeff'] = coeff
        result['cfg_id'] = cfg_id
        all_results.append(result)
        print(f"  {label} DONE: {result['n_valid']}/{args.n_reps} valid | "
              f"home={result['mean_frac_home']:.3f} work={result['mean_frac_work']:.3f}")

    # New method
    for cond_name, hook_fn in [('neutral', None),
                                ('emp', make_hook_raw(v_emp)),
                                ('unemp', make_hook_raw(v_unemp))]:
        label = f"new_{cond_name}"
        result = run_condition(model, inputs, n_input, tokenizer, layer,
                               hook_fn, home_geo, work_geo, label,
                               args.n_reps, count, total, t0)
        result['method'] = 'new'
        result['condition'] = cond_name
        result['cfg_id'] = cfg_id
        all_results.append(result)
        print(f"  {label} DONE: {result['n_valid']}/{args.n_reps} valid | "
              f"home={result['mean_frac_home']:.3f} work={result['mean_frac_work']:.3f}")

    # ── Save ──
    save_data = {
        'meta': {
            'user': uid, 'home_geo_id': home_geo, 'work_geo_id': work_geo,
            'iter': args.iter, 'cfg_id': cfg_id, 'layer': layer,
            'old_coeffs': OLD_COEFFS, 'n_reps': args.n_reps,
            'gen_kwargs': {k: v for k, v in GEN_KWARGS.items()},
            'vectors_unnormalized': True,
        },
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f)

    fsize = os.path.getsize(out_path) / (1024 * 1024)
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Done: {uid[:16]} {args.iter} cfg{cfg_id} L{layer}")
    print(f"  {count[0]} gen in {elapsed:.0f}m ({elapsed/count[0]:.1f}m/gen)")
    print(f"  Saved: {out_path} ({fsize:.1f}MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()