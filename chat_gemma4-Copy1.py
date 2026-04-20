#!/usr/bin/env python3
"""
Test coefficient safety for finegrain steering.
For each (layer, coeff), check if sum(P_codes) stays near 1.0.
If it drops → model is off-manifold → that coeff is too large for that layer.

Run: python3 test_coeff_safety.py [--vec_path PATH] [--exp exp2] [--cfg 4]
"""
import torch, numpy as np, argparse, os, sys, time

# ── Add project to path ──
sys.path.insert(0, '/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7')
from config import MODEL_REGISTRY, get_iter_output_dir

MODEL_KEY = 'gemma4_31b'
MODEL_PATH = 'gemma-4-31B-it-unsloth-bnb-4bit'

# ── Coefficients to test ──
TEST_COEFFS = [-7, -5, -3, -1, 0, 1, 3, 5, 7]

# ── Layers to test (every 5th + last few which are most fragile) ──
TEST_LAYERS = list(range(0, 60, 5)) + [52, 54, 56, 57, 58, 59]
TEST_LAYERS = sorted(set(TEST_LAYERS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='exp2')
    parser.add_argument('--cfg', type=int, default=4)
    parser.add_argument('--iter', default='v8')
    args = parser.parse_args()

    # ── Load vectors ──
    out_base = get_iter_output_dir(MODEL_KEY, args.iter)
    vec_dir = os.path.join(out_base, 'steering', 'nl_vectors')
    vec_path = os.path.join(vec_dir, f"{args.exp}_cfg{args.cfg}_nl_vectors.npz")
    print(f'Loading vectors: {vec_path}')
    vdata = np.load(vec_path)
    v_old = torch.from_numpy(vdata['v_old'])       # (n_layers+1, hidden_dim)
    v_emp = torch.from_numpy(vdata['v_emp_new'])    # (n_layers, hidden_dim)
    v_unemp = torch.from_numpy(vdata['v_unemp_new'])
    print(f'  v_old shape: {v_old.shape}')
    print(f'  v_emp shape: {v_emp.shape}')

    # ── Compute ||V|| and ||h|| stats ──
    print(f'\n{"="*80}')
    print(f'VECTOR NORMS (old method V = mean(H_emp) - mean(H_unemp))')
    print(f'{"="*80}')
    print(f'{"Layer":>6s}  {"||V_old||":>10s}  {"||V_emp||":>10s}  {"||V_unemp||":>12s}')
    print('-' * 50)
    for l in TEST_LAYERS:
        vo = np.linalg.norm(v_old[l+1].numpy())
        ve = np.linalg.norm(v_emp[l].numpy())
        vu = np.linalg.norm(v_unemp[l].numpy())
        print(f'  L{l:>3d}  {vo:>10.2f}  {ve:>10.2f}  {vu:>12.2f}')

    # ── Load model ──
    print(f'\nLoading model...')
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    proc = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype='auto', device_map='auto', local_files_only=True).eval()
    print('Model loaded.')

    # Get decoder layers
    if hasattr(model, 'language_model'):
        decoder_layers = model.language_model.model.layers
    else:
        decoder_layers = model.model.layers
    n_dec = len(decoder_layers)
    print(f'Decoder layers: {n_dec}')

    # ── Build code pool ──
    SPLIT = {'gq','lq','qf','qg','qj','tq','vj','wz','yq','yv'}
    pair_to_id = {}
    for a in 'abcdefghijklmnopqrstuvwxyz':
        for b in 'abcdefghijklmnopqrstuvwxyz':
            p = a + b
            if p in SPLIT or 'q' in p: continue
            ids = tok.encode(p, add_special_tokens=False)
            if len(ids) == 1: pair_to_id[p] = ids[0]
    all_codes = sorted(pair_to_id.keys())
    pct_codes = all_codes[:51]  # 0-100% in 2%
    pct_labels = [f'{i}%' for i in range(0, 102, 2)]

    # ── Build test prompt (FG_E1: P(awake<7AM Tue)) ──
    pct_opts = ', '.join(f'{c}={l}' for c, l in zip(pct_codes, pct_labels))
    prompt = (
        'What is the probability that a typical adult is already '
        'awake before 7:00 AM on a Tuesday? '
        f'Pick one: {pct_opts}. '
        'Reply with ONLY the two-letter code: '
    )
    messages = [{'role': 'user', 'content': prompt}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f'Prompt tokens: {inputs["input_ids"].shape[1]}')

    # ── Hook function ──
    def make_hook(vec, coeff):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h[:, :, :] += coeff * vec.to(h.device, h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    # ── Run baseline (no steering) ──
    print(f'\nRunning baseline...')
    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    baseline = {c: probs[pair_to_id[c]].item() for c in pct_codes}
    base_sum = sum(baseline.values())
    base_ranked = sorted(baseline.items(), key=lambda x: -x[1])
    base_peak = base_ranked[0]
    base_label = pct_labels[pct_codes.index(base_peak[0])]
    print(f'  Baseline sum={base_sum:.6f} peak={base_label}({base_peak[1]:.4f})')
    del out

    # ── Test all (layer, coeff) combinations ──
    print(f'\n{"="*80}')
    print(f'COEFFICIENT SAFETY TEST — Old Method (V_old × coeff)')
    print(f'Prompt: FG_E1 P(awake<7AM Tue)')
    print(f'{"="*80}')

    # Header
    coeff_str = ''.join(f'{c:>8d}' for c in TEST_COEFFS)
    print(f'{"Layer":>6s}  {"||V||":>7s}  {coeff_str}')
    print('-' * (16 + 8 * len(TEST_COEFFS)))

    collapse_map = {}  # (layer, coeff) → sum_P

    for l in TEST_LAYERS:
        if l >= n_dec: continue
        sv = v_old[l + 1]
        v_norm = sv.norm().item()
        row = f'  L{l:>3d}  {v_norm:>7.1f}  '

        for c in TEST_COEFFS:
            if c == 0:
                sp = base_sum
            else:
                handle = decoder_layers[l].register_forward_hook(make_hook(sv, c))
                with torch.inference_mode():
                    out = model(**inputs)
                handle.remove()
                logits = out.logits[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                cp = {code: probs[pair_to_id[code]].item() for code in pct_codes}
                sp = sum(cp.values())
                del out
                torch.cuda.empty_cache()

            collapse_map[(l, c)] = sp

            if sp > 0.95:
                row += f'{sp:>8.4f}'
            elif sp > 0.5:
                row += f'{"!"+f"{sp:.3f}":>8s}'
            else:
                row += f'{"XX"+f"{sp:.2f}":>8s}'

        print(row)

    # ── Also test NEW method (V_emp / V_unemp, no coeff) ──
    print(f'\n{"="*80}')
    print(f'NEW METHOD SAFETY (V_emp and V_unemp, raw, no coefficient)')
    print(f'{"="*80}')
    print(f'{"Layer":>6s}  {"||V_emp||":>10s}  {"sum_emp":>10s}  {"||V_unemp||":>12s}  {"sum_unemp":>12s}')
    print('-' * 60)

    for l in TEST_LAYERS:
        if l >= n_dec: continue
        ve = v_emp[l]; vu = v_unemp[l]
        ve_n = ve.norm().item(); vu_n = vu.norm().item()

        # +V_emp
        handle = decoder_layers[l].register_forward_hook(make_hook(ve, 1.0))
        with torch.inference_mode():
            out = model(**inputs)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        se = sum(probs[pair_to_id[c]].item() for c in pct_codes)
        del out

        # +V_unemp
        handle = decoder_layers[l].register_forward_hook(make_hook(vu, 1.0))
        with torch.inference_mode():
            out = model(**inputs)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        su = sum(probs[pair_to_id[c]].item() for c in pct_codes)
        del out

        torch.cuda.empty_cache()
        se_flag = '' if se > 0.95 else ' !' if se > 0.5 else ' XX'
        su_flag = '' if su > 0.95 else ' !' if su > 0.5 else ' XX'
        print(f'  L{l:>3d}  {ve_n:>10.2f}  {se:>10.4f}{se_flag}  {vu_n:>12.2f}  {su:>12.4f}{su_flag}')

    # ── Summary ──
    print(f'\n{"="*80}')
    print(f'SUMMARY: Safe coefficient ranges per layer group')
    print(f'{"="*80}')

    groups = [(0, 14), (15, 29), (30, 39), (40, 49), (50, 54), (55, 59)]
    for g_start, g_end in groups:
        layers_in = [l for l in TEST_LAYERS if g_start <= l <= g_end]
        if not layers_in: continue

        # Find max safe coeff (sum > 0.95 for ALL layers in group)
        max_safe = 0
        for c in sorted(TEST_COEFFS):
            if c <= 0: continue
            all_safe = all(collapse_map.get((l, c), 0) > 0.95
                          and collapse_map.get((l, -c), 0) > 0.95
                          for l in layers_in)
            if all_safe:
                max_safe = c

        print(f'  L{g_start:>2d}-L{g_end:>2d}: max safe |coeff| = ±{max_safe}')

    print(f'\nLegend: value=sum(P_codes), !=0.5-0.95(borderline), XX=<0.5(collapsed)')
    print('DONE')


if __name__ == '__main__':
    main()
