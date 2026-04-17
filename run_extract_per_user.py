# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Per-user extraction: hidden states + answers for ALL configs.
# One user = one job. Model loaded once.

# Usage:
#     # Gemma 3 12B (original, both V7+V8)
#     python run_extract_per_user.py --user <uid> --model 12b

#     # Gemma 4 31B (CoT only, stops format, V7+V8)
#     python run_extract_per_user.py --user <uid> --model gemma4_31b --cot-only

#     # Single iteration
#     python run_extract_per_user.py --user <uid> --model gemma4_31b --iter v7 --cot-only
# """
# import sys, os, argparse, time
# import numpy as np

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from config import (
#     EXP_CONFIG, MODEL_REGISTRY,
#     CONCEPT_HINT_V7, CONCEPT_HINT_V8,
#     user_output_path_iter, config_label,
# )
# from utils import (
#     load_data, load_model, build_user_prompts, process_sample,
#     get_model_config,
# )


# def run_one_iteration(model, tokenizer, device, model_key, iter_name,
#                       user_df, uid, is_employed,
#                       forced_targets=None, cot_only=False):
#     """Run extraction for one iteration and save npz."""
#     out_path = user_output_path_iter(uid, model_key, iter_name)

#     if os.path.exists(out_path):
#         print(f"  [{iter_name.upper()}] Already done: {out_path}")
#         return out_path

#     print(f"\n  {'─'*50}")
#     print(f"  {iter_name.upper()} — {'no hint' if iter_name == 'v7' else 'with concept hint'}")
#     if cot_only:
#         print(f"  CoT-only mode: skipping non-CoT configs")
#     print(f"  {'─'*50}")

#     concept_hint = CONCEPT_HINT_V7 if iter_name == 'v7' else CONCEPT_HINT_V8
#     use_stops = MODEL_REGISTRY[model_key].get('use_stops', False)

#     prompts_info = build_user_prompts(
#         user_df, uid, is_employed,
#         forced_targets=forced_targets,
#         concept_hint=concept_hint,
#         use_stops=use_stops,
#         cot_only=cot_only)

#     n_prompts = len(prompts_info)
#     if n_prompts == 0:
#         print(f"  [{iter_name.upper()}] No prompts, skipping")
#         return None

#     exp_cfg_counts = {}
#     for p in prompts_info:
#         key = f"{p['exp_name']}_cfg{p['config_id']}"
#         exp_cfg_counts[key] = exp_cfg_counts.get(key, 0) + 1
#     print(f"  [{iter_name.upper()}] {n_prompts} prompts: "
#           + ", ".join(f"{k}={v}" for k, v in sorted(exp_cfg_counts.items())))

#     n_layers_cfg, hidden_dim = get_model_config(model, model_key)
#     n_layers = n_layers_cfg + 1  # +1 for embedding

#     hidden_states = np.zeros((n_prompts, n_layers, hidden_dim), dtype=np.float32)
#     labels = np.zeros(n_prompts, dtype=np.int32)
#     meta_list = []
#     answers_list = []

#     import torch
#     import gc

#     t0 = time.time()
#     for i, pinfo in enumerate(prompts_info):
#         t_start = time.time()

#         # Default: assume OOM. Overwritten on success.
#         hidden = None
#         answer = {
#             'status': 'oom',
#             'top_tokens': [],
#             'top_probs': [],
#             'generated_text': '',
#             'parsed_answer': '',
#             'n_tokens': 0,
#         }

#         try:
#             hidden, answer = process_sample(
#                 model, tokenizer, pinfo['prompt'], pinfo['use_sys'],
#                 device, model_key)
#             hidden_states[i] = hidden
#             answer['status'] = 'ok'
#         except torch.cuda.OutOfMemoryError:
#             print(f"  [{iter_name.upper()}][{i+1}/{n_prompts}] OOM — skipping")
#         finally:
#             # Always save this sample's data
#             answer['prompt_text'] = pinfo['prompt']
#             labels[i] = pinfo['label']
#             meta_list.append(pinfo['meta'])
#             answers_list.append(answer)

#             # Cleanup every sample
#             if hidden is not None:
#                 del hidden
#             gc.collect()
#             torch.cuda.empty_cache()

#         elapsed = time.time() - t_start
#         if (i + 1) % 5 == 0 or i == 0:
#             pct = (i + 1) / n_prompts * 100
#             eta = elapsed * (n_prompts - i - 1) / 60
#             exp_cfg = f"{pinfo['exp_name']}_cfg{pinfo['config_id']}"
#             print(f"  [{iter_name.upper()}][{i+1:3d}/{n_prompts}] {pct:5.1f}% | "
#                   f"{elapsed:.0f}s | ETA {eta:.0f}min | {exp_cfg}")

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     np.savez(out_path,
#              hidden_states=hidden_states,
#              labels=labels,
#              meta=np.array(meta_list, dtype=object),
#              answers=np.array(answers_list, dtype=object))

#     total_time = (time.time() - t0) / 60
#     n_oom = sum(1 for a in answers_list if a.get('status') == 'oom')
#     print(f"  [{iter_name.upper()}] Done: {n_prompts} prompts | {total_time:.1f} min | "
#           f"OOM: {n_oom}")
#     print(f"  [{iter_name.upper()}] Saved: {out_path}")

#     return out_path


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--user', required=True, help='cuebiq_id')
#     parser.add_argument('--model', default='12b',
#                         choices=list(MODEL_REGISTRY.keys()))
#     parser.add_argument('--iter', nargs='+', default=['v7', 'v8'],
#                         help='Iterations to run (default: v7 v8)')
#     parser.add_argument('--align-v7', action='store_true',
#                         help='Use V7 targets for V8 alignment')
#     parser.add_argument('--cot-only', action='store_true',
#                         help='Only run CoT configs (sys_prompt=True)')
#     args = parser.parse_args()

#     uid = args.user
#     model_key = args.model
#     iters = args.iter

#     all_done = all(
#         os.path.exists(user_output_path_iter(uid, model_key, it))
#         for it in iters)
#     if all_done:
#         print(f"All done for {uid} ({', '.join(iters)})")
#         return

#     import torch
#     t0 = time.time()
#     tag = MODEL_REGISTRY[model_key]['tag']
#     print(f"{'='*60}")
#     print(f"Per-User Extraction: {tag}")
#     print(f"  User:  {uid}")
#     print(f"  Iters: {', '.join(iters)}")
#     print(f"  Model: {model_key}")
#     if args.cot_only:
#         print(f"  Mode:  CoT-only")
#     if MODEL_REGISTRY[model_key].get('use_stops', False):
#         print(f"  Format: stops (start_time, end_time)")
#     print(f"{'='*60}")

#     df = load_data()
#     user_df = df[df['cuebiq_id'] == uid]
#     if len(user_df) == 0:
#         print(f"ERROR: User {uid} not found in data")
#         return
#     is_employed = bool(user_df['is_employed'].iloc[0])
#     print(f"  Employed: {is_employed}")

#     model, tokenizer = load_model(model_key)
#     device = next(model.parameters()).device

#     # Load V7 targets for alignment
#     forced_targets = None
#     if args.align_v7:
#         from utils import load_v7_targets
#         forced_targets = load_v7_targets(uid)
#         if forced_targets:
#             print(f"  V7 targets loaded for alignment")
#         else:
#             print(f"  WARNING: No V7 targets found, using random sampling")

#     for iter_name in iters:
#         run_one_iteration(
#             model, tokenizer, device, model_key, iter_name,
#             user_df, uid, is_employed,
#             forced_targets=forced_targets if iter_name == 'v8' else None,
#             cot_only=args.cot_only)

#     total = (time.time() - t0) / 60
#     print(f"\n{'='*60}")
#     print(f"All iterations done in {total:.1f} min")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-user extraction: hidden states + answers for ALL configs.
One user = one job. Model loaded once.

Usage:
    # Gemma 3 12B (original, both V7+V8)
    python run_extract_per_user.py --user <uid> --model 12b

    # Gemma 4 31B (CoT only, stops format, V7+V8)
    python run_extract_per_user.py --user <uid> --model gemma4_31b --cot-only

    # Single iteration
    python run_extract_per_user.py --user <uid> --model gemma4_31b --iter v7 --cot-only
"""
import sys, os, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EXP_CONFIG, MODEL_REGISTRY,
    CONCEPT_HINT_V7, CONCEPT_HINT_V8,
    user_output_path_iter, config_label,
)
from utils import (
    load_data, load_model, build_user_prompts, process_sample,
    get_model_config,
)


def load_v7_targets_from_path(npz_path):
    """Extract target dates from a V7 per-user npz file."""
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    meta = data['meta']
    targets = {}
    seen = set()
    for m in meta:
        key = (m['exp_name'], m['date'], m['pred_time'])
        if key in seen:
            continue
        seen.add(key)
        exp = m['exp_name']
        targets.setdefault(exp, []).append(
            (m['date'], m['pred_time'], m.get('gt_geo_id')))
    return targets if targets else None


def run_one_iteration(model, tokenizer, device, model_key, iter_name,
                      user_df, uid, is_employed,
                      forced_targets=None, cot_only=False):
    """Run extraction for one iteration and save npz."""
    out_path = user_output_path_iter(uid, model_key, iter_name)

    if os.path.exists(out_path):
        print(f"  [{iter_name.upper()}] Already done: {out_path}")
        return out_path

    print(f"\n  {'─'*50}")
    print(f"  {iter_name.upper()} — {'no hint' if iter_name == 'v7' else 'with concept hint'}")
    if cot_only:
        print(f"  CoT-only mode: skipping non-CoT configs")
    print(f"  {'─'*50}")

    concept_hint = CONCEPT_HINT_V7 if iter_name == 'v7' else CONCEPT_HINT_V8
    use_stops = MODEL_REGISTRY[model_key].get('use_stops', False)

    prompts_info = build_user_prompts(
        user_df, uid, is_employed,
        forced_targets=forced_targets,
        concept_hint=concept_hint,
        use_stops=use_stops,
        cot_only=cot_only)

    n_prompts = len(prompts_info)
    if n_prompts == 0:
        print(f"  [{iter_name.upper()}] No prompts, skipping")
        return None

    exp_cfg_counts = {}
    for p in prompts_info:
        key = f"{p['exp_name']}_cfg{p['config_id']}"
        exp_cfg_counts[key] = exp_cfg_counts.get(key, 0) + 1
    print(f"  [{iter_name.upper()}] {n_prompts} prompts: "
          + ", ".join(f"{k}={v}" for k, v in sorted(exp_cfg_counts.items())))

    n_layers_cfg, hidden_dim = get_model_config(model, model_key)
    n_layers = n_layers_cfg + 1  # +1 for embedding

    hidden_states = np.zeros((n_prompts, n_layers, hidden_dim), dtype=np.float32)
    labels = np.zeros(n_prompts, dtype=np.int32)
    meta_list = []
    answers_list = []

    import torch
    import gc

    t0 = time.time()
    for i, pinfo in enumerate(prompts_info):
        t_start = time.time()

        # Default: assume OOM. Overwritten on success.
        hidden = None
        answer = {
            'status': 'oom',
            'top_tokens': [],
            'top_probs': [],
            'generated_text': '',
            'parsed_answer': '',
            'n_tokens': 0,
        }

        try:
            hidden, answer = process_sample(
                model, tokenizer, pinfo['prompt'], pinfo['use_sys'],
                device, model_key)
            hidden_states[i] = hidden
            answer['status'] = 'ok'
        except torch.cuda.OutOfMemoryError:
            print(f"  [{iter_name.upper()}][{i+1}/{n_prompts}] OOM — skipping")
        finally:
            # Always save this sample's data
            answer['prompt_text'] = pinfo['prompt']
            labels[i] = pinfo['label']
            meta_list.append(pinfo['meta'])
            answers_list.append(answer)

            # Cleanup every sample
            if hidden is not None:
                del hidden
            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        if (i + 1) % 5 == 0 or i == 0:
            pct = (i + 1) / n_prompts * 100
            eta = elapsed * (n_prompts - i - 1) / 60
            exp_cfg = f"{pinfo['exp_name']}_cfg{pinfo['config_id']}"
            print(f"  [{iter_name.upper()}][{i+1:3d}/{n_prompts}] {pct:5.1f}% | "
                  f"{elapsed:.0f}s | ETA {eta:.0f}min | {exp_cfg}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             hidden_states=hidden_states,
             labels=labels,
             meta=np.array(meta_list, dtype=object),
             answers=np.array(answers_list, dtype=object))

    total_time = (time.time() - t0) / 60
    n_oom = sum(1 for a in answers_list if a.get('status') == 'oom')
    print(f"  [{iter_name.upper()}] Done: {n_prompts} prompts | {total_time:.1f} min | "
          f"OOM: {n_oom}")
    print(f"  [{iter_name.upper()}] Saved: {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='cuebiq_id')
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--iter', nargs='+', default=['v7', 'v8'],
                        help='Iterations to run (default: v7 v8)')
    parser.add_argument('--align-v7', action='store_true',
                        help='Use V7 targets for V8 alignment')
    parser.add_argument('--cot-only', action='store_true',
                        help='Only run CoT configs (sys_prompt=True)')
    args = parser.parse_args()

    uid = args.user
    model_key = args.model
    iters = args.iter

    all_done = all(
        os.path.exists(user_output_path_iter(uid, model_key, it))
        for it in iters)
    if all_done:
        print(f"All done for {uid} ({', '.join(iters)})")
        return

    import torch
    t0 = time.time()
    tag = MODEL_REGISTRY[model_key]['tag']
    print(f"{'='*60}")
    print(f"Per-User Extraction: {tag}")
    print(f"  User:  {uid}")
    print(f"  Iters: {', '.join(iters)}")
    print(f"  Model: {model_key}")
    if args.cot_only:
        print(f"  Mode:  CoT-only")
    if MODEL_REGISTRY[model_key].get('use_stops', False):
        print(f"  Format: stops (start_time, end_time)")
    print(f"{'='*60}")

    df = load_data()
    user_df = df[df['cuebiq_id'] == uid]
    if len(user_df) == 0:
        print(f"ERROR: User {uid} not found in data")
        return
    is_employed = bool(user_df['is_employed'].iloc[0])
    print(f"  Employed: {is_employed}")

    model, tokenizer = load_model(model_key)
    device = next(model.parameters()).device

    # ── Run iterations ──
    # KEY FIX: align-v7 targets are loaded AFTER V7 completes,
    # from the SAME model's V7 output (not Gemma 3's).
    forced_targets = None

    for iter_name in iters:
        if iter_name == 'v8' and args.align_v7 and forced_targets is None:
            # Load V7 targets from THIS MODEL's V7 output
            v7_path = user_output_path_iter(uid, model_key, 'v7')
            forced_targets = load_v7_targets_from_path(v7_path)
            if forced_targets:
                for exp, tgts in forced_targets.items():
                    dates = sorted(set(d for d, _, _ in tgts))
                    print(f"  [align-v7] {exp}: using {len(dates)} forced dates "
                          f"from {v7_path}")
            else:
                print(f"  WARNING: No V7 targets found at {v7_path}, "
                      f"using random sampling for V8")

        run_one_iteration(
            model, tokenizer, device, model_key, iter_name,
            user_df, uid, is_employed,
            forced_targets=forced_targets if iter_name == 'v8' else None,
            cot_only=args.cot_only)

    total = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"All iterations done in {total:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
