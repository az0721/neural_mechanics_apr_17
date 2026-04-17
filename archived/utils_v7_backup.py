#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for Neural Mechanics V7.
15-min intervals, forward hooks, no checkpoints, per-user architecture.
"""
import json, random, hashlib, re, os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    MODEL_REGISTRY, DATA_DIR, DOW_NAMES, TIME_INTERVAL,
    EXP_CONFIG, SAMPLING_CONFIG, TIME_WINDOWS, RANDOM_SEED,
    SYSTEM_PROMPT_COT, GENERATION_CONFIG,
    MAX_NEW_TOKENS_COT, MAX_NEW_TOKENS_NO_COT,
)


# ══════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════

def load_data():
    path = os.path.join(DATA_DIR, 'trajectories_processed.csv')
    df = pd.read_csv(path, low_memory=False)
    df['date'] = df['date'].astype(str)
    print(f"Loaded {len(df):,} rows, {df['cuebiq_id'].nunique()} users")
    return df

def load_users():
    path = os.path.join(DATA_DIR, 'users.csv')
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════

def _fmt_geo(geo_id, use_hash):
    return hashlib.md5(str(geo_id).encode()).hexdigest()[:8] if use_hash else str(geo_id)

def _fmt_dow(dow_int, use_hash):
    name = DOW_NAMES[int(dow_int)]
    return hashlib.md5(name.encode()).hexdigest()[:6] if use_hash else name


# ══════════════════════════════════════════════════════════════════════
# Time Window — 15-minute filtering
# ══════════════════════════════════════════════════════════════════════

def _filter_window(day_df, window_name):
    """Filter to time window AND 15-min intervals."""
    w = TIME_WINDOWS[window_name]
    minutes = day_df['hour'] * 60 + day_df['min5']
    in_window = (minutes >= w['start']) & (minutes <= w['end'])
    is_15min = (day_df['min5'] % TIME_INTERVAL == 0)
    return day_df[in_window & is_15min].sort_values(['hour', 'min5'])

def _ts(row):
    return f"{int(row['hour']):02d}:{int(row['min5']):02d}"


# ══════════════════════════════════════════════════════════════════════
# Date Helpers
# ══════════════════════════════════════════════════════════════════════

def _date_info(user_df):
    return user_df.groupby('date').first().reset_index()

def _dates_exp1a(di):
    return (di[di['is_weekend'] == 1]['date'].tolist(),
            di[di['is_nonhol_weekday'] == 1]['date'].tolist())

def _dates_workday(di):
    return di[di['is_nonhol_weekday'] == 1]['date'].tolist()


# ══════════════════════════════════════════════════════════════════════
# History Building
# ══════════════════════════════════════════════════════════════════════

def _build_lines(user_df, dates, window, geo_hash, day_hash, include_dow):
    random.shuffle(dates)
    lines = []
    for date in dates:
        day = _filter_window(user_df[user_df['date'] == date], window)
        for _, r in day.iterrows():
            geo = _fmt_geo(r['geo_id'], geo_hash)
            ts = _ts(r)
            if include_dow:
                lines.append(f"{_fmt_dow(r['dow'], day_hash)}, {ts}, {geo}")
            else:
                lines.append(f"{ts}, {geo}")
    return lines

def _history_exp1a(user_df, target, hist_window, geo_hash):
    cfg = SAMPLING_CONFIG['exp1a']
    di = _date_info(user_df[user_df['date'] != target])
    we, wd = _dates_exp1a(di)
    if len(we) < cfg['n_hist_weekends'] or len(wd) < cfg['n_hist_weekdays']:
        return None
    sampled = (random.sample(we, cfg['n_hist_weekends']) +
               random.sample(wd, cfg['n_hist_weekdays']))
    return _build_lines(user_df, sampled, hist_window, geo_hash, False, False)

def _history_exp2(user_df, target, hist_window, geo_hash, day_hash):
    cfg = SAMPLING_CONFIG['exp2']
    di = _date_info(user_df[user_df['date'] != target])
    wd = _dates_workday(di)
    if len(wd) < cfg['n_hist_days']:
        return None
    sampled = random.sample(wd, cfg['n_hist_days'])
    return _build_lines(user_df, sampled, hist_window, geo_hash, day_hash, True)


# ══════════════════════════════════════════════════════════════════════
# Target Selection
# ══════════════════════════════════════════════════════════════════════

def _targets_exp1a(user_df):
    cfg = SAMPLING_CONFIG['exp1a']
    di = _date_info(user_df)
    we, wd = _dates_exp1a(di)
    nwe, nwd = cfg['n_target_weekends'], cfg['n_target_weekdays']
    if len(we) < nwe + cfg['n_hist_weekends'] or len(wd) < nwd + cfg['n_hist_weekdays']:
        return None
    return random.sample(we, nwe) + random.sample(wd, nwd)

def _targets_exp2(user_df):
    cfg = SAMPLING_CONFIG['exp2']
    n_hist = cfg['n_hist_days']
    n_target = cfg['n_target_days']
    di = _date_info(user_df)
    wd = _dates_workday(di)
    if len(wd) < n_hist + n_target:
        return None
    return random.sample(wd, n_target)


# ══════════════════════════════════════════════════════════════════════
# Prompt Construction
# ══════════════════════════════════════════════════════════════════════

def build_prompt(user_df, target_date, predict_time, exp_name, cfg_dict):
    exp_cfg = EXP_CONFIG[exp_name]
    hist_window = exp_cfg['hist_window']
    ctx_window = exp_cfg['time_window']
    gh, dh = cfg_dict['geo_hash'], cfg_dict['day_hash']
    idow = exp_cfg['include_dow']

    if exp_name == 'exp1a':
        hist = _history_exp1a(user_df, target_date, hist_window, gh)
    else:
        hist = _history_exp2(user_df, target_date, hist_window, gh, dh)
    if hist is None:
        return None

    ctx = _filter_window(user_df[user_df['date'] == target_date], ctx_window)
    if len(ctx) == 0:
        return None
    ctx_lines = []
    for _, r in ctx.iterrows():
        geo = _fmt_geo(r['geo_id'], gh)
        ts = _ts(r)
        if idow:
            ctx_lines.append(f"{_fmt_dow(r['dow'], dh)}, {ts}, {geo}")
        else:
            ctx_lines.append(f"{ts}, {geo}")

    return (
        f"Your task is to predict the next location for an individual "
        f"based on this individual's mobility history.\n"
        f"Each record: {exp_cfg['prompt_fmt']}\n\n"
        f"=== Mobility History ===\n"
        + "\n".join(hist) + "\n\n"
        f"=== Current Day Context ===\n"
        + "\n".join(ctx_lines) + "\n\n"
        f"=== Prediction ===\n"
        f"Question: At {predict_time}, what location will this person be at?\n"
        f"Answer:"
    )


# ══════════════════════════════════════════════════════════════════════
# Build All Prompts for One User
# ══════════════════════════════════════════════════════════════════════

def build_user_prompts(user_df, uid, is_employed):
    """Build all prompts for one user across applicable experiments/configs.

    Returns list of dicts:
      {prompt, label, meta, exp_name, config_id, use_sys}
    """
    results = []
    target_fn = {'exp1a': _targets_exp1a, 'exp2': _targets_exp2}

    # Determine applicable experiments
    exps = ['exp2']  # all users do exp2
    if is_employed:
        exps = ['exp1a'] + exps  # employed also do exp1a

    for exp_name in exps:
        exp_cfg = EXP_CONFIG[exp_name]
        pred_times = exp_cfg['prediction_times']
        label_col = exp_cfg['label_col']

        # Filter user data for this experiment
        exp_df = user_df.query(exp_cfg['filter_query']) if '&' in exp_cfg['filter_query'] else user_df

        random.seed(RANDOM_SEED + hash(uid) % 10000)
        targets = target_fn[exp_name](exp_df)
        if targets is None:
            continue

        for cfg_dict in CONFIG_MATRIX_CACHE[exp_name]:
            cfg_id = cfg_dict['id']
            for tdate in targets:
                for pt in pred_times:
                    random.seed(RANDOM_SEED + hash(f"{uid}_{tdate}_{pt}") % 10000)
                    prompt = build_prompt(exp_df, tdate, pt, exp_name, cfg_dict)
                    if prompt is None:
                        continue

                    label = int(exp_df[exp_df['date'] == tdate][label_col].iloc[0])
                    ph, pm = int(pt.split(':')[0]), int(pt.split(':')[1])
                    gt_row = exp_df[(exp_df['date'] == tdate) &
                                    (exp_df['hour'] == ph) & (exp_df['min5'] == pm)]
                    gt_geo = str(gt_row['geo_id'].iloc[0]) if len(gt_row) > 0 else None

                    results.append({
                        'prompt': prompt,
                        'label': label,
                        'exp_name': exp_name,
                        'config_id': cfg_id,
                        'use_sys': cfg_dict['sys_prompt'],
                        'meta': {
                            'user': uid, 'date': tdate, 'pred_time': pt,
                            'gt_geo_id': gt_geo,
                            'is_employed': int(is_employed),
                            'exp_name': exp_name,
                            'config_id': cfg_id,
                        },
                    })
    return results

# Pre-cache config matrices (avoid repeated function calls)
from config import CONFIG_MATRIX
CONFIG_MATRIX_CACHE = {exp: CONFIG_MATRIX[exp] for exp in CONFIG_MATRIX}


# ══════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_model(model_key='12b'):
    cfg = MODEL_REGISTRY[model_key]
    name = cfg['hf_name']
    print(f"Loading model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    if cfg['use_causal_lm']:
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto",
            local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto",
            local_files_only=True)
    model.eval()
    print(f"  Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════
# Chat Template & Parsing
# ══════════════════════════════════════════════════════════════════════

def wrap_with_chat_template(tokenizer, prompt, use_sys_prompt):
    messages = []
    if use_sys_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT_COT})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

def parse_answer(generated_text):
    cleaned = generated_text.replace('<end_of_turn>', '').strip()

    if '</think>' in cleaned:
        cleaned = re.sub(r'.*?</think>', '', cleaned, flags=re.DOTALL).strip()

    # Try 1: ```code blocks``` → last one, extract just the geo_id
    code_blocks = re.findall(r'```(?:text)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if code_blocks:
        block = code_blocks[-1].strip()
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) == 1 and len(lines[0]) < 50:
            return lines[0]
        for line in lines:
            geo = re.search(r'([a-f0-9]{6,15}|[\d]{9,15})$', line)
            if geo:
                return geo.group(1)

    # Try 2: "Final Answer:" with optional **bold**/LaTeX
    final = re.search(
        r'\*?\*?(?:Final Answer|final answer)\*?\*?[:\s]*'
        r'(?:The final answer is\s*)?'
        r'\$?\\?\\?boxed\{([^}]+)\}?\$?',
        cleaned)
    if final:
        return final.group(1).strip()
    final = re.search(
        r'\*?\*?(?:Final Answer|final answer)\*?\*?[:\s]*'
        r'\*?\*?\s*([a-f0-9]{6,15}|[\d]{9,15})',
        cleaned)
    if final:
        return final.group(1).strip()

    # Try 3: "Answer: GEOID"
    ans = re.search(r'\bAnswer:\s*([a-f0-9]{6,15}|[\d]{9,15})', cleaned)
    if ans:
        return ans.group(1).strip()

    # Try 4: last "location/prediction is GEOID" in text
    preds = re.findall(
        r'(?:prediction|predicted location|likely location|probable location'
        r'|location)\s+(?:at \d{2}:\d{2}\s+)?is\s+'
        r'[\'\"*`]*([a-f0-9]{6,15}|[\d]{9,15})[\'\"*`]*',
        cleaned)
    if preds:
        return preds[-1].strip()

    # Try 5: **GEOID** (bold)
    bold = re.findall(r'\*\*([a-f0-9]{6,15}|[\d]{9,15})\*\*', cleaned)
    if bold:
        return bold[-1].strip()

    # Try 6: `backticked` geo_id
    backticks = re.findall(r'`([a-f0-9]{6,15}|[\d]{9,15})`', cleaned)
    if backticks:
        return backticks[-1].strip()

    # Try 7: quoted 'GEOID' or "GEOID"
    quoted = re.findall(r'[\'"]([a-f0-9]{6,15}|[\d]{9,15})[\'"]', cleaned)
    if quoted:
        return quoted[-1].strip()

    # Try 8: last geo_id-like token on its own line
    for line in reversed(cleaned.split('\n')):
        line = line.strip()
        if re.match(r'^[a-f0-9]{6,15}$|^[\d]{9,15}$', line):
            return line

    # Try 9: degenerate looping — same geo_id repeated many times
    geo_candidates = re.findall(r'([a-f0-9]{6,15}|[\d]{9,15})', cleaned)
    if len(geo_candidates) >= 4:
        # Check [-2] and [-3] since [-1] may be truncated
        if geo_candidates[-2] == geo_candidates[-3] == geo_candidates[-4]:
            return geo_candidates[-2]
    if len(geo_candidates) >= 3:
        if geo_candidates[-1] == geo_candidates[-2] == geo_candidates[-3]:
            return geo_candidates[-1]

    # Try 10: last geo_id in last 300 chars (CoT answer is always near the end)
    tail = cleaned[-300:]
    tail_geos = re.findall(r'([a-f0-9]{6,15}|[\d]{9,15})', tail)
    if tail_geos:
        return tail_geos[-1]

    # Fallback: direct answer (non-CoT bare geo_id)
    return cleaned

# def parse_answer(generated_text):
#     # Step 1: Remove <end_of_turn>
#     cleaned = generated_text.replace('<end_of_turn>', '').strip()

#     # Step 2: If <think> tags present, remove everything up to </think>
#     if '</think>' in cleaned:
#         cleaned = re.sub(r'.*?</think>', '', cleaned, flags=re.DOTALL).strip()

#     # Step 3: If code blocks exist, extract LAST one (CoT puts final answer there)
#     code_blocks = re.findall(r'```(?:text)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
#     if code_blocks:
#         return code_blocks[-1].strip()

#     # Step 4: Return as-is (non-CoT bare geo_id)
#     return cleaned




# ══════════════════════════════════════════════════════════════════════
# Hidden State Extraction — Forward Hooks
# ══════════════════════════════════════════════════════════════════════

def _extract_with_hooks(model, inputs):
    """Forward pass via hooks → (logits_last, hidden_numpy).
    Captures last token hidden state from each layer, moves to CPU.
    """
    collected = {}

    def embed_hook(module, inp, out):
        collected['embed'] = out[0, -1, :].float().cpu()

    def make_layer_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            collected[idx] = h[0, -1, :].float().cpu()
        return hook

    handles = []
    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))
    for idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_layer_hook(idx)))

    out = model(**inputs)
    for h in handles:
        h.remove()

    logits_last = out.logits[0, -1, :].clone()
    n_layers = len(model.model.layers)
    hidden = torch.stack([collected['embed']] +
                         [collected[i] for i in range(n_layers)])

    del out, collected
    torch.cuda.empty_cache()
    return logits_last, hidden.detach().numpy()


# ══════════════════════════════════════════════════════════════════════
# Process One Sample — Hidden States + Answer
# ══════════════════════════════════════════════════════════════════════

def process_sample(model, tokenizer, prompt, use_sys, device):
    """Process one prompt: extract hidden state + generate answer.

    Returns: (hidden_numpy, answer_dict)
    """
    full_text = wrap_with_chat_template(tokenizer, prompt, use_sys)
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                       max_length=128000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_tokens = inputs['input_ids'].shape[1]

    # --- Hidden states ---
    with torch.no_grad():
        logits_last, hidden = _extract_with_hooks(model, inputs)

    # --- Top-K tokens ---
    probs = torch.softmax(logits_last, dim=-1)
    topk = torch.topk(probs, k=10)
    top_tokens = [tokenizer.decode(t.item()).strip() for t in topk.indices]
    top_probs = topk.values.cpu().tolist()
    del logits_last, probs, topk
    torch.cuda.empty_cache()

    # --- Generate answer ---
    max_new = MAX_NEW_TOKENS_COT if use_sys else MAX_NEW_TOKENS_NO_COT
    gen_cfg = {**GENERATION_CONFIG, 'max_new_tokens': max_new}
    gen = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        **gen_cfg)
    gen_text = tokenizer.decode(
        gen[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=False).strip()
    del gen, inputs
    torch.cuda.empty_cache()

    answer = {
        'top_tokens': top_tokens,
        'top_probs': top_probs,
        'generated_text': gen_text,
        'parsed_answer': parse_answer(gen_text),
        'n_tokens': n_tokens,
    }
    return hidden, answer
