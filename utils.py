#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for Neural Mechanics V7/V8.
15-min intervals, forward hooks, per-user architecture.
Supports Gemma 3 12B + Gemma 4 31B (stops format + AutoProcessor).
"""
import json, random, hashlib, re, os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    MODEL_REGISTRY, DATA_DIR, DOW_NAMES, TIME_INTERVAL,
    EXP_CONFIG, SAMPLING_CONFIG, TIME_WINDOWS, RANDOM_SEED,
    SYSTEM_PROMPT_COT, GENERATION_CONFIG, GENERATION_CONFIG_GEMMA4,
    MAX_NEW_TOKENS_COT, MAX_NEW_TOKENS_NO_COT, MAX_NEW_TOKENS_COT_GEMMA4,
    CONCEPT_HINT, CONCEPT_HINT_V7, CONCEPT_HINT_V8,
    get_system_prompt_cot,
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
# History Building — Original (15-min intervals)
# ══════════════════════════════════════════════════════════════════════

def _build_lines(user_df, dates, window, geo_hash, day_hash, include_dow):
    """Build history lines in original 15-min interval format."""
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


# ══════════════════════════════════════════════════════════════════════
# History Building — Stops Format (start_time, end_time)
# ══════════════════════════════════════════════════════════════════════

def _group_stops(rows, geo_hash, day_hash, include_dow):
    """Group consecutive same-location records into stops.

    Input:  sorted DataFrame rows with (dow, hour, min5, geo_id)
    Output: list of strings like "Tuesday, 05:00, 20:00, 250214412041"
    """
    lines = []
    cur_geo = None
    cur_dow = None
    start_ts = None
    end_ts = None

    for _, r in rows.iterrows():
        geo_raw = str(r['geo_id'])
        geo = _fmt_geo(r['geo_id'], geo_hash)
        ts = _ts(r)

        if geo_raw == cur_geo:
            # Extend current stop
            end_ts = ts
        else:
            # Emit previous stop
            if cur_geo is not None:
                geo_fmt = _fmt_geo(cur_geo, geo_hash)
                if include_dow:
                    dow_fmt = _fmt_dow(cur_dow, day_hash)
                    lines.append(f"{dow_fmt}, {start_ts}, {end_ts}, {geo_fmt}")
                else:
                    lines.append(f"{start_ts}, {end_ts}, {geo_fmt}")
            # Start new stop
            cur_geo = geo_raw
            cur_dow = r['dow']
            start_ts = ts
            end_ts = ts

    # Emit last stop
    if cur_geo is not None:
        geo_fmt = _fmt_geo(cur_geo, geo_hash)
        if include_dow:
            dow_fmt = _fmt_dow(cur_dow, day_hash)
            lines.append(f"{dow_fmt}, {start_ts}, {end_ts}, {geo_fmt}")
        else:
            lines.append(f"{start_ts}, {end_ts}, {geo_fmt}")

    return lines


def _build_stops_lines(user_df, dates, window, geo_hash, day_hash, include_dow):
    """Build history lines in stops format (start_time, end_time, location)."""
    random.shuffle(dates)
    lines = []
    for date in dates:
        day = _filter_window(user_df[user_df['date'] == date], window)
        lines.extend(_group_stops(day, geo_hash, day_hash, include_dow))
    return lines


def _history_exp1a(user_df, target, hist_window, geo_hash, use_stops=False):
    cfg = SAMPLING_CONFIG['exp1a']
    di = _date_info(user_df[user_df['date'] != target])
    we, wd = _dates_exp1a(di)
    if len(we) < cfg['n_hist_weekends'] or len(wd) < cfg['n_hist_weekdays']:
        return None
    sampled = (random.sample(we, cfg['n_hist_weekends']) +
               random.sample(wd, cfg['n_hist_weekdays']))
    if use_stops:
        return _build_stops_lines(user_df, sampled, hist_window, geo_hash,
                                  False, False)
    return _build_lines(user_df, sampled, hist_window, geo_hash, False, False)


def _history_exp2(user_df, target, hist_window, geo_hash, day_hash,
                  use_stops=False):
    cfg = SAMPLING_CONFIG['exp2']
    di = _date_info(user_df[user_df['date'] != target])
    wd = _dates_workday(di)
    if len(wd) < cfg['n_hist_days']:
        return None
    sampled = random.sample(wd, cfg['n_hist_days'])
    if use_stops:
        return _build_stops_lines(user_df, sampled, hist_window, geo_hash,
                                  day_hash, True)
    return _build_lines(user_df, sampled, hist_window, geo_hash, day_hash, True)


# ══════════════════════════════════════════════════════════════════════
# Target Selection
# ══════════════════════════════════════════════════════════════════════

def _targets_exp1a(user_df):
    cfg = SAMPLING_CONFIG['exp1a']
    di = _date_info(user_df)
    we, wd = _dates_exp1a(di)
    nwe, nwd = cfg['n_target_weekends'], cfg['n_target_weekdays']
    if len(we) < nwe + cfg['n_hist_weekends'] or \
       len(wd) < nwd + cfg['n_hist_weekdays']:
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

def build_prompt(user_df, target_date, predict_time, exp_name, cfg_dict,
                 concept_hint=None, use_stops=False):
    """Build one prompt. use_stops=True for Gemma 4 grouped format."""
    exp_cfg = EXP_CONFIG[exp_name]
    hist_window = exp_cfg['hist_window']
    ctx_window = exp_cfg['time_window']
    gh, dh = cfg_dict['geo_hash'], cfg_dict['day_hash']
    idow = exp_cfg['include_dow']

    if exp_name == 'exp1a':
        hist = _history_exp1a(user_df, target_date, hist_window, gh,
                              use_stops=use_stops)
    else:
        hist = _history_exp2(user_df, target_date, hist_window, gh, dh,
                             use_stops=use_stops)
    if hist is None:
        return None

    # Current Day Context
    ctx = _filter_window(user_df[user_df['date'] == target_date], ctx_window)
    if len(ctx) == 0:
        return None

    if use_stops:
        ctx_lines = _group_stops(ctx, gh, dh, idow)
    else:
        ctx_lines = []
        for _, r in ctx.iterrows():
            geo = _fmt_geo(r['geo_id'], gh)
            ts = _ts(r)
            if idow:
                ctx_lines.append(f"{_fmt_dow(r['dow'], dh)}, {ts}, {geo}")
            else:
                ctx_lines.append(f"{ts}, {geo}")

    hint_text = concept_hint if concept_hint is not None else CONCEPT_HINT[exp_name]

    # Choose prompt format string
    fmt = exp_cfg['prompt_fmt_stops'] if use_stops else exp_cfg['prompt_fmt']

    return (
        f"Your task is to predict the next location for an individual "
        f"based on this individual's mobility trajectory."
        f"{hint_text}\n"
        f"Each record: {fmt}\n\n"
        f"=== Mobility History ===\n"
        + "\n".join(hist) + "\n\n"
        f"=== Current Day Context ===\n"
        + "\n".join(ctx_lines) + "\n\n"
        f"=== Prediction ===\n"
        f"Question: At {predict_time}, what location will this person be at?\n"
        f"Answer:"
    )


# ══════════════════════════════════════════════════════════════════════
# V7 Target Alignment
# ══════════════════════════════════════════════════════════════════════

def load_v7_targets(uid):
    v7_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'outputs_v7', 'per_user', f'{uid}.npz')
    if not os.path.exists(v7_path):
        return None
    data = np.load(v7_path, allow_pickle=True)
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
    return targets


# ══════════════════════════════════════════════════════════════════════
# Build All Prompts for One User
# ══════════════════════════════════════════════════════════════════════

def build_user_prompts(user_df, uid, is_employed, forced_targets=None,
                       concept_hint=None, use_stops=False, cot_only=False):
    """Build all prompts for one user.

    Args:
        use_stops: If True, use grouped start/end time format (Gemma 4).
        cot_only:  If True, only build prompts where sys_prompt=True.
    """
    results = []
    target_fn = {'exp1a': _targets_exp1a, 'exp2': _targets_exp2}

    exps = ['exp2']
    if is_employed:
        exps = ['exp1a'] + exps

    for exp_name in exps:
        exp_cfg = EXP_CONFIG[exp_name]
        pred_times = exp_cfg['prediction_times']
        label_col = exp_cfg['label_col']

        exp_df = (user_df.query(exp_cfg['filter_query'])
                  if '&' in exp_cfg['filter_query'] else user_df)

        if forced_targets is not None and exp_name in forced_targets:
            forced = forced_targets[exp_name]
            target_dates = sorted(set(d for d, _, _ in forced))
            forced_lookup = {(d, pt): gt for d, pt, gt in forced}
            print(f"  [align-v7] {exp_name}: using {len(target_dates)} forced dates")
        else:
            random.seed(RANDOM_SEED + hash(uid) % 10000)
            target_dates = target_fn[exp_name](exp_df)
            forced_lookup = None

        if target_dates is None:
            continue

        hint = concept_hint[exp_name] if concept_hint else None

        for cfg_dict in CONFIG_MATRIX_CACHE[exp_name]:
            cfg_id = cfg_dict['id']
            if cot_only and not cfg_dict['sys_prompt']:
                continue
            for tdate in target_dates:
                for pt in pred_times:
                    if forced_lookup is not None and (tdate, pt) not in forced_lookup:
                        continue

                    random.seed(RANDOM_SEED + hash(f"{uid}_{tdate}_{pt}") % 10000)
                    prompt = build_prompt(exp_df, tdate, pt, exp_name, cfg_dict,
                                          concept_hint=hint, use_stops=use_stops)
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

from config import CONFIG_MATRIX
CONFIG_MATRIX_CACHE = {exp: CONFIG_MATRIX[exp] for exp in CONFIG_MATRIX}


# ══════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_model(model_key='12b'):
    """Load model + tokenizer/processor based on architecture.

    Gemma 3: AutoTokenizer + Gemma3ForCausalLM (unchanged)
    Gemma 4: AutoProcessor + AutoModelForMultimodalLM (per gemma4_explorer.ipynb)

    Returns: (model, tokenizer_or_processor)
    """
    cfg = MODEL_REGISTRY[model_key]
    name = cfg['hf_name']
    arch = cfg['arch']
    print(f"Loading model: {name} (arch={arch})")

    if arch == 'gemma4':
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        tokenizer = AutoProcessor.from_pretrained(name, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            name, dtype="auto", device_map="auto", local_files_only=True)
    elif arch == 'gemma3' and cfg['use_causal_lm']:
        from transformers import Gemma3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = Gemma3ForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto",
            local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto",
            local_files_only=True)

    model.eval()

    if arch == 'gemma4':
        n_layers = model.config.text_config.num_hidden_layers
        h_dim = model.config.text_config.hidden_size
    else:
        n_layers = model.config.num_hidden_layers
        h_dim = model.config.hidden_size
    print(f"  Arch: {arch} | Layers: {n_layers} | Hidden: {h_dim}")

    return model, tokenizer


def get_model_layers(model, model_key='12b'):
    """Return (embed_module, decoder_layers_list)."""
    arch = MODEL_REGISTRY[model_key]['arch']
    if arch == 'gemma4':
        return (model.model.language_model.embed_tokens,
                model.model.language_model.layers)
    return model.model.embed_tokens, model.model.layers


def get_model_config(model, model_key='12b'):
    arch = MODEL_REGISTRY[model_key]['arch']
    if arch == 'gemma4':
        return (model.config.text_config.num_hidden_layers,
                model.config.text_config.hidden_size)
    return model.config.num_hidden_layers, model.config.hidden_size


# ══════════════════════════════════════════════════════════════════════
# Chat Template & Parsing
# ══════════════════════════════════════════════════════════════════════

def wrap_with_chat_template(tokenizer, prompt, use_sys_prompt, model_key='12b'):
    """Wrap prompt in chat template. Returns tokenizer-ready dict or string.

    Gemma 3: returns string (tokenize later)
    Gemma 4: returns dict with input_ids, attention_mask (via processor)
    """
    arch = MODEL_REGISTRY[model_key]['arch']
    sys_prompt = get_system_prompt_cot(model_key) if use_sys_prompt else None

    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    if arch == 'gemma4':
        # processor returns string for text-only; tokenize manually
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        return tok(text, return_tensors="pt")
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def parse_answer(generated_text, model_key='12b'):
    """Parse model output to extract final answer.

    Priority for Gemma 4:
      1. <answer>geo_id</answer> tags (our instruction)
      2. Strip thinking block, then fall through

    Priority for Gemma 3:
      1. Strip </think>, then regex cascade
    """
    cleaned = generated_text.strip()
    arch = MODEL_REGISTRY.get(model_key, {}).get('arch', 'gemma3')

    # ── Gemma 4: try <answer> tags first ──
    answer_tag = re.search(r'<answer>\s*(\S+?)\s*</answer>', cleaned)
    if answer_tag:
        return answer_tag.group(1).strip()

    # ── Strip thinking blocks ──
    if arch == 'gemma4':
        # Gemma 4 format: <|channel>thought\n...<channel|>
        cleaned = re.sub(
            r'<\|channel>thought.*?<channel\|>',
            '', cleaned, flags=re.DOTALL).strip()
    # Gemma 3 format: <think>...</think>
    cleaned = cleaned.replace('<end_of_turn>', '').strip()
    if '</think>' in cleaned:
        cleaned = re.sub(r'.*?</think>', '', cleaned, flags=re.DOTALL).strip()

    # ── Try "Answer: GEOID" ──
    ans = re.search(r'\bAnswer:\s*([a-f0-9]{6,15}|[\d]{9,15})', cleaned)
    if ans:
        return ans.group(1).strip()

    # ── Try "Final Answer:" ──
    final = re.search(
        r'(?:Final Answer|final answer)[:\s]*\*?\*?\s*'
        r'([a-f0-9]{6,15}|[\d]{9,15})', cleaned)
    if final:
        return final.group(1).strip()

    # ── Try ```code block``` ──
    code_blocks = re.findall(r'```(?:text)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if code_blocks:
        block = code_blocks[-1].strip()
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) == 1 and len(lines[0]) < 50:
            return lines[0]

    # ── Try `backticked` or **bold** geo_id ──
    backticks = re.findall(r'`([a-f0-9]{6,15}|[\d]{9,15})`', cleaned)
    if backticks:
        return backticks[-1].strip()
    bold = re.findall(r'\*\*([a-f0-9]{6,15}|[\d]{9,15})\*\*', cleaned)
    if bold:
        return bold[-1].strip()

    # ── Try last geo_id on its own line ──
    for line in reversed(cleaned.split('\n')):
        line = line.strip()
        if re.match(r'^[a-f0-9]{6,15}$|^[\d]{9,15}$', line):
            return line

    # TODO: Matteo flagged that falling through to picking any geo_id
    # from reasoning text gives false positives. For now we keep a
    # conservative tail-only search. Revisit after checking Gemma 4 output.
    tail = cleaned[-300:]
    tail_geos = re.findall(r'([a-f0-9]{6,15}|[\d]{9,15})', tail)
    if tail_geos:
        return tail_geos[-1]

    return cleaned


# ══════════════════════════════════════════════════════════════════════
# Hidden State Extraction — Forward Hooks
# ══════════════════════════════════════════════════════════════════════

def _extract_with_hooks(model, inputs, model_key='12b'):
    """Forward pass via hooks → (logits_last, hidden_numpy)."""
    embed_module, decoder_layers = get_model_layers(model, model_key)
    collected = {}

    def embed_hook(module, inp, out):
        collected['embed'] = out[0, -1, :].float().cpu()

    def make_layer_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            collected[idx] = h[0, -1, :].float().cpu()
        return hook

    handles = []
    handles.append(embed_module.register_forward_hook(embed_hook))
    for idx, layer in enumerate(decoder_layers):
        handles.append(layer.register_forward_hook(make_layer_hook(idx)))

    out = model(**inputs)
    for h in handles:
        h.remove()

    logits_last = out.logits[0, -1, :].clone()
    n_layers = len(decoder_layers)
    hidden = torch.stack([collected['embed']] +
                         [collected[i] for i in range(n_layers)])

    del out, collected
    torch.cuda.empty_cache()
    return logits_last, hidden.detach().numpy()


# ══════════════════════════════════════════════════════════════════════
# Process One Sample
# ══════════════════════════════════════════════════════════════════════

def process_sample(model, tokenizer, prompt, use_sys, device, model_key='12b'):
    """Process one prompt: extract hidden state + generate answer.

    Returns: (hidden_numpy, answer_dict)
    """
    arch = MODEL_REGISTRY[model_key]['arch']
    template_out = wrap_with_chat_template(tokenizer, prompt, use_sys, model_key)

    if arch == 'gemma4':
        inputs = {k: v.to(device) for k, v in template_out.items()}
    else:
        inputs = tokenizer(template_out, return_tensors="pt", truncation=True,
                           max_length=128000)
        inputs = {k: v.to(device) for k, v in inputs.items()}

    n_tokens = inputs['input_ids'].shape[1]

    # --- Hidden states ---
    with torch.no_grad():
        logits_last, hidden = _extract_with_hooks(model, inputs, model_key)

    # --- Top-K tokens ---
    probs = torch.softmax(logits_last, dim=-1)
    topk = torch.topk(probs, k=10)
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    top_tokens = [tok.decode(t.item()).strip() for t in topk.indices]
    top_probs = topk.values.cpu().tolist()
    del logits_last, probs, topk
    torch.cuda.empty_cache()

    # --- Generate answer ---
    if arch == 'gemma4':
        max_new = MAX_NEW_TOKENS_COT_GEMMA4 if use_sys else MAX_NEW_TOKENS_NO_COT
        gen_cfg = {**GENERATION_CONFIG_GEMMA4, 'max_new_tokens': max_new}
    else:
        max_new = MAX_NEW_TOKENS_COT if use_sys else MAX_NEW_TOKENS_NO_COT
        gen_cfg = {**GENERATION_CONFIG, 'max_new_tokens': max_new}

    gen = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id')
                     else getattr(tokenizer, 'eos_token_id', 0),
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
        'parsed_answer': parse_answer(gen_text, model_key),
        'n_tokens': n_tokens,
    }
    return hidden, answer
