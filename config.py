#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration for Neural Mechanics V7.
Changes from V6: 15-min intervals, per-user jobs, no checkpoints,
simultaneous hidden state + answer collection.
"""
import os

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs_v7')
HIDDEN_DIR = os.path.join(OUTPUT_DIR, 'hidden_states')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
ANSWERS_DIR = os.path.join(OUTPUT_DIR, 'answers')
PER_USER_DIR = os.path.join(OUTPUT_DIR, 'per_user')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts', 'user_jobs')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for d in [OUTPUT_DIR, HIDDEN_DIR, RESULTS_DIR, ANSWERS_DIR, PER_USER_DIR,
          SCRIPTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    '12b': {'hf_name': 'google/gemma-3-12b-it', 'use_causal_lm': True,
            'tag': 'gemma3_12b_it'},
}
RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════════════════
# Time Settings — 15-minute intervals (was 5-min in V6)
# ══════════════════════════════════════════════════════════════════════
TIME_INTERVAL = 15  # minutes per slot

TIME_WINDOWS = {
    'morning':  {'start': 300, 'end': 510},   # 5:00–8:30 AM
    'full_day': {'start': 300, 'end': 1435},   # 5:00 AM–23:55 PM
}
# 15-min: full_day = 76 slots/day, morning = 14 slots/day

# ══════════════════════════════════════════════════════════════════════
# Day-of-Week (BQ convention: 1=Sun .. 7=Sat)
# ══════════════════════════════════════════════════════════════════════
DOW_NAMES = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
             5: 'Thursday', 6: 'Friday', 7: 'Saturday'}

# ══════════════════════════════════════════════════════════════════════
# Sampling — same counts as V6
# ══════════════════════════════════════════════════════════════════════
SAMPLING_CONFIG = {
    'exp1a': {
        'n_hist_weekends': 11,
        'n_hist_weekdays': 11,
        'n_target_weekends': 2,
        'n_target_weekdays': 2,
    },
    'exp2': {
        'n_hist_days': 22,
        'n_target_days': 2,
    },
}

# ══════════════════════════════════════════════════════════════════════
# Experiment Definitions
# ══════════════════════════════════════════════════════════════════════
EXP_CONFIG = {
    'exp1a': {
        'name': 'Weekday vs Weekend',
        'hist_window': 'full_day',
        'time_window': 'morning',
        'prediction_times': ['12:00', '17:00'],
        'label_col': 'is_weekday',
        'n_classes': 2,
        'filter_query': "in_exp1 == True",
        'include_dow': False,
        'prompt_fmt': 'time, location_id',
    },
    'exp2': {
        'name': 'Employed vs Unemployed',
        'hist_window': 'full_day',
        'time_window': 'morning',
        'prediction_times': ['12:00', '17:00'],
        'label_col': 'is_employed',
        'n_classes': 2,
        'filter_query': "in_exp2 == True & is_nonhol_weekday == 1",
        'include_dow': True,
        'prompt_fmt': 'day_of_week, time, location_id',
    },
}

# ══════════════════════════════════════════════════════════════════════
# Config Matrix
# ══════════════════════════════════════════════════════════════════════
def _make_configs_4():
    return [
        {'id': 1, 'geo_hash': False, 'day_hash': False, 'sys_prompt': False},
        {'id': 2, 'geo_hash': False, 'day_hash': False, 'sys_prompt': True},
        {'id': 3, 'geo_hash': True,  'day_hash': False, 'sys_prompt': False},
        {'id': 4, 'geo_hash': True,  'day_hash': False, 'sys_prompt': True},
    ]

def _make_configs_8():
    configs = []
    cid = 1
    for geo in [False, True]:
        for day in [False, True]:
            for sp in [False, True]:
                configs.append({'id': cid, 'geo_hash': geo,
                                'day_hash': day, 'sys_prompt': sp})
                cid += 1
    return configs

CONFIG_MATRIX = {
    'exp1a': _make_configs_4(),
    'exp2':  _make_configs_8(),
}

def get_config(exp_name, config_id):
    for c in CONFIG_MATRIX[exp_name]:
        if c['id'] == config_id:
            return c
    raise ValueError(f"Config {config_id} not found for {exp_name}")

def config_label(exp_name, config_id):
    c = get_config(exp_name, config_id)
    parts = [f"geo={'hash' if c['geo_hash'] else 'orig'}"]
    if EXP_CONFIG[exp_name]['include_dow']:
        parts.append(f"day={'hash' if c['day_hash'] else 'orig'}")
    parts.append(f"prompt={'cot' if c['sys_prompt'] else 'none'}")
    return '_'.join(parts)

# ══════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_COT = (
    "You are a highly logical and methodical AI assistant. "
    "Before arriving at a final answer, you must engage in a deep, step-by-step "
    "thinking process. You must enclose your entire reasoning process within "
    "<think> and </think> tags. Break down the problem, question your assumptions, "
    "and show your work. Only after you have closed the </think> tag should you "
    "provide your final, concise answer."
)

GENERATION_CONFIG = {
    'do_sample': False,
    'temperature': 0.0,
    'top_p': 1.0,
}
MAX_NEW_TOKENS_COT = 4096
MAX_NEW_TOKENS_NO_COT = 128

# ══════════════════════════════════════════════════════════════════════
# Per-User Output Helpers
# ══════════════════════════════════════════════════════════════════════
def get_model_dirs(model_key='12b'):
    tag = MODEL_REGISTRY[model_key]['tag']
    dirs = {
        'hidden':  os.path.join(HIDDEN_DIR, tag),
        'answers': os.path.join(ANSWERS_DIR, tag),
        'results': os.path.join(RESULTS_DIR, tag),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def user_output_path(user_id):
    return os.path.join(PER_USER_DIR, f"{user_id}.npz")
