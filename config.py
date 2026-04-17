#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration for Neural Mechanics — Multi-Model Support.
Supports: Gemma 3 12B (unsloth_env) + Gemma 4 31B (gemma4_env).
"""
import os

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts', 'user_jobs')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs_v8')
HIDDEN_DIR = os.path.join(OUTPUT_DIR, 'hidden_states')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
ANSWERS_DIR = os.path.join(OUTPUT_DIR, 'answers')
PER_USER_DIR = os.path.join(OUTPUT_DIR, 'per_user')

for d in [OUTPUT_DIR, HIDDEN_DIR, RESULTS_DIR, ANSWERS_DIR, PER_USER_DIR,
          SCRIPTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Model Registry
# ══════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    '12b': {
        'hf_name': 'google/gemma-3-12b-it',
        'tag': 'gemma3_12b_it',
        'arch': 'gemma3',
        'use_causal_lm': True,
        'conda_env': 'unsloth_env',
        'use_stops': False,
    },
    'gemma4_31b': {
        'hf_name': os.path.join(BASE_DIR, 'gemma-4-31B-it-unsloth-bnb-4bit'),
        'tag': 'gemma4_31b_it_bnb4',
        'arch': 'gemma4',
        'use_causal_lm': False,
        'conda_env': 'gemma4_env',
        'use_stops': True,
    },
    'gemma4_31b_bf16': {
        'hf_name': os.path.join(BASE_DIR, 'google-gemma-4-31B-it'),
        'tag': 'gemma4_31b_it_bf16',
        'arch': 'gemma4',
        'use_causal_lm': False,
        'conda_env': 'gemma4_env',
        'use_stops': True,
    },
}

RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════════════════
# Output Paths
# ══════════════════════════════════════════════════════════════════════

def get_iter_output_dir(model_key, iter_name):
    if model_key == '12b':
        return os.path.join(BASE_DIR, f'outputs_{iter_name}')
    cfg = MODEL_REGISTRY[model_key]
    return os.path.join(cfg['hf_name'], f'outputs_{iter_name}')

def get_iter_per_user_dir(model_key, iter_name):
    d = os.path.join(get_iter_output_dir(model_key, iter_name), 'per_user')
    os.makedirs(d, exist_ok=True)
    return d

def get_iter_model_dirs(model_key, iter_name):
    base = get_iter_output_dir(model_key, iter_name)
    tag = MODEL_REGISTRY[model_key]['tag']
    dirs = {
        'hidden':  os.path.join(base, 'hidden_states', tag),
        'answers': os.path.join(base, 'answers', tag),
        'results': os.path.join(base, 'results', tag),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def user_output_path_iter(user_id, model_key, iter_name):
    return os.path.join(get_iter_per_user_dir(model_key, iter_name),
                        f"{user_id}.npz")

def user_output_path(user_id):
    return os.path.join(PER_USER_DIR, f"{user_id}.npz")

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

# ══════════════════════════════════════════════════════════════════════
# Time Settings
# ══════════════════════════════════════════════════════════════════════
TIME_INTERVAL = 15
TIME_WINDOWS = {
    'morning':  {'start': 300, 'end': 510},
    'full_day': {'start': 300, 'end': 1435},
}
DOW_NAMES = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
             5: 'Thursday', 6: 'Friday', 7: 'Saturday'}

# ══════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════
SAMPLING_CONFIG = {
    'exp1a': {
        'n_hist_weekends': 9, 'n_hist_weekdays': 9,####change here if oom, was 11 and 11
        'n_target_weekends': 2, 'n_target_weekdays': 2,
    },
    'exp2': {
        'n_hist_days': 18, 'n_target_days': 2,#### change here if oom, was 22
    },
}

# ══════════════════════════════════════════════════════════════════════
# Concept Hints
# ══════════════════════════════════════════════════════════════════════
CONCEPT_HINT_V7 = {'exp1a': '', 'exp2': ''}
CONCEPT_HINT_V8 = {
    'exp1a': ("\nNote that visitation patterns may differ depending on "
              "whether the day is a weekday or weekend."),
    'exp2':  ("\nNote that visitation patterns may be affected by whether "
              "or not the individual is employed or unemployed."),
}
CONCEPT_HINT = CONCEPT_HINT_V8
ACTIVE_EXPERIMENTS = ['exp1a', 'exp2']

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
        'prompt_fmt_stops': 'start_time, end_time, location_id',
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
        'prompt_fmt_stops': 'day_of_week, start_time, end_time, location_id',
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

COT_ONLY_IDS = {
    'exp1a': [2, 4],
    'exp2':  [2, 4, 6, 8],
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
# System Prompts & Generation
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_COT_GEMMA3 = (
    "You are a highly logical and methodical AI assistant. "
    "Before arriving at a final answer, you must engage in a deep, step-by-step "
    "thinking process. You must enclose your entire reasoning process within "
    "<think> and </think> tags. Break down the problem, question your assumptions, "
    "and show your work. Only after you have closed the </think> tag should you "
    "provide your final, concise answer."
)

# Gemma 4: <|think|> at start enables thinking mode (per Google model card).
# Answer format instructions ensure parseable output.
# this way force the think mode on, need to remove  "<|think|>\n" if want to use none cot later.
SYSTEM_PROMPT_COT_GEMMA4 = (
    "<|think|>\n"
    "You are a mobility pattern analysis assistant. "
    "Think step-by-step about the person's movement patterns.\n\n"
    "IMPORTANT: After your reasoning, provide your final answer as a single "
    "location_id wrapped in <answer> and </answer> tags.\n"
    "Example: <answer>250214412041</answer>\n"
    "Do NOT include any other text between the answer tags."
)

SYSTEM_PROMPT_COT = SYSTEM_PROMPT_COT_GEMMA3

def get_system_prompt_cot(model_key='12b'):
    arch = MODEL_REGISTRY[model_key]['arch']
    if arch == 'gemma4':
        return SYSTEM_PROMPT_COT_GEMMA4
    return SYSTEM_PROMPT_COT_GEMMA3

GENERATION_CONFIG = {
    'do_sample': False, 'temperature': 0.0, 'top_p': 1.0,
}
GENERATION_CONFIG_GEMMA4 = {
    'do_sample': True, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 64,
}
MAX_NEW_TOKENS_COT = 4096
MAX_NEW_TOKENS_NO_COT = 128
MAX_NEW_TOKENS_COT_GEMMA4 = 4096
