#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NL-based steering prompts v9 (Apr 20).

10 fine-grained questions using two-letter codes.
Each code is a single token in Gemma's tokenizer (verified).

Exp2  (emp/unemp):   5 questions — time(2) + spending(1) + asset(1) + social(1)
Exp1a (wkday/wkend): 5 questions — time(2) + food(1) + leisure(1) + social(1)

Answer formats:
  'pct'    — 0% to 100% in 2% steps, 51 options (codes aa..ca)
  'time'   — 5:00AM to 11:55AM in 5-min steps, 84 options (codes aa..di)
  'dollar' — $2,000 to $27,000 in $500 steps, 51 options (codes aa..ca)
  'count'  — 0 to 98 in steps of 2, plus 100+, 51 options (codes aa..ca)
  'hours'  — 0.0h to 25.0h in 0.5h steps, 51 options (codes aa..ca)

Tested baseline distributions on Gemma-4-31B (Apr 20, 2026):
  FG_E2  P(breakfast 8AM Tue) Peak=0.43  >0.5%=8   ← time
  FG_E3  P(coffee 7AM Tue)    Peak=0.50  >0.5%=6   ← time
  FG_E4  P(lunch>$10 Tue)     Peak=0.50  >0.5%=7   ← spending
  FG_E5  Car value $2k-$27k   Peak=0.54  >0.5%=4   ← asset
  FG_E6  Msgs to coworkers    Peak=0.34  >0.5%=8   ← social (BEST)
  FG_W1  Wake-up time         Peak=0.54  >0.5%=3   ← time
  FG_W4  Meals cooked today%  Peak=0.40  >0.5%=10  ← food
  FG_W5  Entertainment today% Peak=0.46  >0.5%=10  ← leisure
  FG_W6  Hours social w/ F&F  Peak=0.37  >0.5%=6   ← social
  FG_W7  P(breakfast 8AM)     Peak=0.31  >0.5%=6   ← time (BEST overall)
"""

# ══════════════════════════════════════════════════════════════════════
# Code pools and labels
# ══════════════════════════════════════════════════════════════════════

_SPLIT_PAIRS = {'gq', 'lq', 'qf', 'qg', 'qj', 'tq', 'vj', 'wz', 'yq', 'yv'}

TWOLETTER_CODES = sorted([
    a + b
    for a in 'abcdefghijklmnopqrstuvwxyz'
    for b in 'abcdefghijklmnopqrstuvwxyz'
    if (a + b) not in _SPLIT_PAIRS and 'q' not in (a + b)
])  # 622 codes

# ── pct: 0%..100% in 2% steps = 51 options ──
FG_PCT_CODES  = TWOLETTER_CODES[:51]   # aa..ca
FG_PCT_LABELS = [f'{i}%' for i in range(0, 102, 2)]

# ── time: 5:00AM..11:55AM in 5-min steps = 84 options ──
FG_TIME_CODES = TWOLETTER_CODES[:84]   # aa..di
FG_TIME_LABELS = []
for _h in range(5, 12):
    for _m in range(0, 60, 5):
        FG_TIME_LABELS.append(f'{_h}:{_m:02d}AM')

# ── dollar: $2,000..$27,000 in $500 steps = 51 options ──
FG_DOLLAR_CODES  = TWOLETTER_CODES[:51]
FG_DOLLAR_LABELS = [f'${2000 + i * 500:,}' for i in range(51)]

# ── count: 0,2,4,...,98,100+ = 51 options ──
FG_COUNT_CODES  = TWOLETTER_CODES[:51]
FG_COUNT_LABELS = [str(i * 2) for i in range(50)] + ['100+']

# ── hours: 0.0h..25.0h in 0.5h steps = 51 options ──
FG_HOURS_CODES  = TWOLETTER_CODES[:51]
FG_HOURS_LABELS = [f'{i * 0.5:.1f}h' for i in range(51)]


# Helper to build "Pick one: aa=X, ab=Y, ..." string
def _opts(codes, labels):
    return ', '.join(f'{c}={l}' for c, l in zip(codes, labels))


_PCT_OPTS = _opts(FG_PCT_CODES, FG_PCT_LABELS)
_DOLLAR_OPTS = _opts(FG_DOLLAR_CODES, FG_DOLLAR_LABELS)
_COUNT_OPTS = _opts(FG_COUNT_CODES, FG_COUNT_LABELS)
_HOURS_OPTS = _opts(FG_HOURS_CODES, FG_HOURS_LABELS)
_TIME_OPTS = _opts(FG_TIME_CODES, FG_TIME_LABELS)


# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Employed vs Unemployed  (5 questions)
# ══════════════════════════════════════════════════════════════════════
# E2/E3: on Tuesday (controls weekday/weekend variable).
# E4: on Tuesday (same reason).
# E5/E6: on Tuesday (employment-specific).

EXP2_QUESTIONS = [
    # ── E2: Time dimension — breakfast ──
    {
        'id': 'FG_E2',
        'type': 'finegrain',
        'exp': 'exp2',
        'answer_format': 'pct',
        'fg_codes': FG_PCT_CODES,
        'fg_labels': FG_PCT_LABELS,
        'description': 'P(breakfast by 8AM Tue)',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        'prompt': (
            'What is the probability that a typical adult has eaten '
            'breakfast by 8:00 AM on a Tuesday? '
            'Pick one: ' + _PCT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── E3: Time dimension — coffee ──
    {
        'id': 'FG_E3',
        'type': 'finegrain',
        'exp': 'exp2',
        'answer_format': 'pct',
        'fg_codes': FG_PCT_CODES,
        'fg_labels': FG_PCT_LABELS,
        'description': 'P(coffee by 7AM Tue)',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        'prompt': (
            'What is the probability that a typical adult has had '
            'coffee by 7:00 AM on a Tuesday? '
            'Pick one: ' + _PCT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── E4: Spending dimension — lunch cost ──
    {
        'id': 'FG_E4',
        'type': 'finegrain',
        'exp': 'exp2',
        'answer_format': 'pct',
        'fg_codes': FG_PCT_CODES,
        'fg_labels': FG_PCT_LABELS,
        'description': 'P(lunch > $10 on Tue)',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Employed: eat out at work → higher P(>$10)
        # Unemployed: eat at home → lower P(>$10)
        # Baseline: Peak=0.50 at 60%, Spread(>0.5%)=7
        'prompt': (
            'What is the probability that a typical adult spends '
            'more than 10 US dollars on lunch on a Tuesday? '
            'Pick one: ' + _PCT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── E5: Asset dimension — car value ──
    {
        'id': 'FG_E5',
        'type': 'finegrain',
        'exp': 'exp2',
        'answer_format': 'dollar',
        'fg_codes': FG_DOLLAR_CODES,
        'fg_labels': FG_DOLLAR_LABELS,
        'description': 'Value of car driven (USD)',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Employed: higher income → more expensive car
        # Unemployed: lower income → cheaper car
        # Baseline: Peak=0.54 at $15k, Spread(>0.5%)=4
        'prompt': (
            'What is the value of the car that a typical adult '
            'in the United States drives? '
            'Pick one: ' + _DOLLAR_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── E6: Social dimension — work messages ──
    {
        'id': 'FG_E6',
        'type': 'finegrain',
        'exp': 'exp2',
        'answer_format': 'count',
        'fg_codes': FG_COUNT_CODES,
        'fg_labels': FG_COUNT_LABELS,
        'description': 'Work messages to coworkers Tue',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Employed: many work messages → higher count
        # Unemployed: no coworkers → zero or near-zero
        # Baseline: Peak=0.34 at 10, Spread(>0.5%)=8
        'prompt': (
            'How many texts, emails, or chat messages does a typical '
            'adult send to coworkers on a Tuesday? '
            'Pick one: ' + _COUNT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════
# Experiment 1a: Weekday vs Weekend  (5 questions)
# ══════════════════════════════════════════════════════════════════════
# No specific day mentioned (intentionally vague).
# Steering with V_weekday should shift toward weekday behavior.
# Steering with V_weekend should shift toward weekend behavior.

EXP1A_QUESTIONS = [
    # ── W1: Time dimension — wake-up time ──
    {
        'id': 'FG_W1',
        'type': 'finegrain',
        'exp': 'exp1a',
        'answer_format': 'time',
        'fg_codes': FG_TIME_CODES,
        'fg_labels': FG_TIME_LABELS,
        'description': 'Wake-up time (finegrain)',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Weekday: earlier (~6:30AM). Weekend: later (~8:30AM).
        # Baseline: Peak=0.54 at 6:30AM, Spread(>0.5%)=3
        'prompt': (
            'At what time does a typical adult wake up in the morning? '
            'Pick one: ' + _TIME_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── W4: Food dimension — cooking at home ──
    {
        'id': 'FG_W4',
        'type': 'finegrain',
        'exp': 'exp1a',
        'answer_format': 'pct',
        'fg_codes': FG_PCT_CODES,
        'fg_labels': FG_PCT_LABELS,
        'description': 'Pct meals cooked at home today',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Weekend: more time to cook → higher %
        # Weekday: grab-and-go / eat out → lower %
        # Baseline: Peak=0.40 at 80%, Spread(>0.5%)=10
        'prompt': (
            'What percentage of meals does a typical adult cook '
            'at home today? '
            'Pick one: ' + _PCT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── W5: Leisure dimension — entertainment ──
    {
        'id': 'FG_W5',
        'type': 'finegrain',
        'exp': 'exp1a',
        'answer_format': 'pct',
        'fg_codes': FG_PCT_CODES,
        'fg_labels': FG_PCT_LABELS,
        'description': 'Pct free time on entertainment today',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Weekend: more free time → higher % on entertainment
        # Weekday: work occupies most time → lower %
        # Baseline: Peak=0.46 at 50%, Spread(>0.5%)=10
        'prompt': (
            'What percentage of their free time does a typical adult '
            'spend on entertainment such as watching TV, playing games, '
            'or browsing the internet today? '
            'Pick one: ' + _PCT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── W6: Social dimension — voluntary socializing ──
    {
        'id': 'FG_W6',
        'type': 'finegrain',
        'exp': 'exp1a',
        'answer_format': 'hours',
        'fg_codes': FG_HOURS_CODES,
        'fg_labels': FG_HOURS_LABELS,
        'description': 'Hours voluntarily socializing',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Weekend: more voluntary social time → higher hours
        # Weekday: work limits social time → lower hours
        # Baseline: Peak=0.37 at 3.0h, Spread(>0.5%)=6
        'prompt': (
            'How many hours does a typical adult spend voluntarily '
            'socializing with friends or family? '
            'Pick one: ' + _HOURS_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
    # ── W7: Time dimension — breakfast (no day specified) ──
    {
        'id': 'FG_W7',
        'type': 'finegrain',
        'exp': 'exp1a',
        'answer_format': 'pct',
        'fg_codes': FG_PCT_CODES,
        'fg_labels': FG_PCT_LABELS,
        'description': 'P(breakfast by 8AM in the morning)',
        'eval_method': 'finegrain_distribution',
        'is_control': False,
        'strength': 'strong',
        # Weekday: eat before work → higher P
        # Weekend: sleep in, brunch → lower P
        # Baseline: Peak=0.31 at 60%, Spread(>0.5%)=6
        'prompt': (
            'What is the probability that a typical adult has eaten '
            'breakfast by 8:00 AM in the morning? '
            'Pick one: ' + _PCT_OPTS + '. '
            'Reply with ONLY the two-letter code: '
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════
# Exports
# ══════════════════════════════════════════════════════════════════════

ALL_QUESTIONS = {
    'exp2': EXP2_QUESTIONS,
    'exp1a': EXP1A_QUESTIONS,
}
