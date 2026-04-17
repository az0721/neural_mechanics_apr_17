# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# NL-based steering prompts v5.

# Revision based on GPT-4 critique of v4 (Apr 4):
#   - REMOVED all income/lifestyle-confounded prompts:
#       meals outside, free time, transport budget, dinner time, day description
#   - REMOVED overly conversational/abstract prompts:
#       phone call response, routine predictability (kept 1 as secondary)
#   - ALL remaining prompts are either:
#       (a) specific time-point behavioral states ("at X:00, where/what?")
#       (b) observable mobility timing ("what time does person do Y?")
#       (c) mobility-grounded durations ("how many hours at/away from home?")
#   - These map directly to GPS trajectory data (verifiable)
#   - Satisfy instrumental variable: correlated with employment/weekday,
#     NOT confounded by income, lifestyle, personality, family structure

# Design criteria (Matteo Apr 3 meeting):
#   - No "Boston", no "employed/work/weekday/weekend"
#   - Prefer numeric (quantitative, easier to interpret)
#   - Balanced MCQ choices (~33% neutral baseline each)
#   - Timing proxies > spending proxies > abstract proxies

# Exp2 (Employment):  10 Numeric + 10 MCQ = 20 questions
# Exp1a (Weekday):    10 Numeric + 10 MCQ = 20 questions
# """

# # ══════════════════════════════════════════════════════════════════════
# # Experiment 2: Employed vs Unemployed
# # ══════════════════════════════════════════════════════════════════════
# # positive coeff = toward employed, negative = toward unemployed
# # All users in this exp are on WEEKDAYS only.

# EXP2_QUESTIONS = [
#     # ══════════════════════════════════════
#     # NUMERIC (10) — all timing/mobility-grounded
#     # ══════════════════════════════════════
#     {
#         'id': 'E1', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Complete with a single number only.\n'
#             'A typical adult wakes up at approximately ___:00 AM '
#             'on a Tuesday.\n'
#         ),
#         'description': 'Wake-up hour (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',  # Matteo's canonical example
#     },
#     {
#         'id': 'E2', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Complete with a single number only.\n'
#             'A typical adult leaves home at approximately ___:00 AM '
#             'on a Tuesday.\n'
#         ),
#         'description': 'Leave-home hour (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E3', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Complete with a single number only.\n'
#             'A typical adult arrives home at approximately ___:00 PM '
#             'on a Tuesday.\n'
#         ),
#         'description': 'Arrive-home hour (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E4', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many hours does a typical adult spend at home '
#             'between 8 AM and 6 PM on a Tuesday? ___\n'
#         ),
#         'description': 'Daytime hours at home (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',  # grounded, directly from GPS
#     },
#     {
#         'id': 'E5', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'What is the longest continuous stretch of hours '
#             'a typical adult spends outside the home on a Tuesday? ___\n'
#         ),
#         'description': 'Longest stretch outside (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',  # captures "office-like" block
#     },
#     {
#         'id': 'E6', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many total hours is a typical adult away from home '
#             'on a Tuesday? ___\n'
#         ),
#         'description': 'Total hours away (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E7', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many hours does a typical adult spend at their '
#             'most frequently visited non-home location on a Tuesday? ___\n'
#         ),
#         'description': 'Hours at primary non-home location (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',  # captures regularity without saying "work"
#     },
#     {
#         'id': 'E8', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many times does a typical adult return home '
#             'during the daytime (8 AM to 6 PM) on a Tuesday? ___\n'
#         ),
#         'description': 'Daytime home returns (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'medium',  # employed=0-1, unemployed=multiple
#     },
#     {
#         'id': 'E9', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many hours does a typical adult spend commuting '
#             'or in transit on a Tuesday? ___\n'
#         ),
#         'description': 'Commute/transit hours (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'medium',  # employed commute more
#     },
#     {
#         'id': 'E10', 'type': 'numeric', 'exp': 'exp2',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'On a scale of 0 to 9, how consistent is a typical '
#             'adult\'s Tuesday schedule from week to week? '
#             '(0 = completely different, 9 = identical every week) ___\n'
#         ),
#         'description': 'Week-to-week consistency (Tue)',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'secondary',  # more abstract, but still timing-related
#     },

#     # ══════════════════════════════════════
#     # MCQ (10) — all time-anchored behavioral states
#     # ══════════════════════════════════════
#     {
#         'id': 'E11', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'At 8:00 AM on a Tuesday, a typical adult is most likely:\n'
#             '(A) Still sleeping or just waking up\n'
#             '(B) Awake at home, starting the day slowly\n'
#             '(C) Already out of the house or about to leave\n'
#             'Answer: ('
#         ),
#         'description': '8 AM Tuesday status',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E12', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'A typical adult is asked: "Are you usually free to meet '
#             'for coffee at 10:30 AM on a Tuesday?"\n'
#             'The most likely reply is:\n'
#             '(A) "Yes, I am usually available around that time."\n'
#             '(B) "It depends on the week; sometimes I can, sometimes not."\n'
#             '(C) "No, I am almost always occupied at that hour."\n'
#             'Answer: ('
#         ),
#         'description': '10:30 AM coffee availability',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E13', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'A typical adult is asked: "Can you meet me for lunch '
#             'at noon on a Tuesday?"\n'
#             'The most likely reply is:\n'
#             '(A) "Sure, I have no fixed plans around that time."\n'
#             '(B) "I can, but I only have about 30 minutes."\n'
#             '(C) "Sorry, I cannot leave where I am at noon."\n'
#             'Answer: ('
#         ),
#         'description': 'Noon Tuesday availability',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E14', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'If you wanted to visit a typical adult at 1 PM '
#             'on a Tuesday, which reply would be most likely?\n'
#             '(A) "I am usually at home around then."\n'
#             '(B) "I am usually at a regular daytime commitment then."\n'
#             '(C) "It varies; I might be in different places."\n'
#             'Answer: ('
#         ),
#         'description': '1 PM Tuesday availability',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E15', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'At 3:00 PM on a Tuesday, a typical adult is most likely:\n'
#             '(A) At home\n'
#             '(B) At the same non-home location they have been all day\n'
#             '(C) Moving between different locations\n'
#             'Answer: ('
#         ),
#         'description': '3 PM Tuesday location',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E16', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'At 5:00 PM on a Tuesday, a typical adult is most likely:\n'
#             '(A) At home, where they have been most of the day\n'
#             '(B) Heading home from a regular daytime location\n'
#             '(C) Still at a non-home location with no plans to leave yet\n'
#             'Answer: ('
#         ),
#         'description': '5 PM Tuesday status',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'E17', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'At 7:00 PM on a Tuesday, a typical adult is most likely:\n'
#             '(A) Has been home most of the day\n'
#             '(B) Recently arrived home after being out all day\n'
#             '(C) Still out, not yet headed home\n'
#             'Answer: ('
#         ),
#         'description': '7 PM Tuesday status',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'E18', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'Does a typical adult use an alarm clock on Tuesday mornings?\n'
#             '(A) No, they wake up naturally whenever they feel like it\n'
#             '(B) Sometimes, depending on whether they have plans\n'
#             '(C) Yes, they set an alarm at the same time every Tuesday\n'
#             'Answer: ('
#         ),
#         'description': 'Alarm clock habit (Tue)',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'E19', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'Does a typical adult leave home at the same time '
#             'every Tuesday?\n'
#             '(A) No, the departure time varies a lot\n'
#             '(B) Roughly similar, but not exact\n'
#             '(C) Yes, within a few minutes of the same time\n'
#             'Answer: ('
#         ),
#         'description': 'Departure time consistency (Tue)',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'E20', 'type': 'mcq', 'exp': 'exp2',
#         'prompt': (
#             'Is a typical adult at the same non-home location '
#             'at noon every Tuesday?\n'
#             '(A) No, they are usually at home at noon\n'
#             '(B) Sometimes the same place, sometimes not\n'
#             '(C) Yes, almost always at the same location\n'
#             'Answer: ('
#         ),
#         'description': 'Noon location consistency (Tue)',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
# ]


# # ══════════════════════════════════════════════════════════════════════
# # Experiment 1a: Weekday vs Weekend
# # ══════════════════════════════════════════════════════════════════════
# # positive coeff = toward weekday, negative = toward weekend
# # All users in this exp are EMPLOYED.

# EXP1A_QUESTIONS = [
#     # ══════════════════════════════════════
#     # NUMERIC (10) — all timing/mobility-grounded
#     # ══════════════════════════════════════
#     {
#         'id': 'W1', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Complete with a single number only.\n'
#             'An individual wakes up at approximately ___:00 AM.\n'
#         ),
#         'description': 'Wake-up hour',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W2', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Complete with a single number only.\n'
#             'An individual leaves home at approximately ___:00 AM.\n'
#         ),
#         'description': 'Leave-home hour',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W3', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Complete with a single number only.\n'
#             'An individual arrives home at approximately ___:00 PM.\n'
#         ),
#         'description': 'Arrive-home hour',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W4', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many hours does an individual spend at home '
#             'between 8 AM and 6 PM? ___\n'
#         ),
#         'description': 'Daytime hours at home',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W5', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'What is the longest continuous stretch of hours '
#             'an individual spends outside the home today? ___\n'
#         ),
#         'description': 'Longest stretch outside',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W6', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many total hours is an individual away from home '
#             'today? ___\n'
#         ),
#         'description': 'Total hours away',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W7', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many hours does an individual spend at their '
#             'most frequently visited non-home location today? ___\n'
#         ),
#         'description': 'Hours at primary non-home location',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W8', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many times does an individual return home '
#             'during the daytime (8 AM to 6 PM) today? ___\n'
#         ),
#         'description': 'Daytime home returns',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'W9', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'How many hours does an individual spend commuting '
#             'or in transit today? ___\n'
#         ),
#         'description': 'Commute/transit hours',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'W10', 'type': 'numeric', 'exp': 'exp1a',
#         'prompt': (
#             'Answer with a single digit from 0 to 9.\n'
#             'On a scale of 0 to 9, how consistent is an individual\'s '
#             'schedule today compared to the same day last week? '
#             '(0 = completely different, 9 = identical) ___\n'
#         ),
#         'description': 'Week-to-week consistency',
#         'eval_method': 'digit_distribution', 'is_control': False,
#         'strength': 'secondary',
#     },

#     # ══════════════════════════════════════
#     # MCQ (10) — all time-anchored behavioral states
#     # ══════════════════════════════════════
#     {
#         'id': 'W11', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'At 8:00 AM, an individual is most likely:\n'
#             '(A) Still sleeping or just waking up\n'
#             '(B) Awake at home, starting the day slowly\n'
#             '(C) Already out of the house or about to leave\n'
#             'Answer: ('
#         ),
#         'description': '8 AM status',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W12', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'An individual is asked: "Are you usually free to meet '
#             'for coffee at 10:30 AM?"\n'
#             'The most likely reply is:\n'
#             '(A) "Yes, I am usually available around that time."\n'
#             '(B) "It depends; sometimes I can, sometimes not."\n'
#             '(C) "No, I am almost always occupied at that hour."\n'
#             'Answer: ('
#         ),
#         'description': '10:30 AM coffee availability',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W13', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'An individual is asked: "Can you meet me for lunch '
#             'at noon?"\n'
#             'The most likely reply is:\n'
#             '(A) "Sure, I have no fixed plans around that time."\n'
#             '(B) "I can, but I only have about 30 minutes."\n'
#             '(C) "Sorry, I cannot leave where I am at noon."\n'
#             'Answer: ('
#         ),
#         'description': 'Noon availability',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W14', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'At 3:00 PM, an individual is most likely:\n'
#             '(A) At home\n'
#             '(B) At the same non-home location they have been all day\n'
#             '(C) Moving between different locations\n'
#             'Answer: ('
#         ),
#         'description': '3 PM location',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W15', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'At 5:00 PM, an individual is most likely:\n'
#             '(A) At home, where they have been most of the day\n'
#             '(B) Heading home from a regular daytime location\n'
#             '(C) Still at a non-home location with no plans to leave yet\n'
#             'Answer: ('
#         ),
#         'description': '5 PM status',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
#     {
#         'id': 'W16', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'At 7:00 PM, an individual is most likely:\n'
#             '(A) Has been home most of the day\n'
#             '(B) Recently arrived home after being out all day\n'
#             '(C) Still out, not yet headed home\n'
#             'Answer: ('
#         ),
#         'description': '7 PM status',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'W17', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'Does an individual use an alarm clock this morning?\n'
#             '(A) No, they wake up naturally\n'
#             '(B) Maybe, depending on plans\n'
#             '(C) Yes, set for a specific early time\n'
#             'Answer: ('
#         ),
#         'description': 'Alarm clock usage',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'W18', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'Does an individual leave home at the same time today '
#             'as they did the same day last week?\n'
#             '(A) No, the departure time is very different\n'
#             '(B) Roughly similar, but not exact\n'
#             '(C) Yes, within a few minutes of the same time\n'
#             'Answer: ('
#         ),
#         'description': 'Departure time consistency',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'W19', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'Is an individual at the same non-home location '
#             'at noon today as they were at noon last week?\n'
#             '(A) No, they are at home today\n'
#             '(B) They are out, but at a different location\n'
#             '(C) Yes, at the same location\n'
#             'Answer: ('
#         ),
#         'description': 'Noon location consistency',
#         'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'medium',
#     },
#     {
#         'id': 'W20', 'type': 'mcq', 'exp': 'exp1a',
#         'prompt': (
#             'If an individual is asked at 1 PM: '
#             '"Where are you right now?"\n'
#             'The most likely answer is:\n'
#             '(A) "I am at home."\n'
#             '(B) "I am at a regular place I go to most days."\n'
#             '(C) "I am out, but somewhere different from usual."\n'
#             'Answer: ('
#         ),
#         'description': '1 PM location',
#         'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
#         'eval_method': 'mcq', 'is_control': False,
#         'strength': 'strong',
#     },
# ]


# ALL_QUESTIONS = {
#     'exp2': EXP2_QUESTIONS,
#     'exp1a': EXP1A_QUESTIONS,
# }

# MCQ_TOKENS = ['A', 'B', 'C']
# DIGIT_TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NL-based steering prompts v6.

Changes from v5:
  - ALL MCQ prompts: 'Answer: (' → 'Answer with only the letter A, B, or C: '
    Fixes Gemma 4 logit extraction: P(A)+P(B)+P(C) now sums to ~1.0
    (was ~0.08 because model wanted to output '(' before the letter)
  - Exp1a: kept "today" intentionally — specifying "Tuesday" would
    reveal weekday identity, defeating the weekday-vs-weekend contrast.
    Exp1a steering tests whether the model's concept of "today" shifts
    toward weekday or weekend behavior patterns.
  - Exp2: already specifies "Tuesday" throughout (correct for
    employed-vs-unemployed on weekdays).

Design criteria (Matteo Apr 3 meeting):
  - No "Boston", no "employed/work/weekday/weekend"
  - Prefer numeric (quantitative, easier to interpret)
  - Balanced MCQ choices (~33% neutral baseline each)
  - Timing proxies > spending proxies > abstract proxies

Exp2 (Employment):  10 Numeric + 10 MCQ = 20 questions
Exp1a (Weekday):    10 Numeric + 10 MCQ = 20 questions
"""

MCQ_SUFFIX = 'Answer with only the letter A, B, or C: '

# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Employed vs Unemployed
# ══════════════════════════════════════════════════════════════════════
# positive coeff = toward employed, negative = toward unemployed
# All users in this exp are on WEEKDAYS only.

EXP2_QUESTIONS = [
    # ══════════════════════════════════════
    # NUMERIC (10) — all timing/mobility-grounded
    # ══════════════════════════════════════
    {
        'id': 'E1', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Complete with a single number only.\n'
            'A typical adult wakes up at approximately ___:00 AM '
            'on a Tuesday.\n'
        ),
        'description': 'Wake-up hour (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E2', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Complete with a single number only.\n'
            'A typical adult leaves home at approximately ___:00 AM '
            'on a Tuesday.\n'
        ),
        'description': 'Leave-home hour (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E3', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Complete with a single number only.\n'
            'A typical adult arrives home at approximately ___:00 PM '
            'on a Tuesday.\n'
        ),
        'description': 'Arrive-home hour (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E4', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many hours does a typical adult spend at home '
            'between 8 AM and 6 PM on a Tuesday? ___\n'
        ),
        'description': 'Daytime hours at home (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E5', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'What is the longest continuous stretch of hours '
            'a typical adult spends outside the home on a Tuesday? ___\n'
        ),
        'description': 'Longest stretch outside (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E6', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many total hours is a typical adult away from home '
            'on a Tuesday? ___\n'
        ),
        'description': 'Total hours away (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E7', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many hours does a typical adult spend at their '
            'most frequently visited non-home location on a Tuesday? ___\n'
        ),
        'description': 'Hours at primary non-home location (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E8', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many times does a typical adult return home '
            'during the daytime (8 AM to 6 PM) on a Tuesday? ___\n'
        ),
        'description': 'Daytime home returns (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'E9', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many hours does a typical adult spend commuting '
            'or in transit on a Tuesday? ___\n'
        ),
        'description': 'Commute/transit hours (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'E10', 'type': 'numeric', 'exp': 'exp2',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'On a scale of 0 to 9, how consistent is a typical '
            'adult\'s Tuesday schedule from week to week? '
            '(0 = completely different, 9 = identical every week) ___\n'
        ),
        'description': 'Week-to-week consistency (Tue)',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'secondary',
    },

    # ══════════════════════════════════════
    # MCQ (10) — all time-anchored behavioral states
    # ══════════════════════════════════════
    {
        'id': 'E11', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'At 8:00 AM on a Tuesday, a typical adult is most likely:\n'
            '(A) Still sleeping or just waking up\n'
            '(B) Awake at home, starting the day slowly\n'
            '(C) Already out of the house or about to leave\n'
            + MCQ_SUFFIX
        ),
        'description': '8 AM Tuesday status',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E12', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'A typical adult is asked: "Are you usually free to meet '
            'for coffee at 10:30 AM on a Tuesday?"\n'
            'The most likely reply is:\n'
            '(A) "Yes, I am usually available around that time."\n'
            '(B) "It depends on the week; sometimes I can, sometimes not."\n'
            '(C) "No, I am almost always occupied at that hour."\n'
            + MCQ_SUFFIX
        ),
        'description': '10:30 AM coffee availability',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E13', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'A typical adult is asked: "Can you meet me for lunch '
            'at noon on a Tuesday?"\n'
            'The most likely reply is:\n'
            '(A) "Sure, I have no fixed plans around that time."\n'
            '(B) "I can, but I only have about 30 minutes."\n'
            '(C) "Sorry, I cannot leave where I am at noon."\n'
            + MCQ_SUFFIX
        ),
        'description': 'Noon Tuesday availability',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E14', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'If you wanted to visit a typical adult at 1 PM '
            'on a Tuesday, which reply would be most likely?\n'
            '(A) "I am usually at home around then."\n'
            '(B) "I am usually at a regular daytime commitment then."\n'
            '(C) "It varies; I might be in different places."\n'
            + MCQ_SUFFIX
        ),
        'description': '1 PM Tuesday availability',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E15', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'At 3:00 PM on a Tuesday, a typical adult is most likely:\n'
            '(A) At home\n'
            '(B) At the same non-home location they have been all day\n'
            '(C) Moving between different locations\n'
            + MCQ_SUFFIX
        ),
        'description': '3 PM Tuesday location',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E16', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'At 5:00 PM on a Tuesday, a typical adult is most likely:\n'
            '(A) At home, where they have been most of the day\n'
            '(B) Heading home from a regular daytime location\n'
            '(C) Still at a non-home location with no plans to leave yet\n'
            + MCQ_SUFFIX
        ),
        'description': '5 PM Tuesday status',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'E17', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'At 7:00 PM on a Tuesday, a typical adult is most likely:\n'
            '(A) Has been home most of the day\n'
            '(B) Recently arrived home after being out all day\n'
            '(C) Still out, not yet headed home\n'
            + MCQ_SUFFIX
        ),
        'description': '7 PM Tuesday status',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'E18', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'Does a typical adult use an alarm clock on Tuesday mornings?\n'
            '(A) No, they wake up naturally whenever they feel like it\n'
            '(B) Sometimes, depending on whether they have plans\n'
            '(C) Yes, they set an alarm at the same time every Tuesday\n'
            + MCQ_SUFFIX
        ),
        'description': 'Alarm clock habit (Tue)',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'E19', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'Does a typical adult leave home at the same time '
            'every Tuesday?\n'
            '(A) No, the departure time varies a lot\n'
            '(B) Roughly similar, but not exact\n'
            '(C) Yes, within a few minutes of the same time\n'
            + MCQ_SUFFIX
        ),
        'description': 'Departure time consistency (Tue)',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'E20', 'type': 'mcq', 'exp': 'exp2',
        'prompt': (
            'Is a typical adult at the same non-home location '
            'at noon every Tuesday?\n'
            '(A) No, they are usually at home at noon\n'
            '(B) Sometimes the same place, sometimes not\n'
            '(C) Yes, almost always at the same location\n'
            + MCQ_SUFFIX
        ),
        'description': 'Noon location consistency (Tue)',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
]


# ══════════════════════════════════════════════════════════════════════
# Experiment 1a: Weekday vs Weekend
# ══════════════════════════════════════════════════════════════════════
# positive coeff = toward weekday, negative = toward weekend
# All users in this exp are EMPLOYED.
# NOTE: prompts intentionally do NOT specify day-of-week.
# "today" is kept vague so the steering vector can push the model's
# concept of "today" toward weekday or weekend behavior.

EXP1A_QUESTIONS = [
    # ══════════════════════════════════════
    # NUMERIC (10) — all timing/mobility-grounded
    # ══════════════════════════════════════
    {
        'id': 'W1', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Complete with a single number only.\n'
            'An individual wakes up at approximately ___:00 AM today.\n'
        ),
        'description': 'Wake-up hour',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W2', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Complete with a single number only.\n'
            'An individual leaves home at approximately ___:00 AM today.\n'
        ),
        'description': 'Leave-home hour',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W3', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Complete with a single number only.\n'
            'An individual arrives home at approximately ___:00 PM today.\n'
        ),
        'description': 'Arrive-home hour',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W4', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many hours does an individual spend at home '
            'between 8 AM and 6 PM today? ___\n'
        ),
        'description': 'Daytime hours at home',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W5', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'What is the longest continuous stretch of hours '
            'an individual spends outside the home today? ___\n'
        ),
        'description': 'Longest stretch outside',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W6', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many total hours is an individual away from home '
            'today? ___\n'
        ),
        'description': 'Total hours away',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W7', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many hours does an individual spend at their '
            'most frequently visited non-home location today? ___\n'
        ),
        'description': 'Hours at primary non-home location',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W8', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many times does an individual return home '
            'during the daytime (8 AM to 6 PM) today? ___\n'
        ),
        'description': 'Daytime home returns',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'W9', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'How many hours does an individual spend commuting '
            'or in transit today? ___\n'
        ),
        'description': 'Commute/transit hours',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'W10', 'type': 'numeric', 'exp': 'exp1a',
        'prompt': (
            'Answer with a single digit from 0 to 9.\n'
            'On a scale of 0 to 9, how consistent is an individual\'s '
            'schedule today compared to the same day last week? '
            '(0 = completely different, 9 = identical) ___\n'
        ),
        'description': 'Week-to-week consistency',
        'eval_method': 'digit_distribution', 'is_control': False,
        'strength': 'secondary',
    },

    # ══════════════════════════════════════
    # MCQ (10) — all time-anchored behavioral states
    # ══════════════════════════════════════
    {
        'id': 'W11', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'At 8:00 AM today, an individual is most likely:\n'
            '(A) Still sleeping or just waking up\n'
            '(B) Awake at home, starting the day slowly\n'
            '(C) Already out of the house or about to leave\n'
            + MCQ_SUFFIX
        ),
        'description': '8 AM status',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W12', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'An individual is asked: "Are you usually free to meet '
            'for coffee at 10:30 AM today?"\n'
            'The most likely reply is:\n'
            '(A) "Yes, I am usually available around that time."\n'
            '(B) "It depends; sometimes I can, sometimes not."\n'
            '(C) "No, I am almost always occupied at that hour."\n'
            + MCQ_SUFFIX
        ),
        'description': '10:30 AM coffee availability',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W13', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'An individual is asked: "Can you meet me for lunch '
            'at noon today?"\n'
            'The most likely reply is:\n'
            '(A) "Sure, I have no fixed plans around that time."\n'
            '(B) "I can, but I only have about 30 minutes."\n'
            '(C) "Sorry, I cannot leave where I am at noon."\n'
            + MCQ_SUFFIX
        ),
        'description': 'Noon availability',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W14', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'At 3:00 PM today, an individual is most likely:\n'
            '(A) At home\n'
            '(B) At the same non-home location they have been all day\n'
            '(C) Moving between different locations\n'
            + MCQ_SUFFIX
        ),
        'description': '3 PM location',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W15', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'At 5:00 PM today, an individual is most likely:\n'
            '(A) At home, where they have been most of the day\n'
            '(B) Heading home from a regular daytime location\n'
            '(C) Still at a non-home location with no plans to leave yet\n'
            + MCQ_SUFFIX
        ),
        'description': '5 PM status',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
    {
        'id': 'W16', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'At 7:00 PM today, an individual is most likely:\n'
            '(A) Has been home most of the day\n'
            '(B) Recently arrived home after being out all day\n'
            '(C) Still out, not yet headed home\n'
            + MCQ_SUFFIX
        ),
        'description': '7 PM status',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'W17', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'Does an individual use an alarm clock this morning?\n'
            '(A) No, they wake up naturally\n'
            '(B) Maybe, depending on plans\n'
            '(C) Yes, set for a specific early time\n'
            + MCQ_SUFFIX
        ),
        'description': 'Alarm clock usage',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'W18', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'Does an individual leave home at the same time today '
            'as they did the same day last week?\n'
            '(A) No, the departure time is very different\n'
            '(B) Roughly similar, but not exact\n'
            '(C) Yes, within a few minutes of the same time\n'
            + MCQ_SUFFIX
        ),
        'description': 'Departure time consistency',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'W19', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'Is an individual at the same non-home location '
            'at noon today as they were at noon last week?\n'
            '(A) No, they are at home today\n'
            '(B) They are out, but at a different location\n'
            '(C) Yes, at the same location\n'
            + MCQ_SUFFIX
        ),
        'description': 'Noon location consistency',
        'target_option_pos': 'C', 'target_option_neg': 'A', 'target_option_neu': 'B',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'medium',
    },
    {
        'id': 'W20', 'type': 'mcq', 'exp': 'exp1a',
        'prompt': (
            'If an individual is asked at 1 PM today: '
            '"Where are you right now?"\n'
            'The most likely answer is:\n'
            '(A) "I am at home."\n'
            '(B) "I am at a regular place I go to most days."\n'
            '(C) "I am out, but somewhere different from usual."\n'
            + MCQ_SUFFIX
        ),
        'description': '1 PM location',
        'target_option_pos': 'B', 'target_option_neg': 'A', 'target_option_neu': 'C',
        'eval_method': 'mcq', 'is_control': False,
        'strength': 'strong',
    },
]


ALL_QUESTIONS = {
    'exp2': EXP2_QUESTIONS,
    'exp1a': EXP1A_QUESTIONS,
}

MCQ_TOKENS = ['A', 'B', 'C']
DIGIT_TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
