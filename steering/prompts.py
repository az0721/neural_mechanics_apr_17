# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """Prompt templates for steering experiments."""

# STEERING_PROMPTS = {
#     'binary': {
#         'question': (
#             "=== Question ===\n"
#             "Based on the mobility pattern shown above, is this person "
#             "currently employed?\n"
#             "Answer with only Yes or No:"
#         ),
#         'targets': ['Yes', 'No'],
#         'employed_target': 'Yes',
#     },
#     'behavioral': {
#         'question': (
#             "=== Question ===\n"
#             "Based on this person's typical daily mobility pattern, at 12:00 "
#             "on a typical weekday, where is this person most likely?\n"
#             "(A) At home or near home\n"
#             "(B) At a workplace or office area\n"
#             "(C) Traveling between locations\n"
#             "Answer with only the letter:"
#         ),
#         'targets': ['A', 'B', 'C'],
#         'employed_target': 'B',
#     },
#     'routine': {
#         'question': (
#             "=== Question ===\n"
#             "This person's daily mobility pattern most closely resembles:\n"
#             "(A) Someone with a regular 9-to-5 job who commutes daily\n"
#             "(B) Someone who stays home most of the day without regular commute\n"
#             "(C) Someone with an irregular or part-time schedule\n"
#             "Answer with only the letter:"
#         ),
#         'targets': ['A', 'B', 'C'],
#         'employed_target': 'A',
#     },
#     'location': {
#         'question': (
#             "=== Prediction ===\n"
#             "Question: At 12:00, what location will this person be at?\n"
#             "Answer:"
#         ),
#         'targets': [],  # no fixed targets — check top-10
#         'employed_target': None,
#     },
# }
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt templates for steering experiments — Phase 2b.

4 prompts:
  behavioral:        12:00 location (A/B/C), EDA gap=0.75
  binary:            employed? (Yes/No), control/baseline
  departure:         left home before 8AM? (A/B), EDA gap=0.55
  return_prediction: still away at noon? (A/B), EDA gap=0.75
"""

STEERING_PROMPTS = {
    'behavioral': {
        'question': (
            "=== Question ===\n"
            "Based on this person's typical daily mobility pattern, at 12:00 "
            "on a typical weekday, where is this person most likely?\n"
            "(A) At home or near home\n"
            "(B) At a workplace or office area\n"
            "(C) Traveling between locations\n"
            "Answer with only the letter:"
        ),
        'targets': ['A', 'B', 'C'],
        'employed_target': 'B',
    },
    'binary': {
        'question': (
            "=== Question ===\n"
            "Based on the mobility pattern shown above, is this person "
            "currently employed?\n"
            "Answer with only Yes or No:"
        ),
        'targets': ['Yes', 'No'],
        'employed_target': 'Yes',
    },
    'departure': {
        'question': (
            "=== Question ===\n"
            "Based on this person's morning mobility data, did this person "
            "leave their home location before 8:00 AM?\n"
            "(A) Yes - this person left home before 8:00 AM\n"
            "(B) No - this person was still at or near home at 8:00 AM\n"
            "Answer with only the letter:"
        ),
        'targets': ['A', 'B'],
        'employed_target': 'A',
    },
    'return_prediction': {
        'question': (
            "=== Question ===\n"
            "Based on this person's mobility history, will this person "
            "likely return to their morning location by 12:00?\n"
            "(A) Yes - likely back at or near their morning location by noon\n"
            "(B) No - likely still away from their morning location at noon\n"
            "Answer with only the letter:"
        ),
        'targets': ['A', 'B'],
        'employed_target': 'B',
    },
}
