#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-build trajectory generation prompts for Phase 2c (Iter 8).
UPDATED: Stronger format instructions + few-shot example + 14 days default.

Changes from previous version:
  - DEFAULT_MAX_DAYS: 28 → 14 (reduces ~72K → ~36K tokens)
  - Added explicit output format rules with example lines
  - Seeds first output line to anchor the generation format
  - Added "Output ONLY data lines. No text, no explanation."

Output: data/steering_prompts_v8/{uid[:32]}.json

Usage:
    python steering/prebuild_steering_prompts_v8.py
    python steering/prebuild_steering_prompts_v8.py --max-days 14
    python steering/prebuild_steering_prompts_v8.py --max-days 28 --force
"""
import sys, os, argparse, time, json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, get_iter_model_dirs, MODEL_REGISTRY

PROMPT_DIR = os.path.join(DATA_DIR, 'steering_prompts_v8')
os.makedirs(PROMPT_DIR, exist_ok=True)

DOW_NAMES = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
             5: 'Thursday', 6: 'Friday', 7: 'Saturday'}

TARGET_DAY = 'Tuesday'
TARGET_DATE = 'Mar 03, 2020'

HOUR_MIN = 5
HOUR_MAX = 23

DEFAULT_MAX_DAYS = 14  # 2 weeks (~36K tokens vs 72K for 28 days)


def build_prompt(user_df, uid, home_geo, work_geo, max_days):
    """Build trajectory generation prompt with strong format constraints."""
    user_df = user_df[user_df['min5'] % 15 == 0].copy()
    user_df = user_df[(user_df['hour'] >= HOUR_MIN) &
                      (user_df['hour'] <= HOUR_MAX)]
    user_df = user_df.sort_values(['date', 'hour', 'min5'])

    dates = sorted(user_df['date'].unique())
    if len(dates) > max_days:
        dates = dates[-max_days:]
        user_df = user_df[user_df['date'].isin(dates)]

    hist_lines = []
    for _, r in user_df.iterrows():
        dow = DOW_NAMES.get(int(r['dow']), 'Unknown')
        date_str = (pd.Timestamp(r['date']).strftime('%b %d, %Y')
                    if '-' in str(r['date']) else r['date'])
        time_str = f"{int(r['hour']):02d}:{int(r['min5']):02d}"
        geo = str(r['geo_id'])
        hist_lines.append(f"{dow}, {date_str}, {time_str}, {geo}")

    seed_geo = home_geo

    prompt = (
        f"You are given the location history of an individual.\n"
        f"Each record has exactly 4 fields: day_of_week, date, time, location_id\n\n"
        f"=== Location History ===\n"
        + "\n".join(hist_lines) + "\n\n"
        f"=== Task ===\n"
        f"Generate a plausible full-day location trajectory for this person "
        f"on {TARGET_DAY}, {TARGET_DATE}, from 05:00 to 23:00 "
        f"at 15-minute intervals.\n\n"
        f"RULES:\n"
        f"- Output ONLY data lines, one per time slot\n"
        f"- Each line: day_of_week, date, time, location_id\n"
        f"- Use only location_ids from the history above\n"
        f"- No explanation, no code, no commentary\n"
        f"- Exactly 73 lines (05:00 to 23:00 at 15-min intervals)\n\n"
        f"Example (first 3 lines of correct output):\n"
        f"Tuesday, {TARGET_DATE}, 05:00, {seed_geo}\n"
        f"Tuesday, {TARGET_DATE}, 05:15, {seed_geo}\n"
        f"Tuesday, {TARGET_DATE}, 05:30, {seed_geo}\n\n"
        f"Now generate all 73 lines:\n"
        f"Tuesday, {TARGET_DATE}, 05:00, {seed_geo}\n"
        f"Tuesday, {TARGET_DATE}, 05:15,"
    )

    return prompt, len(dates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-days', type=int, default=DEFAULT_MAX_DAYS)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model key (default: 12b)')
    args = parser.parse_args()

    t0 = time.time()
    steer_users = os.path.join(DATA_DIR, 'steering_users_v2.csv')
    steer_traj = os.path.join(DATA_DIR, 'steering_trajectories_v2.csv')

    if not os.path.exists(steer_users) or not os.path.exists(steer_traj):
        print(f"ERROR: Missing data files"); return

    users = pd.read_csv(steer_users)
    traj = pd.read_csv(steer_traj)

    print(f"Pre-building steering v8 prompts → {PROMPT_DIR}")
    print(f"  Users: {len(users)}, Max days: {args.max_days}")

    n_built = 0
    for _, urow in users.iterrows():
        uid = urow['cuebiq_id']
        home_geo = str(urow['home_geo_id'])
        work_geo = str(urow['work_geo_id'])

        out_path = os.path.join(PROMPT_DIR, f"{uid[:32]}.json")
        if os.path.exists(out_path) and not args.force:
            n_built += 1; continue

        user_traj = traj[traj['cuebiq_id'] == uid]
        if len(user_traj) == 0:
            print(f"  {uid[:16]}: no data, skip"); continue

        prompt, n_days = build_prompt(user_traj, uid, home_geo, work_geo,
                                      args.max_days)
        est_tokens = len(prompt) // 4

        save_data = {
            'meta': {
                'user': uid, 'home_geo_id': home_geo, 'work_geo_id': work_geo,
                'n_days': n_days, 'est_tokens': est_tokens,
                'max_days': args.max_days,
            },
            'prompt': prompt,
        }
        with open(out_path, 'w') as f:
            json.dump(save_data, f)
        print(f"  {uid[:16]}: {n_days}d, ~{est_tokens//1000}K tok")
        n_built += 1

    print(f"Done: {n_built} prompts, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()