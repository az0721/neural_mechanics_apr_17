#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix multi-model path issues in downstream analysis scripts.

Bug 1: prediction_results.py::load_answers() never reads/returns the file
Bug 2: geometry scripts hardcode MODEL_REGISTRY['12b'] and BASE_DIR paths

Usage:
    python fix_multimodel_paths.py --dry-run   # preview changes
    python fix_multimodel_paths.py --apply      # apply fixes
"""
import os, sys, shutil, argparse

# ═══════════════════════════════════════════════════════════════
# Fix 1: prediction_results.py — load_answers() missing body
# ═══════════════════════════════════════════════════════════════

PRED_OLD = '''def load_answers(iter_name, model_key, exp_name, config_id):
    """Load answers JSON for a specific iteration."""
    from config import get_iter_output_dir
    tag = MODEL_REGISTRY[model_key]['tag']
    base = get_iter_output_dir(model_key, iter_name)
    path = os.path.join(base, 'answers', tag,
                        f"{exp_name}_cfg{config_id}_answers.json")'''

PRED_NEW = '''def load_answers(iter_name, model_key, exp_name, config_id):
    """Load answers JSON for a specific iteration."""
    from config import get_iter_output_dir
    tag = MODEL_REGISTRY[model_key]['tag']
    base = get_iter_output_dir(model_key, iter_name)
    path = os.path.join(base, 'answers', tag,
                        f"{exp_name}_cfg{config_id}_answers.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)'''

# ═══════════════════════════════════════════════════════════════
# Fix 2: geometry scripts — load_hidden() hardcoded to 12b
#
# All three scripts have the SAME broken load_hidden():
#   def load_hidden(iter_name, exp, cfg_id):
#       tag = MODEL_REGISTRY['12b']['tag']
#       path = os.path.join(BASE_DIR, f'outputs_{iter_name}', ...)
#
# Fix: accept model_key, use get_iter_model_dirs() from config
# ═══════════════════════════════════════════════════════════════

GEO_LOAD_OLD = '''def load_hidden(iter_name, exp, cfg_id):
    tag = MODEL_REGISTRY['12b']['tag']
    path = os.path.join(BASE_DIR, f'outputs_{iter_name}', 'hidden_states',
                        tag, f'{exp}_cfg{cfg_id}.npz')
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels']'''

GEO_LOAD_NEW = '''def load_hidden(iter_name, exp, cfg_id, model_key='12b'):
    from config import get_iter_model_dirs
    dirs = get_iter_model_dirs(model_key, iter_name)
    path = os.path.join(dirs['hidden'], f'{exp}_cfg{cfg_id}.npz')
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return d['hidden_states'], d['labels']'''

# Also need to fix the call sites: load_hidden(args.iter, args.exp, cid)
# → load_hidden(args.iter, args.exp, cid, args.model)

GEO_CALL_OLD = 'X, y = load_hidden(args.iter, args.exp, cid)'
GEO_CALL_NEW = 'X, y = load_hidden(args.iter, args.exp, cid, args.model)'

# Also fix output directory for geometry scripts
# Currently: OUT_DIR = os.path.join(BASE_DIR, 'geometry', 'outputs')
# Should be model-aware

GEO_OUTDIR_OLD = "OUT_DIR = os.path.join(BASE_DIR, 'geometry', 'outputs')"
# We'll handle this differently: make OUT_DIR dynamic in main()

# For PCA/DA and direction analysis, also fix output path in main()
# The out_path is defined in main() and should be model-aware

# ═══════════════════════════════════════════════════════════════
# Apply fixes
# ═══════════════════════════════════════════════════════════════

def fix_file(filepath, replacements, dry_run=True):
    """Apply a list of (old, new) replacements to a file."""
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False

    with open(filepath) as f:
        code = f.read()

    original = code
    applied = []

    for old, new, desc in replacements:
        if old in code:
            code = code.replace(old, new, 1)
            applied.append(desc)
        else:
            # Check if already fixed
            if new in code:
                print(f"  ALREADY FIXED: {desc}")
            else:
                print(f"  WARNING: pattern not found for: {desc}")

    if code == original:
        print(f"  NO CHANGES NEEDED: {filepath}")
        return False

    if dry_run:
        print(f"  WOULD FIX: {filepath}")
        for a in applied:
            print(f"    ✓ {a}")
        return True

    # Backup and write
    backup = filepath + '.prefixfix.bak'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"  Backup: {backup}")

    with open(filepath, 'w') as f:
        f.write(code)
    print(f"  FIXED: {filepath}")
    for a in applied:
        print(f"    ✓ {a}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--base-dir', default='.', help='Project base directory')
    args = parser.parse_args()
    dry_run = not args.apply
    base = os.path.abspath(args.base_dir)

    print("=" * 70)
    print(f"Multi-Model Path Fix {'(DRY RUN)' if dry_run else '(APPLYING)'}")
    print(f"Base: {base}")
    print("=" * 70)

    n_fixed = 0

    # ── Fix 1: prediction_results.py ──
    print(f"\n[1/4] prediction_results.py")
    pred_path = os.path.join(base, 'prediction_results.py')
    if fix_file(pred_path, [
        (PRED_OLD, PRED_NEW, "load_answers(): add file read + return"),
    ], dry_run):
        n_fixed += 1

    # ── Fix 2-4: geometry scripts ──
    geo_scripts = [
        ('run_pca_da.py', 'geometry'),
        ('run_tsne_umap.py', 'geometry'),
        ('run_direction_analysis.py', 'geometry'),
    ]

    for i, (script, subdir) in enumerate(geo_scripts, 2):
        print(f"\n[{i}/4] {subdir}/{script}")
        # Try both locations
        script_path = os.path.join(base, subdir, script)
        if not os.path.exists(script_path):
            script_path = os.path.join(base, script)

        if fix_file(script_path, [
            (GEO_LOAD_OLD, GEO_LOAD_NEW,
             "load_hidden(): use model-aware paths via get_iter_model_dirs()"),
            (GEO_CALL_OLD, GEO_CALL_NEW,
             "load_hidden() call: pass args.model"),
        ], dry_run):
            n_fixed += 1

    print(f"\n{'=' * 70}")
    print(f"{'Would fix' if dry_run else 'Fixed'}: {n_fixed}/4 files")
    if dry_run:
        print(f"\nTo apply: python {os.path.basename(__file__)} --apply --base-dir {base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
