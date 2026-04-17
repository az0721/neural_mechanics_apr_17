#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapt all downstream analysis scripts for multi-model (Gemma 3 + Gemma 4) support.

This script modifies each file to:
  1. Accept --model and --iter arguments
  2. Use get_iter_model_dirs() instead of hardcoded paths
  3. Save outputs to the correct model-specific directory
  4. Print output locations

DRY RUN by default. Use --apply to actually modify files.

Usage:
    python adapt_scripts_multimodel.py --dry-run    # show what would change
    python adapt_scripts_multimodel.py --apply       # apply changes
"""
import os, re, sys, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════
# Files to adapt and their specific changes
# ══════════════════════════════════════════════════════════════════════

SCRIPTS = {
    # ── Group 1: Core analysis (read hidden_states / answers) ──
    # 'prob_results.py': {
    #     'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
    #     'import_remove': ['OUTPUT_DIR'],
    #     'add_args': ['model', 'iter'],
    #     'dirs_pattern': r"get_model_dirs\(([^)]*)\)",
    #     'dirs_replace': "get_iter_model_dirs({model}, {iter})",
    # },
    # 'prediction_results.py': {
    #     'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
    #     'add_args': ['model', 'iter'],
    #     'note': 'Also fix load_answers to use iter-aware path',
    # },
    'geometry_v7_v8.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'compare_v7_v8.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model'],
        'note': 'Already compares v7 vs v8, just needs --model',
    },
    'run_pca_da.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'run_direction_analysis.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'run_tsne_umap.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },

    # ── Group 2: Steering ──
    'compute_vectors.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'compute_vectors_nl.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'run_steering_per_user_v8.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
        'note': 'Also needs load_model(model_key) update',
    },
    'run_steering_nl.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'analyze_steering_v8.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'analyze_steering_nl.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'generate_steering_jobs_v8.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'generate_steering_jobs_nl.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model', 'iter'],
    },
    'prebuild_steering_prompts_v8.py': {
        'import_add': 'get_iter_model_dirs, MODEL_REGISTRY',
        'add_args': ['model'],
    },

    # prompts_nl.py: No changes needed (just prompt definitions)
}

# ══════════════════════════════════════════════════════════════════════
# The universal patch: add --model and --iter to argparse
# ══════════════════════════════════════════════════════════════════════

ARGPARSE_MODEL = """    parser.add_argument('--model', default='12b',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model key (default: 12b)')"""

ARGPARSE_ITER = """    parser.add_argument('--iter', default='v7',
                        choices=['v7', 'v8'],
                        help='Iteration (default: v7)')"""


def patch_file(filepath, spec, dry_run=True):
    """Apply multi-model patches to one script."""
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False

    with open(filepath) as f:
        code = f.read()

    original = code
    changes = []

    # 1. Add MODEL_REGISTRY import if missing
    if 'MODEL_REGISTRY' not in code:
        # Find the config import line and extend it
        config_import = re.search(
            r'from config import \(([^)]+)\)', code, re.DOTALL)
        if config_import:
            old_imports = config_import.group(1)
            adds = spec.get('import_add', 'get_iter_model_dirs, MODEL_REGISTRY')
            for add in adds.split(', '):
                if add.strip() not in old_imports:
                    old_imports = old_imports.rstrip().rstrip(',') + f',\n                    {add.strip()}'
            new_line = f"from config import ({old_imports})"
            code = code[:config_import.start()] + new_line + code[config_import.end():]
            changes.append(f"  + Added imports: {adds}")
        else:
            # Try single-line import
            m = re.search(r'from config import (.+)', code)
            if m:
                old = m.group(1)
                adds = spec.get('import_add', 'get_iter_model_dirs, MODEL_REGISTRY')
                for add in adds.split(', '):
                    if add.strip() not in old:
                        old += f', {add.strip()}'
                code = code[:m.start()] + f'from config import {old}' + code[m.end():]
                changes.append(f"  + Added imports: {adds}")

    # 2. Add --model and --iter to argparse
    args_to_add = spec.get('add_args', ['model', 'iter'])

    if 'model' in args_to_add and '--model' not in code:
        # Find parser.add_argument or parser.parse_args
        m = re.search(r'(    args = parser\.parse_args\(\))', code)
        if m:
            insert = ARGPARSE_MODEL + '\n'
            if 'iter' in args_to_add and '--iter' not in code:
                insert += ARGPARSE_ITER + '\n'
            code = code[:m.start()] + insert + m.group(1) + code[m.end():]
            changes.append("  + Added --model and --iter arguments")

    # 3. Replace get_model_dirs → get_iter_model_dirs
    if 'get_model_dirs' in code and 'get_iter_model_dirs' not in code:
        # Pattern: dirs = get_model_dirs(args.model) or get_model_dirs('12b')
        code = re.sub(
            r'get_model_dirs\(([^)]*)\)',
            lambda m: f'get_iter_model_dirs({m.group(1)}, args.iter)',
            code)
        changes.append("  + Replaced get_model_dirs → get_iter_model_dirs")

    # 4. Replace OUTPUT_DIR with iter-aware path where used for results
    if 'OUTPUT_DIR' in code and 'get_iter_output_dir' not in code:
        # Add import
        if 'get_iter_output_dir' not in code:
            code = code.replace(
                'from config import',
                'from config import get_iter_output_dir,', 1)
        changes.append("  + Added get_iter_output_dir import")

    if code == original:
        print(f"  NO CHANGES: {filepath}")
        return False

    if dry_run:
        print(f"  WOULD PATCH: {filepath}")
        for c in changes:
            print(c)
        if spec.get('note'):
            print(f"  NOTE: {spec['note']}")
        return True

    # Backup and write
    backup = filepath + '.bak'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
    with open(filepath, 'w') as f:
        f.write(code)
    print(f"  PATCHED: {filepath}")
    for c in changes:
        print(c)
    if spec.get('note'):
        print(f"  ⚠ MANUAL CHECK: {spec['note']}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                        help='Actually modify files (default: dry run)')
    parser.add_argument('--dry-run', action='store_true', default=True)
    args = parser.parse_args()

    dry_run = not args.apply

    print("=" * 60)
    print(f"Multi-Model Script Adapter {'(DRY RUN)' if dry_run else '(APPLYING)'}")
    print("=" * 60)

    n_patched = 0
    for script, spec in SCRIPTS.items():
        filepath = os.path.join(BASE_DIR, script)
        # Check in subdirectories too
        if not os.path.exists(filepath):
            for sub in ['steering', 'scripts']:
                alt = os.path.join(BASE_DIR, sub, script)
                if os.path.exists(alt):
                    filepath = alt
                    break
        if patch_file(filepath, spec, dry_run):
            n_patched += 1

    print(f"\n{'='*60}")
    print(f"{'Would patch' if dry_run else 'Patched'}: {n_patched}/{len(SCRIPTS)} files")
    if dry_run:
        print(f"\nTo apply: python {os.path.basename(__file__)} --apply")
    else:
        print(f"\nBackups saved as *.bak files")
        print(f"\nUsage pattern for ALL scripts:")
        print(f"  python <script>.py --model gemma4_31b --iter v7")
        print(f"  python <script>.py --model 12b --iter v8  (Gemma 3 unchanged)")
    print("=" * 60)


if __name__ == "__main__":
    main()
