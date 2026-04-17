#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download GemmaScope 2 SAE weights to local disk.
Run on LOGIN NODE (has internet).

Usage:
    python download_sae.py --layers 12 24 31 41
    python download_sae.py --layers 41 --widths 65k 16k
"""
import os, argparse
from sae_lens import SAE

SAE_CACHE = "/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7/sae_cache"

# sae_lens has two possible release name formats — try both
RELEASE_FORMATS = [
    ("gemma-scope-2-12b-it-resid_post",
     "layer_{layer}_width_{width}_l0_{l0}"),
    ("google/gemma-scope-2-12b-it",
     "resid_post/layer_{layer}_width_{width}_l0_{l0}"),
]


def download_one(gs_layer, width="65k", l0="medium"):
    save_dir = os.path.join(SAE_CACHE, f"layer_{gs_layer}_width_{width}")
    if os.path.exists(os.path.join(save_dir, "cfg.json")):
        print(f"  Already exists: {save_dir}")
        return save_dir

    os.makedirs(save_dir, exist_ok=True)

    for release_fmt, sae_id_fmt in RELEASE_FORMATS:
        release = release_fmt
        sae_id = sae_id_fmt.format(layer=gs_layer, width=width, l0=l0)
        try:
            print(f"  Trying: release={release}, sae_id={sae_id}")
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=release, sae_id=sae_id, device="cpu",
            )
            sae.save_to_disk(save_dir)
            print(f"  OK → {save_dir}  (d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae})")
            return save_dir
        except (ValueError, Exception) as e:
            print(f"    Failed: {e}")
            continue

    print(f"  ERROR: could not download layer {gs_layer} width {width}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[12, 24, 31, 41])
    parser.add_argument("--widths", nargs="+", default=["65k"])
    parser.add_argument("--l0", default="medium")
    args = parser.parse_args()

    os.makedirs(SAE_CACHE, exist_ok=True)
    print(f"Cache dir: {SAE_CACHE}\n")

    for layer in args.layers:
        for width in args.widths:
            print(f"Layer {layer}, width {width}:")
            download_one(layer, width, args.l0)
            print()

    print("Done. Run sae_feature_check.py on compute node.")


if __name__ == "__main__":
    main()
