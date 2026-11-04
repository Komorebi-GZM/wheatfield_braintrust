"""Build train/val/test splits and export to ms-swift & LLaMA-Factory formats.

Rules (from the model design):
  - Real pages: sorted by filename, last `--test-pages` -> test, previous
    `--val-pages` -> val, remainder -> real train.
  - Synthetic pages: train only (never val/test) — they are weak supervision.
  - Split by PAGE, never split lines of the same page across sets.

Inputs:
  --real-dir      dir of real page images
  --real-ann      jsonl: {"image": <path or name>, "markdown": "..."}
  --synth-manifest jsonl from build_synth.py
Outputs in --out-dir:
  train.jsonl / val.jsonl / test.jsonl   (split manifests, reference format)
  train_swift.jsonl                      (ms-swift multimodal SFT)
  train_sharegpt.json                    (LLaMA-Factory sharegpt)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Dict

USER_INSTRUCTION = "<image>\n请转写这页手写课程笔记为结构化 Markdown（保留标题、列表、表格与公式）。"


def _read_jsonl(path: str) -> List[Dict]:
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _rebase(items: List[Dict], base: str) -> None:
    """Make image paths absolute relative to their owning base dir (in place)."""
    for it in items:
        p = str(it["image"])
        if not os.path.isabs(p):
            it["image"] = os.path.abspath(os.path.join(base, p))


def _to_swift(item: Dict) -> Dict:
    return {
        "messages": [
            {"role": "user", "content": USER_INSTRUCTION},
            {"role": "assistant", "content": item["markdown"]},
        ],
        "images": [item["image"]],
    }


def _to_sharegpt(item: Dict) -> Dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>" + USER_INSTRUCTION.replace("<image>", "")},
            {"role": "assistant", "content": item["markdown"]},
        ],
        "images": [item["image"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="data/raw")
    ap.add_argument("--real-ann", default="data/annotations/real.jsonl")
    ap.add_argument("--synth-manifest", default="data/synth/manifest.jsonl")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--val-pages", type=int, default=5)
    ap.add_argument("--test-pages", type=int, default=5)
    args = ap.parse_args()

    real = _read_jsonl(args.real_ann)
    synth = _read_jsonl(args.synth_manifest)
    real.sort(key=lambda x: str(x.get("image")))
    os.makedirs(args.out_dir, exist_ok=True)

    # Rebase image paths to absolute relative to their own base dirs.
    _rebase(real, args.real_dir)
    _rebase(synth, os.path.dirname(args.synth_manifest) or ".")

    n_test = min(args.test_pages, max(0, len(real) - 1))
    n_val = min(args.val_pages, max(0, len(real) - n_test))
    test = real[len(real) - n_test:] if n_test else []
    val = real[len(real) - n_test - n_val:len(real) - n_test] if n_val else []
    train_real = real[:len(real) - n_test - n_val] if (n_test or n_val) else real
    train = train_real + synth

    def dump(name, items):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump("train.jsonl", train)
    dump("val.jsonl", val)
    dump("test.jsonl", test)

    with open(os.path.join(args.out_dir, "train_swift.jsonl"), "w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(_to_swift(it), ensure_ascii=False) + "\n")

    with open(os.path.join(args.out_dir, "train_sharegpt.json"), "w", encoding="utf-8") as f:
        json.dump([_to_sharegpt(it) for it in train], f, ensure_ascii=False, indent=1)

    print(f"[prepare_dataset] real={len(real)} synth={len(synth)} "
          f"-> train={len(train)} val={len(val)} test={len(test)}")
    print(f"[prepare_dataset] NOTE: val/test 仅含真实页（合成页只进 train）。输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
