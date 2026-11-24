"""Prove the LoRA adapter is attached to the right base modules with the right weights.

Loads the base model exactly as the inference engine does, attaches
`checkpoint-4`, and prints the four independent verification checks described in
`ocr/engine.py:attach_adapter`. No generation is performed, so this is cheap and
can be run before any evaluation.

Usage:
    python scripts/verify_adapter.py --adapter outputs/qlora-v1/.../checkpoint-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.engine import DEFAULT_BASE, OCREngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default="outputs/protocol/adapter_check.json")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    eng = OCREngine(base_model=args.base, adapter_dir=args.adapter,
                    load_in_4bit=not args.no_4bit)
    eng._load()
    meta = eng._adapter_meta

    # sanity: PEFT wrapper must exist and be frozen / in eval mode
    import torch
    meta["is_peft_model"] = type(eng._model).__name__
    meta["n_named_parameters"] = sum(1 for _ in eng._model.named_parameters())
    meta["any_param_training"] = any(m.training for m in eng._model.modules())
    meta["cuda_available"] = torch.cuda.is_available()

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nWROTE", args.out)
    print("weights_verified =", meta["weights_verified"])


if __name__ == "__main__":
    main()
