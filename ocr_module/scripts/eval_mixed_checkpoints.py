"""Evaluate every checkpoint of a mixed-training run on the in-domain val pages.

For each `checkpoint-*` under a run directory: run inference with protocol v4
deliver defaults (repetition_penalty 1.1, min_new_tokens 0, cap 1024) on the
fixed val manifest, then report text CER / CER-with-math / repetition / cap hits.
Picks the best by text CER.

    python scripts/eval_mixed_checkpoints.py --run outputs/qlora-mixed/mixed-x600-20261126-121254
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import cer, plain_text  # noqa: E402
from scripts.eval_supplement import text_with_math  # noqa: E402

VAL = "datasets/handwriting_notes/splits/val.jsonl"


def _read(path: Path):
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            out[os.path.basename(rec["image"])] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", default=VAL)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    ckpts = sorted([d for d in run.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[1]))
    if not ckpts:
        print("no checkpoints found")
        return 1

    refs = {os.path.basename(r["image"]): r["markdown"] for r in
            (json.loads(l) for l in open(ROOT / args.manifest, encoding="utf-8") if l.strip())}

    out_root = ROOT / "outputs/eval-mixed" / run.name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for ck in ckpts:
        pred_path = out_root / f"{ck.name}.jsonl"
        if not (args.skip_existing and pred_path.exists()):
            cmd = [sys.executable, str(ROOT / "scripts/infer_test_baseline.py"),
                   "--manifest", str(ROOT / args.manifest),
                   "--output", str(pred_path),
                   "--adapter", str(ck),
                   "--max-new-tokens", str(args.max_new_tokens)]
            print(f"[{ck.name}] running inference ...", flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                print(f"[{ck.name}] FAILED rc={proc.returncode}\n{proc.stderr[-800:]}")
                continue
        preds = _read(pred_path)
        if not preds:
            print(f"[{ck.name}] no predictions")
            continue
        cers, cerms, reps, caps = [], [], 0, 0
        for name, ref in refs.items():
            rec = preds.get(name)
            if not rec:
                continue
            md = rec["markdown"]
            cers.append(cer(plain_text(md), plain_text(ref)))
            cerms.append(cer(text_with_math(md), text_with_math(ref)))
            reps += 1 if rec["run"]["repetition"]["detected"] else 0
            caps += 1 if rec["run"]["generation"]["finish_reason"] == "length" else 0
        if not cers:
            continue
        rows.append({
            "checkpoint": ck.name,
            "pages": len(cers),
            "cer": sum(cers) / len(cers),
            "cer_math": sum(cerms) / len(cerms),
            "rep_pages": reps,
            "cap_pages": caps,
            "pred_file": str(pred_path),
        })

    if not rows:
        print("no evaluable checkpoints")
        return 1

    rows.sort(key=lambda r: r["cer"])
    print("\n%-14s %5s %9s %11s %8s %7s" % ("checkpoint", "pages", "text_cer", "cer_math", "rep", "cap"))
    for r in rows:
        print("%-14s %5d %9.4f %11.4f %8d %7d" % (
            r["checkpoint"], r["pages"], r["cer"], r["cer_math"], r["rep_pages"], r["cap_pages"]))
    print("\nbest by text CER:", rows[0]["checkpoint"])
    (out_root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                           encoding="utf-8", newline="\n")
    print("wrote", out_root / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
