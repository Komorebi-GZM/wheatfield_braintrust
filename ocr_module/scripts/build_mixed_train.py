"""Build the mixed-training manifest from HANDOFF §20.

Mix (first round, as agreed with the remote side):
  * main domain: 13 pages x 8 oversample  (~104 samples/epoch, keeps gradient lead)
  * CASIA-HWDB2-line train: sample 8,000 lines (~24% of steps)
  * GNHK train: sample 80 pages (~18%)
  * validation: unchanged `val_swift.jsonl` (domain only, no external data)

Output is a Swift-format jsonl (`messages` + `images`) so the existing
`scripts/train_qlora_v2.ps1` can consume it via `--dataset`.

Usage:
    python scripts/build_mixed_train.py --check     # verify data availability, write nothing
    python scripts/build_mixed_train.py             # write outputs/mixed/train_mixed_swift.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.prompts import INSTRUCTION  # noqa: E402

MAIN = "datasets/handwriting_notes/splits/train_swift.jsonl"
VAL = "datasets/handwriting_notes/splits/val_swift.jsonl"
CASIA = "datasets/external/casia_hwdb2/casia_train.jsonl"
GNHK = "datasets/external/gnhk/gnhk_train.jsonl"
CASIA_DIR = "datasets/external/casia_hwdb2"
GNHK_DIR = "datasets/external/gnhk"


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve(base: Path, rel: str) -> Path:
    return (base / rel).resolve()


def _swift_row(image: Path, markdown: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>\n" + INSTRUCTION},
            {"role": "assistant", "content": markdown},
        ],
        "images": [str(image)],
    }


def survey(rows, base: Path):
    present = [r for r in rows if _resolve(base, r["image"]).is_file()]
    missing = [r for r in rows if not _resolve(base, r["image"]).is_file()]
    return present, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Build HANDOFF §20 mixed-training manifest.")
    ap.add_argument("--check", action="store_true", help="只体检，不写文件")
    ap.add_argument("--main-repeat", type=int, default=8)
    ap.add_argument("--casia-lines", type=int, default=8000)
    ap.add_argument("--gnhk-pages", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/mixed/train_mixed_swift.jsonl")
    ap.add_argument("--allow-missing", action="store_true",
                    help="允许缺失图片（仅用于统计；真正训练不要加）")
    args = ap.parse_args()

    main_rows = _read_jsonl(ROOT / MAIN)
    val_rows = _read_jsonl(ROOT / VAL)
    casia_rows = _read_jsonl(ROOT / CASIA)
    gnhk_rows = _read_jsonl(ROOT / GNHK)

    casia_present, casia_missing = survey(casia_rows, ROOT / CASIA_DIR)
    gnhk_present, gnhk_missing = survey(gnhk_rows, ROOT / GNHK_DIR)

    print("source              rows    images_present  images_missing")
    print("main train          %4d    %4d            %4d" % (len(main_rows), len(main_rows), 0))
    print("main val            %4d    %4d            %4d" % (len(val_rows), len(val_rows), 0))
    print("casia train         %4d    %4d            %4d" % (len(casia_rows), len(casia_present), len(casia_missing)))
    print("gnhk train          %4d    %4d            %4d" % (len(gnhk_rows), len(gnhk_present), len(gnhk_missing)))

    if args.check:
        ok = (len(casia_present) >= args.casia_lines and
              len(gnhk_present) >= args.gnhk_pages)
        print("\nCHECK:", "data ready" if ok else "BLOCKED — not enough extracted images")
        if not ok:
            print("  need %d CASIA lines (have %d) and %d GNHK pages (have %d)"
                  % (args.casia_lines, len(casia_present), args.gnhk_pages, len(gnhk_present)))
            print("  extract with: python scripts/extract_mixed_images.py")
        return 0 if ok else 2

    if len(casia_present) < args.casia_lines or len(gnhk_present) < args.gnhk_pages:
        print("\nREFUSING: need %d CASIA images (have %d) and %d GNHK pages (have %d)."
              % (args.casia_lines, len(casia_present), args.gnhk_pages, len(gnhk_present)))
        print("  extract with: python scripts/extract_mixed_images.py")
        return 2

    rng = random.Random(args.seed)
    casia_pick = rng.sample(casia_present, min(args.casia_lines, len(casia_present)))
    gnhk_pick = rng.sample(gnhk_present, min(args.gnhk_pages, len(gnhk_present)))

    out_rows = []
    for _ in range(args.main_repeat):
        out_rows.extend(main_rows)
    for r in casia_pick:
        out_rows.append(_swift_row(_resolve(ROOT / CASIA_DIR, r["image"]), r["markdown"]))
    for r in gnhk_pick:
        out_rows.append(_swift_row(_resolve(ROOT / GNHK_DIR, r["image"]), r["markdown"]))

    rng.shuffle(out_rows)

    n_main = len(main_rows) * args.main_repeat
    total = len(out_rows)
    print("\nmixed plan")
    print("  main  %5d  (%5.1f%%)" % (n_main, 100.0 * n_main / total))
    print("  casia %5d  (%5.1f%%)" % (len(casia_pick), 100.0 * len(casia_pick) / total))
    print("  gnhk  %5d  (%5.1f%%)" % (len(gnhk_pick), 100.0 * len(gnhk_pick) / total))
    print("  total %5d" % total)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out)

    meta = {
        "main": {"file": MAIN, "rows": len(main_rows), "repeat": args.main_repeat},
        "casia": {"file": CASIA, "sampled": len(casia_pick), "seed": args.seed},
        "gnhk": {"file": GNHK, "sampled": len(gnhk_pick), "seed": args.seed},
        "val": {"file": VAL, "rows": len(val_rows)},
        "total": total,
        "instruction": INSTRUCTION,
        "note": "HANDOFF §20 首轮配比；外部数据只进 train，val 保持域内 5 页。",
    }
    (out.parent / "mix_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
