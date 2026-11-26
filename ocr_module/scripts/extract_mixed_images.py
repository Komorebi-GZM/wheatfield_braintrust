"""Extract ONLY the images needed by the HANDOFF §20 mixed-training plan.

Why not run the upstream converters as-is: they materialise the whole external
corpus (52k CASIA lines + 672 GNHK pages, ~1.6 GB). A first mixed run needs
8,000 lines and 80 pages, so we pull exactly those, deterministically (seed 42),
without touching the committed jsonl/MANIFEST files.

Inputs (expected, gitignored):
  datasets/external/casia_hwdb2/raw/train.parquet   (mirror Teklia/CASIA-HWDB2-line)
  jsonl lists already committed: casia_train.jsonl / gnhk_train.jsonl

Outputs (gitignored):
  datasets/external/casia_hwdb2/images/train_XXXXXX.jpg
  datasets/external/gnhk/images/train_<stem>.jpg
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASIA = ROOT / "datasets/external/casia_hwdb2"
GNHK = ROOT / "datasets/external/gnhk"
MIRROR = "https://hf-mirror.com"
GNHK_REPO = "aryaman27/goodnotes-handwritten-dataset"
GNHK_REV = "9042977"


def _rows(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_casia(n: int, seed: int) -> int:
    import pyarrow.parquet as pq

    rows = _rows(CASIA / "casia_train.jsonl")
    rng = random.Random(seed)
    picked = rng.sample(range(len(rows)), min(n, len(rows)))
    picked.sort()
    wanted = {}
    for i in picked:
        name = Path(rows[i]["image"]).name
        idx = int(name.split("_")[1].split(".")[0])
        wanted[idx] = rows[i]["image"]

    pq_file = CASIA / "raw/train.parquet"
    if not pq_file.exists():
        print(f"[casia] missing {pq_file}")
        return 0
    print(f"[casia] reading {pq_file.name} ({pq_file.stat().st_size/1e6:.0f} MB) ...", flush=True)
    table = pq.read_table(pq_file, columns=["image"])
    images = table.column("image").to_pylist()

    out_dir = CASIA / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, rel in wanted.items():
        if idx >= len(images):
            continue
        data = (images[idx] or {}).get("bytes")
        if not data:
            continue
        (ROOT / "datasets/external/casia_hwdb2" / rel).write_bytes(data)
        written += 1
    print(f"[casia] wrote {written} images for {len(picked)} sampled lines")
    return written


def extract_gnhk(n: int, seed: int) -> int:
    rows = _rows(GNHK / "gnhk_train.jsonl")
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    out_dir = GNHK / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in picked:
        base = Path(row["image"]).name          # train_eng_AF_001.jpg
        stem = base[len("train_"):]             # eng_AF_001.jpg
        url = f"{MIRROR}/datasets/{GNHK_REPO}/resolve/{GNHK_REV}/train_data/train/{stem}"
        dst = out_dir / base
        if dst.exists():
            written += 1
            continue
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    dst.write_bytes(r.read())
                written += 1
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"[gnhk] FAILED {stem}: {type(exc).__name__} {exc}")
                else:
                    time.sleep(1.5)
    print(f"[gnhk] wrote {written}/{len(picked)} pages")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casia-lines", type=int, default=8000)
    ap.add_argument("--gnhk-pages", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["casia", "gnhk", "both"], default="both")
    args = ap.parse_args()

    if args.only in ("casia", "both"):
        extract_casia(args.casia_lines, args.seed)
    if args.only in ("gnhk", "both"):
        extract_gnhk(args.gnhk_pages, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
