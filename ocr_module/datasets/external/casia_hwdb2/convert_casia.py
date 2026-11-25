#!/usr/bin/env python3
"""Convert the CASIA-HWDB2-line dataset (HuggingFace Teklia/CASIA-HWDB2-line,
parquet with columns `text` + `image.bytes` JPEG) into the wheatfield OCR
jsonl schema: {"image": <relative path>, "markdown": <transcription>}.

Line-level samples are kept as-is (one line image = one sample). They are
intended for MIXED training with the domain pages only (suggested 20-30%),
not as a replacement — see HANDOFF §16 for rationale.

Raw parquet files live in raw/ (gitignored); extracted images land in images/
(gitignored); this script + the manifest are what gets committed.
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "images"
SPLITS = ["train", "validation", "test"]


def main(limit_per_split: int | None = None):
    OUT.mkdir(exist_ok=True)
    manifest = {"source": "Teklia/CASIA-HWDB2-line",
                "license": "research/academic (see upstream CASIA-HWDB2 terms)",
                "note": "line-level Chinese handwriting; mixed-training only",
                "splits": {}}
    for split in SPLITS:
        pq_file = RAW / f"{split}.parquet"
        if not pq_file.exists():
            print(f"[skip] {pq_file} missing")
            continue
        table = pq.read_table(pq_file)
        rows = table.to_pylist()
        if limit_per_split:
            rows = rows[:limit_per_split]
        jsonl_path = HERE / f"casia_{split}.jsonl"
        count, skipped = 0, 0
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                text = (row.get("text") or "").strip()
                img = row.get("image") or {}
                data = img.get("bytes")
                if not text or not data:
                    skipped += 1
                    continue
                rel = f"images/{split}_{i:06d}.jpg"
                (HERE / rel).write_bytes(data)
                fh.write(json.dumps({"image": rel, "markdown": text},
                                    ensure_ascii=False) + "\n")
                count += 1
        manifest["splits"][split] = {"rows": count, "skipped": skipped,
                                     "jsonl": jsonl_path.name}
        print(f"[{split}] wrote {count}, skipped {skipped}")
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=1).encode()
    (HERE / "MANIFEST.json").write_bytes(manifest_bytes)
    print("manifest sha256:", hashlib.sha256(manifest_bytes).hexdigest()[:16])


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
