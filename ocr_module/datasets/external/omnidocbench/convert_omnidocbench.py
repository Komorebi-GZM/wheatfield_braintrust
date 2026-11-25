#!/usr/bin/env python3
"""Extract the handwritten-note subset from OmniDocBench (opendatalab,
research-use license) into the wheatfield jsonl schema.

The full benchmark has 1651 pages across 10 document types; we take only
`data_source == "note"` (118 pages: 70 simplified-Chinese + 47 mixed zh/en
handwritten notes), because it matches the wheatfield domain (Chinese notes
with LaTeX math). Page transcription = text blocks ordered by `order`,
joined with newlines. Eval-only usage by default (do not train on it —
it is a public benchmark, potential leakage).

Requires raw/OmniDocBench.json + images/ (both gitignored; rebuild via
download_rest.py + HF tree API).
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "raw" / "OmniDocBench.json").read_text())
    notes = [p for p in data
             if p["page_info"]["page_attribute"]["data_source"] == "note"]
    langs = Counter(p["page_info"]["page_attribute"].get("language")
                    for p in notes)
    jsonl_path = HERE / "omnidocbench_notes.jsonl"
    count, skipped = 0, 0
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for page in notes:
            img_name = page["page_info"]["image_path"]
            img = HERE / "images" / img_name
            if not img.exists() or img.stat().st_size == 0:
                skipped += 1
                continue
            dets = [d for d in page["layout_dets"]
                    if d.get("text") and not d.get("ignore")]
            dets.sort(key=lambda d: (d.get("order") is None, d.get("order") or 0))
            md = "\n".join(d["text"] for d in dets)
            if not md.strip():
                skipped += 1
                continue
            rel = f"images/{img_name}"
            fh.write(json.dumps({"image": rel, "markdown": md,
                                 "language": page["page_info"]["page_attribute"].get("language")},
                                ensure_ascii=False) + "\n")
            count += 1
    manifest = {
        "source": "opendatalab/OmniDocBench (note subset)",
        "license": "research use only, not for commercial use",
        "usage": "EVAL ONLY — public benchmark, do not train on it",
        "languages": dict(langs),
        "splits": {"note": {"pages": count, "skipped": skipped,
                            "jsonl": jsonl_path.name}},
    }
    mb = json.dumps(manifest, ensure_ascii=False, indent=1).encode()
    (HERE / "MANIFEST.json").write_bytes(mb)
    print(f"note pages: {count}, skipped {skipped}, langs {dict(langs)}")
    print("manifest sha256:", hashlib.sha256(mb).hexdigest()[:16])


if __name__ == "__main__":
    main()
