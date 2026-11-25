#!/usr/bin/env python3
"""Convert the GNHK dataset (GoodNotes Handwriting Kollection, ICDAR 2021,
CC-BY-4.0; mirror: HF aryaman27/goodnotes-handwritten-dataset @9042977)
into the wheatfield OCR jsonl schema.

raw/ holds <id>.jpg + <id>.json (list of line entries: text / polygon /
line_idx / type). Page transcription = lines sorted by line_idx, joined
with newlines; special tokens like %math% kept verbatim.

GNHK is English handwriting in the wild (camera-captured notes). Domain
differs from the Chinese notes main set — mixed training only, 20-30%,
see HANDOFF §18.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def page_markdown(entries):
    lines = sorted(entries, key=lambda e: e.get("line_idx", 0))
    return "\n".join(e["text"] for e in lines if e.get("text"))


def split_lists():
    """Pages belong to train/test per the HF tree listings (cached at
    /tmp is not reproducible — instead derive split from the filename
    ranges using the upstream counts: train 500 pages, test 172 pages,
    ids interleaved, so use the remote listing files if present else
    fall back to index parity of sorted stems."""
    lists = {}
    for split in ["train", "test"]:
        p = Path(f"/tmp/gnhk_{split}.list")
        if p.exists():
            lists[split] = {Path(l.strip()).stem for l in p.read_text().splitlines()
                            if l.strip().endswith(".json")}
    return lists


def main():
    lists = split_lists()
    all_jsons = sorted(RAW.glob("*.json"))
    if not lists.get("train") or not lists.get("test"):
        # fallback: alternate assignment (GNHK ids are interleaved)
        lists = {"train": set(), "test": set()}
        for i, jf in enumerate(all_jsons):
            lists["train" if i % 2 == 0 else "test"].add(jf.stem)
    out = {}
    for split in ["train", "test"]:
        jsonl_path = HERE / f"gnhk_{split}.jsonl"
        count, skipped = 0, 0
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for jf in all_jsons:
                if jf.stem not in lists[split]:
                    continue
                stem = jf.stem
                img = RAW / f"{stem}.jpg"
                if not img.exists():
                    skipped += 1
                    continue
                md = page_markdown(json.loads(jf.read_text(encoding="utf-8")))
                if not md.strip():
                    skipped += 1
                    continue
                rel = f"images/{split}_{stem}.jpg"
                (HERE / rel).parent.mkdir(exist_ok=True)
                (HERE / rel).write_bytes(img.read_bytes())
                fh.write(json.dumps({"image": rel, "markdown": md},
                                    ensure_ascii=False) + "\n")
                count += 1
        out[split] = {"pages": count, "skipped": skipped,
                      "jsonl": jsonl_path.name}
        print(f"[{split}] wrote {count}, skipped {skipped}")
    manifest = {"source": "GoodNotes/GNHK (ICDAR 2021), mirror "
                        "aryaman27/goodnotes-handwritten-dataset@9042977",
                "license": "CC-BY-4.0",
                "note": "English handwriting in the wild; page-level; "
                        "mixed training only",
                "splits": out}
    mb = json.dumps(manifest, ensure_ascii=False, indent=1).encode()
    (HERE / "MANIFEST.json").write_bytes(mb)
    print("manifest sha256:", hashlib.sha256(mb).hexdigest()[:16])


if __name__ == "__main__":
    main()
