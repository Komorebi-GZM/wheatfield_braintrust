"""Download the 118 OmniDocBench note page images (eval-only track ②).

The repo layout is `images/<page-id>.png`; we fetch only the basenames listed in
`omnidocbench_notes.jsonl`, so no 1.6k-page bulk download.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ODD = ROOT / "datasets/external/omnidocbench"
MIRROR = "https://hf-mirror.com"
REPO = "opendatalab/OmniDocBench"
REV = "main"


def main() -> int:
    rows = [json.loads(l) for l in (ODD / "omnidocbench_notes.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    out = ODD / "images"
    out.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    for row in rows:
        name = Path(row["image"]).name
        dst = out / name
        if dst.exists():
            skip += 1
            continue
        url = f"{MIRROR}/datasets/{REPO}/resolve/{REV}/images/{name}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    dst.write_bytes(r.read())
                ok += 1
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    fail += 1
                    print(f"[fail] {name}: {type(exc).__name__} {exc}", flush=True)
                else:
                    time.sleep(1.5)
    print(f"downloaded={ok} skipped={skip} failed={fail} total={len(rows)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
