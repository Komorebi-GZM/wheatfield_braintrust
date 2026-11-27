"""Paired blind comparison of two prediction files on the same manifest.

Writes a per-page CSV plus an aggregate table (text CER / CER-with-math,
cap hits, repetition, win/loss counts) for the given arms.

    python scripts/compare_blind_paired.py \
        --manifest outputs/eval-mixed/blind118_abs.jsonl \
        --a outputs/eval-mixed/blind118-mixed/pred.jsonl --a-name mixed-12000 \
        --b outputs/eval-mixed/blind118-base/pred.jsonl  --b-name base-26 \
        --csv outputs/eval-mixed/blind118_paired.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import cer, plain_text  # noqa: E402
from scripts.eval_supplement import text_with_math  # noqa: E402


def load(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            out[os.path.basename(rec["image"])] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--csv")
    args = ap.parse_args()

    refs, langs = {}, {}
    for line in Path(args.manifest).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            name = os.path.basename(r["image"])
            refs[name] = r["markdown"]
            langs[name] = r.get("language", "?")

    A = load(Path(args.a))
    B = load(Path(args.b))
    pages = sorted(p for p in refs if p in A and p in B)
    if not pages:
        print(f"no paired pages (manifest {len(refs)}, A {len(A)}, B {len(B)})")
        return 1

    rows = []
    for p in pages:
        ref, a, b = refs[p], A[p]["markdown"], B[p]["markdown"]
        ca, cb = cer(plain_text(a), plain_text(ref)), cer(plain_text(b), plain_text(ref))
        rows.append({
            "page": p,
            "lang": langs[p],
            f"cer_{args.a_name}": round(ca, 6),
            f"cer_{args.b_name}": round(cb, 6),
            "delta": round(ca - cb, 6),
            f"cerm_{args.a_name}": round(cer(text_with_math(a), text_with_math(ref)), 6),
            f"cerm_{args.b_name}": round(cer(text_with_math(b), text_with_math(ref)), 6),
            f"cap_{args.a_name}": A[p]["run"]["generation"]["finish_reason"] == "length",
            f"cap_{args.b_name}": B[p]["run"]["generation"]["finish_reason"] == "length",
            f"rep_{args.a_name}": A[p]["run"]["repetition"]["detected"],
            f"rep_{args.b_name}": B[p]["run"]["repetition"]["detected"],
            "chars_a": len(a),
            "chars_b": len(b),
        })

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("csv:", out)

    def mean(key):
        return statistics.mean(r[key] for r in rows)

    better = sum(1 for r in rows if r["delta"] < -1e-9)
    worse = sum(1 for r in rows if r["delta"] > 1e-9)
    print(f"paired pages: {len(rows)}  (A={args.a_name}, B={args.b_name})")
    print(f"mean CER  {args.a_name} {mean('cer_'+args.a_name):.4f} | "
          f"{args.b_name} {mean('cer_'+args.b_name):.4f} | delta {mean('delta'):+.4f}")
    print(f"median delta {statistics.median(r['delta'] for r in rows):+.4f}")
    print(f"mean CERmath {args.a_name} {mean('cerm_'+args.a_name):.4f} | "
          f"{args.b_name} {mean('cerm_'+args.b_name):.4f}")
    print(f"wins(A better)={better}  losses={worse}  ties={len(rows)-better-worse}")
    print(f"cap pages {args.a_name}={sum(r['cap_'+args.a_name] for r in rows)} | "
          f"{args.b_name}={sum(r['cap_'+args.b_name] for r in rows)}")
    print(f"rep pages {args.a_name}={sum(r['rep_'+args.a_name] for r in rows)} | "
          f"{args.b_name}={sum(r['rep_'+args.b_name] for r in rows)}")
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        a_m = statistics.mean(r["cer_" + args.a_name] for r in sub)
        b_m = statistics.mean(r["cer_" + args.b_name] for r in sub)
        print(f"  {lang:<18} n={len(sub):3d}  {args.a_name} {a_m:.4f}  "
              f"{args.b_name} {b_m:.4f}  delta {statistics.mean(r['delta'] for r in sub):+.4f}")
    worst = sorted(rows, key=lambda r: -r["delta"])[:5]
    print("worst regressions (A - B):")
    for r in worst:
        print(f"  {r['page']}  {r['delta']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
