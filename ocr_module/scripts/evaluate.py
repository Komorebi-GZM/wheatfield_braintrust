"""Evaluation: text CER, structure preservation, formula fidelity.

Usage:
    python scripts/evaluate.py --pred outputs/pred/test_pred.jsonl --ref data/splits/test.jsonl --out outputs/eval

Pred / ref jsonl each line: {"image": <id>, "markdown": "..."}

Metrics:
  - text CER            : edit distance over plain text (markdown/latex stripped)
  - structure           : heading-level sequence match, list/table count match
  - formula             : normalized exact-match rate, formula CER
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Dict, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.postprocess import extract_formulas, validate_structure  # noqa: E402


# --------------------------------------------------------------------------- #
# Core text helpers
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def plain_text(markdown: str) -> str:
    """Strip markdown/latex to comparable plain text."""
    t = markdown
    t = re.sub(r"\$\$.+?\$\$", " ", t, flags=re.DOTALL)      # display math
    t = re.sub(r"\$.+?\$", " ", t)                            # inline math
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)  # heading marks
    t = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", t, flags=re.MULTILINE)  # list marks
    t = re.sub(r"\|", " ", t)                                # table pipes
    t = re.sub(r"^\s*:?-{3,}.*$", "", t, flags=re.MULTILINE)  # table sep
    t = re.sub(r"\s+", "", t)
    return t


def norm_formula(f: str) -> str:
    f = f.strip()
    for tok in (r"\left", r"\right", r"\,", r"\;", r"\ ", r"\!", " ", "\n"):
        f = f.replace(tok, "")
    return f


def formula_metrics(ref_md: str, hyp_md: str) -> Dict:
    ref = [norm_formula(x) for x in extract_formulas(ref_md)]
    hyp = [norm_formula(x) for x in extract_formulas(hyp_md)]
    if not ref:
        return {"n_ref": 0, "exact_match_rate": None, "formula_cer": None}
    # greedy exact match set overlap
    pool = list(hyp)
    hits = 0
    for r in ref:
        if r in pool:
            pool.remove(r); hits += 1
    # formula CER: compare concatenations (order-sensitive, coarse)
    fcer = cer("\n".join(ref), "\n".join(hyp))
    return {"n_ref": len(ref), "n_hyp": len(hyp),
            "exact_match_rate": hits / len(ref), "formula_cer": fcer}


# --------------------------------------------------------------------------- #
# Structure comparison
# --------------------------------------------------------------------------- #
def _heading_levels(md: str) -> List[int]:
    return [len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s+\S", md, re.MULTILINE)]


def structure_metrics(ref_md: str, hyp_md: str) -> Dict:
    _, _, rs = validate_structure(ref_md)
    _, _, hs = validate_structure(hyp_md)
    rl, hl = _heading_levels(ref_md), _heading_levels(hyp_md)
    return {
        "ref": rs, "hyp": hs,
        "headings_count_match": rs["headings"] == hs["headings"],
        "heading_seq_match": rl == hl,
        "list_count_match": rs["list_items"] == hs["list_items"],
        "table_count_match": rs["tables"] == hs["tables"],
    }


# --------------------------------------------------------------------------- #
# Drive
# --------------------------------------------------------------------------- #
def _read(path: str) -> Dict[str, str]:
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            d[str(it["image"])] = it["markdown"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    pred = _read(args.pred)
    ref = _read(args.ref)
    os.makedirs(args.out, exist_ok=True)

    per_page = []
    text_cers, f_em, f_cer = [], [], []
    seq_ok = 0
    for img, ref_md in ref.items():
        hyp_md = pred.get(img, "")
        c = cer(plain_text(ref_md), plain_text(hyp_md))
        fm = formula_metrics(ref_md, hyp_md)
        sm = structure_metrics(ref_md, hyp_md)
        text_cers.append(c)
        if fm["exact_match_rate"] is not None:
            f_em.append(fm["exact_match_rate"]); f_cer.append(fm["formula_cer"])
        seq_ok += int(sm["heading_seq_match"])
        per_page.append({"image": img, "text_cer": c, "formula": fm, "structure": sm})
        # save error page
        if c > 0.15 or (fm["exact_match_rate"] or 1) < 0.8:
            with open(os.path.join(args.out, f"err_{os.path.basename(img)}.md"), "w", encoding="utf-8") as ef:
                ef.write("# REF\n" + ref_md + "\n\n# HYP\n" + hyp_md)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "n_pages": len(ref),
        "text_cer": avg(text_cers),
        "formula_exact_match_rate": avg(f_em),
        "formula_cer": avg(f_cer),
        "structure_heading_seq_match_rate": round(seq_ok / len(ref), 4) if ref else None,
        "per_page": per_page,
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_page"}, ensure_ascii=False, indent=2))
    print(f"[evaluate] per-page + error samples -> {args.out}")


if __name__ == "__main__":
    main()
