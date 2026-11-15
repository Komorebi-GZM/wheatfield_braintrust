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

from ocr.postprocess import extract_formulas, normalize_output, validate_structure  # noqa: E402


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
    missing = set(ref) - set(pred)
    extra = set(pred) - set(ref)
    if missing or extra:
        raise ValueError(f'Prediction/reference image mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    os.makedirs(args.out, exist_ok=True)

    per_page = []
    text_cers, f_em, f_cer = [], [], []
    seq_ok = 0
    ed_sum, ref_chars_sum = 0, 0
    pages_hyp_without_formula = 0
    for img, ref_md in ref.items():
        hyp_md = pred[img]
        if '□' in ref_md:
            per_page.append({'image': img, 'text_cer': None, 'formula': None,
                             'structure': None, 'excluded_reason': 'unknown_reference'})
            continue
        # Identical normalisation on both sides: strip one whole-output wrapper
        # (```fence``` / <md></md>) and unify math delimiters to the annotation
        # convention. Reference text carries no wrapper, so it is a no-op there.
        ref_norm, ref_norm_info = normalize_output(ref_md)
        hyp_norm, hyp_norm_info = normalize_output(hyp_md)
        ref_plain, hyp_plain = plain_text(ref_norm), plain_text(hyp_norm)
        ed = levenshtein(ref_plain, hyp_plain)
        c = cer(ref_plain, hyp_plain)
        fm = formula_metrics(ref_norm, hyp_norm)
        sm = structure_metrics(ref_norm, hyp_norm)
        text_cers.append(c)
        ed_sum += ed
        ref_chars_sum += len(ref_plain)
        if fm["exact_match_rate"] is not None:
            f_em.append(fm["exact_match_rate"]); f_cer.append(fm["formula_cer"])
        if fm["n_ref"] and not fm["n_hyp"]:
            pages_hyp_without_formula += 1
        seq_ok += int(sm["heading_seq_match"])
        per_page.append({"image": img, "text_cer": c, "formula": fm, "structure": sm,
                         "ref_chars": len(ref_plain), "hyp_chars": len(hyp_plain),
                         "edit_distance": ed,
                         "hyp_normalization": hyp_norm_info,
                         "ref_normalization": ref_norm_info})
        # save error page
        if c > 0.15 or (fm["exact_match_rate"] is not None and fm["exact_match_rate"] < 0.8):
            with open(os.path.join(args.out, f"err_{os.path.basename(img)}.md"), "w", encoding="utf-8") as ef:
                ef.write("# REF\n" + ref_norm + "\n\n# HYP\n" + hyp_norm)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "n_pages": len(ref),
        "n_scored_pages": len(text_cers),
        "n_excluded_unknown_reference": len(ref) - len(text_cers),
        "text_cer": avg(text_cers),
        "text_cer_macro_definition": "各可评分页 CER 的算术平均（每页等权）",
        "text_cer_micro": (round(ed_sum / ref_chars_sum, 4) if ref_chars_sum else None),
        "text_cer_micro_definition": "sum(编辑距离) / sum(参考字符数)，按字符长度加权；分母是可评分页参考纯文本字符总数",
        "total_edit_distance": ed_sum,
        "total_ref_chars": ref_chars_sum,
        "cer_can_exceed_100pct": True,
        "n_pages_ref_has_formula_but_hyp_none": pages_hyp_without_formula,
        "formula_extractor_delimiters": ["$...$", "$$...$$"],
        "formula_exact_match_rate": avg(f_em),
        "formula_cer": avg(f_cer),
        "structure_heading_seq_match_rate": round(seq_ok / len(text_cers), 4) if text_cers else None,
        "per_page": per_page,
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_page"}, ensure_ascii=False, indent=2))
    print(f"[evaluate] per-page + error samples -> {args.out}")


if __name__ == "__main__":
    main()
