"""Supplemental metric: text CER with formula bodies *included*.

Why this exists
---------------
`scripts/evaluate.py` (the frozen evaluator) compares plain text produced by
`plain_text()`, which deletes `$...$` and `$$...$$` spans entirely. That is fine
for measuring prose, but it means a model that wraps most of its output in math
delimiters gets that content erased before scoring. The new adapter does exactly
that, so its main-metric CER could look better for the wrong reason.

This script recomputes CER from the *same* normalized predictions with the same
edit-distance code, but keeps the math bodies (only the delimiters are removed).
Both arms go through the identical path, and the no-math column is printed next
to the main metric so the difference is auditable.

Output: outputs/eval_v3/supplement.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.postprocess import normalize_output  # noqa: E402
from scripts.evaluate import cer, levenshtein, plain_text  # noqa: E402

_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)


def text_with_math(md: str) -> str:
    """plain_text() but formula bodies survive as plain characters."""
    def keep(m):
        return " " + (m.group(1) or m.group(2) or "") + " "

    t = _MATH_SPAN.sub(keep, md)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"^\s*:?-{3,}.*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s+", "", t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="datasets/handwriting_notes/splits/test.jsonl")
    ap.add_argument("--base-pred", default="outputs/eval_v3/baseline/test_pred_v3.jsonl")
    ap.add_argument("--ft-pred", default="outputs/eval_v3/finetuned/test_pred_v3.jsonl")
    ap.add_argument("--out", default="outputs/eval_v3/supplement.json")
    args = ap.parse_args()

    ref = {}
    for line in open(args.ref, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            ref[os.path.basename(it["image"])] = it["markdown"]

    def load(p):
        d = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                it = json.loads(line)
                d[os.path.basename(it["image"])] = it["markdown"]
        return d

    arms = {"base": load(args.base_pred), "finetuned": load(args.ft_pred)}
    out = {"definition": {
        "no_math": "与冻结评估器一致：plain_text() 删除 $...$ / $$...$$ 整段",
        "with_math": "同上但保留公式体、仅去掉定界符；分母为参考侧的字符数",
        "normalization": "两侧均调用 ocr.postprocess.normalize_output",
        "exclusion": "参考含 □ 的页整页排除（与主指标同一规则）",
    }, "arms": {}}

    for name, pred in arms.items():
        rows, ed_no, ref_no, ed_w, ref_w = [], 0, 0, 0, 0
        for img, ref_md in ref.items():
            if "□" in ref_md:
                rows.append({"page": img, "excluded": "unknown_reference"})
                continue
            r, _ = normalize_output(ref_md)
            h, _ = normalize_output(pred[img])
            r_plain, h_plain = plain_text(r), plain_text(h)
            r_wm, h_wm = text_with_math(r), text_with_math(h)
            e_no = levenshtein(r_plain, h_plain)
            e_w = levenshtein(r_wm, h_wm)
            ed_no += e_no; ref_no += len(r_plain)
            ed_w += e_w; ref_w += len(r_wm)
            rows.append({
                "page": img,
                "cer_no_math": cer(r_plain, h_plain),
                "cer_with_math": cer(r_wm, h_wm),
                "ref_chars_no_math": len(r_plain), "hyp_chars_no_math": len(h_plain),
                "ref_chars_with_math": len(r_wm), "hyp_chars_with_math": len(h_wm),
                "edit_no_math": e_no, "edit_with_math": e_w,
            })
        scored = [x for x in rows if "excluded" not in x]
        out["arms"][name] = {
            "n_scored": len(scored),
            "macro_cer_no_math": round(sum(x["cer_no_math"] for x in scored) / len(scored), 4),
            "macro_cer_with_math": round(sum(x["cer_with_math"] for x in scored) / len(scored), 4),
            "micro_cer_no_math": round(ed_no / ref_no, 4) if ref_no else None,
            "micro_cer_with_math": round(ed_w / ref_w, 4) if ref_w else None,
            "total_ref_chars_no_math": ref_no, "total_ref_chars_with_math": ref_w,
            "per_page": rows,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for name in ("base", "finetuned"):
        a = out["arms"][name]
        print(f"{name:10s} macro no_math={a['macro_cer_no_math']:.4f} "
              f"with_math={a['macro_cer_with_math']:.4f} | "
              f"micro no_math={a['micro_cer_no_math']:.4f} with_math={a['micro_cer_with_math']:.4f}")
    print("\nper page (base -> ft):")
    b = {x["page"]: x for x in out["arms"]["base"]["per_page"]}
    f_ = {x["page"]: x for x in out["arms"]["finetuned"]["per_page"]}
    for img in b:
        if "excluded" in b[img]:
            print(f"  {os.path.basename(img):32s} excluded")
            continue
        print(f"  {os.path.basename(img):32s} no_math {b[img]['cer_no_math']:.4f}->{f_[img]['cer_no_math']:.4f}"
              f"   with_math {b[img]['cer_with_math']:.4f}->{f_[img]['cer_with_math']:.4f}"
              f"   hyp_chars(with_math) {b[img]['hyp_chars_with_math']}->{f_[img]['hyp_chars_with_math']}")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
