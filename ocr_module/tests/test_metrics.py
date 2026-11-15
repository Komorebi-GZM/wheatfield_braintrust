"""Unit tests for metrics / postprocess (no heavy deps required).

Run:  pytest -q   (or)  python tests/test_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import levenshtein, cer, plain_text, formula_metrics, norm_formula
from ocr.postprocess import extract_formulas, validate_structure, otsl_to_markdown, validate_latex


def test_levenshtein_basic():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3


def test_cer():
    assert cer("abcd", "abcd") == 0.0
    assert abs(cer("abcd", "abce") - 0.25) < 1e-9


def test_plain_text_strips_markup():
    md = "# 标题\n- 要点一\n$$ E=mc^2 $$\n| a | b |\n|---|---|\n| 1 | 2 |"
    t = plain_text(md)
    assert "标题" in t and "要点一" in t
    assert "$$" not in t and "#" not in t


def test_extract_and_norm_formula():
    md = "行内 $a+b$ 与行间 $$\\frac{1}{2}$$"
    fs = extract_formulas(md)
    assert len(fs) == 2
    assert norm_formula(r"\left( x \right)") == "(x)"


def test_formula_metrics_exact():
    ref = "$$ x = 1 $$"
    hyp = "$$x=1$$"
    m = formula_metrics(ref, hyp)
    assert m["n_ref"] == 1
    assert m["exact_match_rate"] == 1.0


def test_structure_ok():
    md = "# A\n- b\n- c\n\n$$ x=1 $$"
    ok, issues, stats = validate_structure(md)
    assert stats["headings"] == 1
    assert stats["list_items"] == 2
    assert stats["formulas"] == 1


def test_validate_latex_unbalanced():
    issues = validate_latex("$$ \\frac{1}{2 $$")
    assert len(issues) >= 1


def test_otsl_to_markdown():
    otsl = "<fcel>量<fcel>符号<nl><fcel>磁通<fcel>Φ<nl>"
    md = otsl_to_markdown(otsl)
    assert "|" in md and "量" in md


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
