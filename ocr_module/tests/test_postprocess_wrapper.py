"""Unit tests for output normalisation and repetition rules (no GPU, no model).

Every assertion here corresponds to a failure shape actually observed on the
test/val pages, so the rules are grounded in real samples rather than guesses.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.postprocess import (  # noqa: E402
    normalize_math_delimiters,
    normalize_output,
    strip_md_tag,
    strip_outer_fence,
)
from ocr.engine import detect_repetition  # noqa: E402


# --------------------------------------------------------------------------- #
# wrapper stripping
# --------------------------------------------------------------------------- #
def test_strip_code_fence():
    text, info = normalize_output("```markdown\n# A\n\n$x=1$\n```")
    assert text == "# A\n\n$x=1$"
    assert info["wrapper_stripped"] == "code_fence"


def test_strip_code_fence_no_language():
    text, info = normalize_output("```\nplain\n```")
    assert text == "plain"
    assert info["wrapper_stripped"] == "code_fence"


def test_strip_md_tag():
    # observed on the base model: <md> ... </md>
    text, info = normalize_output("<md> | 4-2 | x | \n</md>")
    assert text.startswith("| 4-2 |")
    assert "</md>" not in text
    assert info["wrapper_stripped"] == "md_tag"


def test_unclosed_fence_is_left_alone():
    # truncation ends mid-fence; stripping would destroy the only marker
    raw = "```markdown\n### 4-2\nW_{12} = ..."
    text, stripped = strip_outer_fence(raw)
    assert stripped is False
    assert text == raw


def test_unclosed_md_tag_is_left_alone():
    raw = "<md> table cut off"
    text, stripped = strip_md_tag(raw)
    assert stripped is False
    assert text == raw


def test_inner_code_block_survives():
    raw = "```markdown\nintro\n\n```asm\nmov a,b\n```\n\noutro\n```"
    text, stripped = strip_outer_fence(raw)
    assert stripped is True
    assert "```asm" in text and "```" in text
    assert not text.startswith("```markdown")


def test_leading_prose_blocks_stripping():
    raw = "以下是转写结果：\n```markdown\n# A\n```"
    text, stripped = strip_outer_fence(raw)
    assert stripped is False
    assert text == raw


# --------------------------------------------------------------------------- #
# math delimiters
# --------------------------------------------------------------------------- #
def test_math_delimiters_converted():
    src = r"\[ C = \frac{a}{b} \] 以及 \( x^2 \)"
    out, counts = normalize_math_delimiters(src)
    assert "$$ C = \\frac{a}{b} $$".replace("$$ ", "$$") in out or "$$C = \\frac{a}{b}$$" in out
    assert "$x^2$" in out
    assert counts == {"display_converted": 1, "inline_converted": 1}


def test_annotation_style_untouched():
    src = "$$a=b$$ 和 $c$"
    out, counts = normalize_math_delimiters(src)
    assert out == src
    assert counts == {"display_converted": 0, "inline_converted": 0}


def test_normalize_output_is_idempotent():
    once, _ = normalize_output("```markdown\n\\[ a=b \\]\n```")
    twice, info = normalize_output(once)
    assert twice == once
    assert info["changed"] is False


def test_normalize_output_reports_change_flag():
    _, info = normalize_output("```markdown\n# A\n```")
    assert info["changed"] is True
    _, info2 = normalize_output("# A")
    assert info2["changed"] is False
    assert info2["wrapper_stripped"] is None


# --------------------------------------------------------------------------- #
# repetition rules  (samples taken from outputs/diagnosis/*.json)
# --------------------------------------------------------------------------- #
def test_ngram_rule_catches_observed_checkpoint4_loop():
    # real shape from d2: identical display formula, only the index increments
    loop = "\n\n".join(r"\[ W_{%d} = \frac{\partial f}{\partial x} \]" % i for i in range(12, 34))
    rep = detect_repetition(loop)
    assert rep["detected"] is True
    assert rep["ngram"]["flagged"] is True


def test_template_rule_catches_index_degeneration():
    loop = "\n\n".join(r"\[ W_{%d} = \frac{\partial f}{\partial x} \]" % i for i in range(12, 22))
    rep = detect_repetition(loop)
    assert rep["template_line_run"]["flagged"] is True
    assert rep["template_line_run"]["max_consecutive"] >= 5


def test_table_rows_do_not_trip_template_rule():
    table = "| a | 1 |\n| --- | --- |\n" + "\n".join("| x | %d |" % i for i in range(1, 12))
    rep = detect_repetition(table)
    assert rep["template_line_run"]["flagged"] is False


def test_wide_table_separator_is_not_repetition():
    # real false positive before the table exclusion: a legit wide header row
    # produced ` | ---` x12 and tripped unit_run on IMG_20261104_235946
    head = "| 概览 | 源入 | 位数 | " + " | ".join(str(i) for i in range(9)) + " |"
    sep = "| --- " * 12 + "|"
    rows = "\n".join(["| 小管 | TIP | 横轴 | " + " | ".join(str(i) for i in range(9)) + " |"] * 3)
    rep = detect_repetition("\n".join([head, sep, rows]))
    assert rep["unit_run"]["flagged"] is False
    assert rep["detected"] is False


def test_degenerate_table_filling_is_still_caught():
    # real degeneration on IMG_20261104_235955: one cell pair repeated to fill the page
    text = "| " + " | ".join(["课程名称 | 课程代码"] * 60) + " |"
    rep = detect_repetition(text)
    assert rep["ngram"]["flagged"] is True
    assert rep["detected"] is True


def test_normal_page_is_not_flagged():
    # real head of the val page output (d4): legit, no repetition
    ok = (
        "## 4-2\n\n**变曲根距模型中由导数和微分确定时**\n\n"
        "$$C = \\frac{\\partial f}{\\partial x}$$\n\n"
        "因此电感式与微分方程属于非线性系统。\n\n"
        "## 4-3\n\n微分函数\n\n$$k_x \\quad \\frac{\\partial f}{\\partial x}$$\n"
    )
    rep = detect_repetition(ok)
    assert rep["detected"] is False


def test_line_run_rule_catches_verbatim_repeat():
    rep = detect_repetition("\n".join(["同样一行"] * 8))
    assert rep["line_run"]["flagged"] is True
