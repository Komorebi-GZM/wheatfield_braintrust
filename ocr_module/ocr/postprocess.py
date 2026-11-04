"""Post-processing and validation for structured OCR Markdown.

Responsibilities:
- best-effort OTSL -> Markdown table conversion (production should reuse the
  official Xiaomi-OCR-0 `postprocess/` OTSL converter);
- LaTeX validity checks on `$...$` / `$$...$$` spans;
- Markdown structure sanity (headings, list items, tables, brace pairing).

All functions are dependency-light: `pylatexenc` is optional.
"""

import re
from typing import List, Tuple, Dict

try:  # optional
    from pylatexenc.latex2text import LatexNodes2Text  # type: ignore
    _HAS_PYLATEXENC = True
except Exception:  # pragma: no cover
    _HAS_PYLATEXENC = False


# --------------------------------------------------------------------------- #
# Formula extraction / validation
# --------------------------------------------------------------------------- #
_FORMULA_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)


def extract_formulas(markdown: str) -> List[str]:
    """Return all LaTeX formulas (display + inline) in document order."""
    out: List[str] = []
    for m in _FORMULA_RE.finditer(markdown):
        out.append((m.group(1) or m.group(2) or "").strip())
    return out


def _basic_latex_ok(expr: str) -> bool:
    if not expr.strip():
        return False
    # balanced braces
    depth = 0
    for ch in expr:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    # unclosed \left / \right pair
    if expr.count(r"\left") != expr.count(r"\right"):
        return False
    return True


def validate_latex(markdown: str) -> List[str]:
    """Return a list of problem descriptions for malformed formulas."""
    issues: List[str] = []
    for i, expr in enumerate(extract_formulas(markdown)):
        if not _basic_latex_ok(expr):
            issues.append(f"公式#{i} 括号/结构不平衡: {expr[:60]!r}")
            continue
        if _HAS_PYLATEXENC:
            try:
                LatexNodes2Text().latex_to_text(expr)
            except Exception as e:  # pragma: no cover
                issues.append(f"公式#{i} 解析告警: {expr[:60]!r} ({e})")
    return issues


# --------------------------------------------------------------------------- #
# OTSL -> Markdown table (best-effort)
# --------------------------------------------------------------------------- #
_OTSL_CELL = {"<fcel>", "<ecel>", "<ched>", "<rhed>", "<cell>", "<ucel>", "<lcel>", "<xcel>"}


def otsl_to_markdown(otsl: str) -> str:
    """Convert a best-effort OTSL token stream to a Markdown table.

    Supports the common cell tokens and `<nl>` row separators. This is a
    skeleton converter; for production fidelity reuse the official Xiaomi
    `postprocess/` OTSL converter.
    """
    text = otsl.strip()
    if "<nl>" not in text:
        return otsl  # not OTSL-looking; leave untouched
    rows: List[List[str]] = []
    cur: List[str] = []
    buf = ""
    i = 0
    while i < len(text):
        matched = None
        for tok in sorted(_OTSL_CELL | {"<nl>"}, key=len, reverse=True):
            if text.startswith(tok, i):
                matched = tok
                break
        if matched:
            if matched == "<nl>":
                cur.append(buf.strip()); rows.append(cur); cur = []; buf = ""
            else:
                cur.append(buf.strip()); buf = ""
            i += len(matched)
        else:
            buf += text[i]; i += 1
    if buf.strip() or cur:
        cur.append(buf.strip()); rows.append(cur)
    if not rows:
        return otsl
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header = rows[0]
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * ncol) + "|"]
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def convert_tables(markdown: str) -> str:
    """Replace OTSL-looking blocks with Markdown tables (best-effort)."""
    # Split on blank-line blocks and convert any block containing OTSL tokens.
    blocks = re.split(r"\n\s*\n", markdown)
    out = []
    for b in blocks:
        if "<nl>" in b and any(t in b for t in _OTSL_CELL):
            out.append(otsl_to_markdown(b))
        else:
            out.append(b)
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Structure validation
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S", re.MULTILINE)


def _valid_markdown_table(lines: List[str]) -> bool:
    """A table block is a run of lines starting with '|' with a separator row."""
    if len(lines) < 2:
        return True  # not a table
    sep = [l for l in lines if re.match(r"^\s*\|?\s*:?-{3,}", l)]
    return len(sep) >= 1


def validate_structure(markdown: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """Return (ok, issues, stats). Checks heading/list/table well-formedness."""
    issues: List[str] = []
    headings = _HEADING_RE.findall(markdown)
    levels = [len(h) for h in headings]
    # heading level jumps (e.g. # -> ###) are suspicious but not fatal
    prev = 0
    jumps = 0
    for lv in levels:
        if prev and lv > prev + 1:
            jumps += 1
        prev = lv
    if jumps:
        issues.append(f"标题层级存在 {jumps} 处跳级")

    # table blocks
    lines = markdown.splitlines()
    i = 0
    tables = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            block = []
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                block.append(lines[j]); j += 1
            tables += 1
            if not _valid_markdown_table(block):
                issues.append("存在缺少分隔行的疑似表格")
            i = j
        else:
            i += 1

    stats = {
        "headings": len(headings),
        "list_items": len(_LIST_RE.findall(markdown)),
        "tables": tables,
        "formulas": len(extract_formulas(markdown)),
    }
    ok = len([x for x in issues if "跳级" not in x]) == 0
    return ok, issues, stats


def postprocess(markdown: str) -> Dict:
    """Full pipeline: table conversion + validation. Returns dict of results."""
    md = convert_tables(markdown)
    ok, struct_issues, stats = validate_structure(md)
    latex_issues = validate_latex(md)
    return {
        "markdown": md,
        "structure_ok": ok,
        "structure_issues": struct_issues,
        "latex_issues": latex_issues,
        "stats": stats,
    }
