"""Scaffold, Q&A-form, assemble, check and build the independent `test2` set.

Why a separate set: the 23 pages used so far have been looked at repeatedly, so
they can no longer serve as a "never seen" test set (HANDOFF §8). `test2` is new
and nobody has tuned on it.

Two labelling styles are supported:

* **Q&A form (default)** — `form` writes one fill-in-the-blank questionnaire per
  page (`answers/<page>.form.md`); you only answer the questions, then
  `assemble` turns the answers into standard Markdown (`answers/<page>.md`).
* **Free-form** — write `answers/<page>.md` directly.

Usage (run from `ocr_module/`):

    python scripts/test2_tool.py form       # write one Q&A form per image
    python scripts/test2_tool.py assemble   # answers -> answers/<page>.md
    python scripts/test2_tool.py check      # report status, non-zero if incomplete
    python scripts/test2_tool.py build      # validate + write splits/test2.jsonl

`--root` points at another directory when rehearsing the flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SKIP_ANSWERS = {"", "无", "沒有", "没有", "none", "n/a", "-"}

ANSWER_TEMPLATE = """<!-- page: {name}
     在这里写这一页的转写。规则见 ../LABELING_GUIDE.md：
     - 只用 Markdown（标题/列表/表格）+ LaTeX 公式（$...$ / $$...$$）
     - 看不清的字用 □，整块看不清用 > [无法辨认：区域描述]
     - 不要补充图中没有的内容，也不要省略图中有的内容
     写完后运行: python scripts/test2_tool.py check
-->
"""

FORM_TEMPLATE = """<!-- page: {name}
     问答式标注：只在每题的 A 后面作答，不要改问题行。
     不需要的题直接写「无」。全部答完运行:
       python scripts/test2_tool.py assemble
-->

## Q1 这页的标题或编号是什么？（照抄图上的编号/标题；没有就写「无」）
A1: 

## Q2 正文写了什么？（按图上顺序，一条一行；看不清的字写 □，整块看不清写 > [无法辨认：…]）
A2: 

## Q3 有没有表格？（每行写成 | 单元格 | 单元格 |；没有写「无」）
A3: 

## Q4 有没有公式？（行内写 $...$，独立一行写 $$...$$；没有写「无」）
A4: 

## Q5 有没有红笔批注 / 页边注？（直接写内容；没有写「无」）
A5: 

## Q6 有没有插图、框图、示波器截图？（写 > [图示：简述]；没有写「无」）
A6: 
"""

TODO_MARKERS = ("TODO", "待补", "TBD", "FIXME")
QUESTION_RE = re.compile(r"^## Q(\d)\s*(.*)$", re.M)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_template_comments(text: str) -> str:
    """Remove the HTML comment scaffold so it never counts as content."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()


def _images(root: Path):
    d = root / "images"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _answer_path(root: Path, image: Path) -> Path:
    return root / "answers" / (image.stem + ".md")


def _form_path(root: Path, image: Path) -> Path:
    return root / "answers" / (image.stem + ".form.md")


def parse_form(text: str) -> dict:
    """Return {question_number: answer_text} from a filled Q&A form."""
    text = _strip_template_comments(text)
    parts = QUESTION_RE.split(text)
    answers = {}
    # parts = [preamble, num, question, body, num, question, body, ...]
    for i in range(1, len(parts) - 2, 3):
        num = int(parts[i])
        body = parts[i + 2]
        lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("A%d:" % num) or re.match(r"^A\d+:", stripped):
                lines.append(re.sub(r"^A\d+:\s?", "", stripped))
            else:
                lines.append(line.rstrip())
        answers[num] = "\n".join(lines).strip()
    return answers


def assemble_markdown(answers: dict) -> str:
    blocks = []
    title = answers.get(1, "").strip()
    if title.lower() not in SKIP_ANSWERS:
        blocks.append("## " + title.lstrip("# ").strip())
    for key in (2, 3, 4, 5, 6):
        body = answers.get(key, "").strip()
        if body.lower() in SKIP_ANSWERS:
            continue
        blocks.append(body)
    return "\n\n".join(blocks).strip()


def _page_report(root: Path):
    rows = []
    for image in _images(root):
        ap = _answer_path(root, image)
        fp = _form_path(root, image)
        raw = ap.read_text(encoding="utf-8") if ap.is_file() else ""
        body = _strip_template_comments(raw)
        form_filled = False
        if fp.is_file():
            ans = parse_form(fp.read_text(encoding="utf-8"))
            form_filled = any(v.strip().lower() not in SKIP_ANSWERS for v in ans.values())
        rows.append({
            "page": image.stem,
            "image": image.name,
            "answer": ap.name,
            "exists": ap.is_file(),
            "form": fp.is_file(),
            "form_filled": form_filled,
            "chars": len(body),
            "box": body.count("□"),
            "formulas": len(re.findall(r"\$\$?.+?\$\$?", body, flags=re.S)),
            "todo": [m for m in TODO_MARKERS if m in body],
        })
    return rows


def cmd_prepare(root: Path) -> int:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "answers").mkdir(parents=True, exist_ok=True)
    images = _images(root)
    if not images:
        print(f"[prepare] no images in {root / 'images'} — drop the photos there first")
        return 0
    created = 0
    for image in images:
        ap = _answer_path(root, image)
        if not ap.exists():
            ap.write_text(ANSWER_TEMPLATE.format(name=image.name), encoding="utf-8", newline="\n")
            created += 1
    print(f"[prepare] {len(images)} images, {created} free-form answer file(s) created")
    return 0


def cmd_form(root: Path, force: bool) -> int:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "answers").mkdir(parents=True, exist_ok=True)
    images = _images(root)
    if not images:
        print(f"[form] no images in {root / 'images'} — drop the photos there first")
        return 0
    created = skipped = 0
    for image in images:
        fp = _form_path(root, image)
        if fp.exists() and not force:
            skipped += 1
            continue
        fp.write_text(FORM_TEMPLATE.format(name=image.name), encoding="utf-8", newline="\n")
        created += 1
    print(f"[form] {len(images)} images: {created} form(s) written, {skipped} kept (use --force to overwrite)")
    print(f"[form] answer in {root / 'answers'} then run: python scripts/test2_tool.py assemble")
    return 0


def cmd_assemble(root: Path) -> int:
    images = _images(root)
    if not images:
        print(f"[assemble] no images in {root / 'images'}")
        return 1
    built = empty = 0
    for image in images:
        fp = _form_path(root, image)
        if not fp.is_file():
            print(f"[assemble] {image.stem}: no form file, skipped")
            empty += 1
            continue
        answers = parse_form(fp.read_text(encoding="utf-8"))
        md = assemble_markdown(answers)
        if not md:
            print(f"[assemble] {image.stem}: form has no answers yet, skipped")
            empty += 1
            continue
        _answer_path(root, image).write_text(md + "\n", encoding="utf-8", newline="\n")
        built += 1
        print(f"[assemble] {image.stem}: {len(md)} chars -> {_answer_path(root, image).name}")
    print(f"[assemble] {built} assembled, {empty} still empty")
    return 0 if built else 1


def cmd_check(root: Path) -> int:
    rows = _page_report(root)
    if not rows:
        print(f"[check] no images in {root / 'images'}")
        return 1
    print("%-30s %-6s %-6s %-6s %-6s %-5s %-9s %s" % (
        "page", "form", "filled", "answer", "chars", "□", "formulas", "status"))
    bad = 0
    for r in rows:
        problems = []
        if not r["exists"]:
            problems.append("no-answer")
        elif r["chars"] == 0:
            problems.append("empty")
        if r["todo"]:
            problems.append("todo:" + ",".join(r["todo"]))
        if problems:
            bad += 1
        print("%-30s %-6s %-6s %-6s %6d %5d %9d %s" % (
            r["page"], "yes" if r["form"] else "-", "yes" if r["form_filled"] else "-",
            "yes" if r["exists"] else "NO", r["chars"], r["box"], r["formulas"],
            "OK" if not problems else " ".join(problems)))
    print(f"\n[check] {len(rows)} pages, {len(rows) - bad} ready, {bad} need work")
    if bad:
        print("[check] tip: fill answers/<page>.form.md then run `assemble`")
    return 0 if bad == 0 else 1


def cmd_build(root: Path, splits_dir: Path, reviewed: bool) -> int:
    if cmd_check(root) != 0:
        print("[build] refused: fix the pages above first")
        return 1
    rows = _page_report(root)
    splits_dir.mkdir(parents=True, exist_ok=True)
    out = splits_dir / "test2.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            body = _strip_template_comments(
                _answer_path(root, Path(r["image"])).read_text(encoding="utf-8"))
            f.write(json.dumps({"image": "images/" + r["image"], "markdown": body,
                                "reviewed": reviewed}, ensure_ascii=False) + "\n")

    manifest = {
        "set": "test2",
        "n_pages": len(rows),
        "reviewed": reviewed,
        "note": "独立测试集：标注时未参考任何旧 test/val 页，也未参与任何调参。",
        "pages": [{
            "page": r["page"],
            "image": r["image"],
            "image_sha256": _sha256(root / "images" / r["image"]),
            "answer": r["answer"],
            "answer_sha256": _sha256(root / "answers" / r["answer"]),
            "answer_chars": r["chars"],
            "box_placeholders": r["box"],
            "formulas": r["formulas"],
        } for r in rows],
    }
    (root / "test2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"[build] wrote {out} ({len(rows)} pages) and {root / 'test2_manifest.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Scaffold / Q&A-form / assemble / check / build test2.")
    ap.add_argument("action", choices=["prepare", "form", "assemble", "check", "build"])
    ap.add_argument("--root", default="datasets/handwriting_notes/test2")
    ap.add_argument("--splits-dir", default="datasets/handwriting_notes/splits")
    ap.add_argument("--mark-reviewed", action="store_true",
                    help="set reviewed=true in the built jsonl (only after a human pass)")
    ap.add_argument("--force", action="store_true", help="overwrite existing form files")
    args = ap.parse_args()
    root = Path(args.root)
    if args.action == "prepare":
        return cmd_prepare(root)
    if args.action == "form":
        return cmd_form(root, args.force)
    if args.action == "assemble":
        return cmd_assemble(root)
    if args.action == "check":
        return cmd_check(root)
    return cmd_build(root, Path(args.splits_dir), args.mark_reviewed)


if __name__ == "__main__":
    sys.exit(main())
