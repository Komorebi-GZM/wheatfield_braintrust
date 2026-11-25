"""Scaffold, check and build the independent `test2` annotation set.

Why a separate tool: the 23 pages used so far have been looked at repeatedly, so
they can no longer serve as a "never seen" test set (HANDOFF §8). `test2` is a new
set that nobody has tuned on. A human labels it by dropping photos into
`test2/images/` and writing one Markdown file per page into `test2/answers/`.

Usage (run from `ocr_module/`):

    python scripts/test2_tool.py prepare   # create missing answer files
    python scripts/test2_tool.py check     # report status, non-zero exit if incomplete
    python scripts/test2_tool.py build     # validate + write splits/test2.jsonl

`--root` lets you point at a scratch directory when rehearsing the flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ANSWER_TEMPLATE = """<!-- page: {name}
     在这里写这一页的转写。规则见 ../LABELING_GUIDE.md：
     - 只用 Markdown（标题/列表/表格）+ LaTeX 公式（$...$ / $$...$$）
     - 看不清的字用 □，整块看不清用 > [无法辨认：区域描述]
     - 不要补充图中没有的内容，也不要省略图中有的内容
     写完后运行: python scripts/test2_tool.py check
-->
"""

TODO_MARKERS = ("TODO", "待补", "TBD", "FIXME")


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
    return sorted(p for p in (root / "images").iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _answer_path(root: Path, image: Path) -> Path:
    return root / "answers" / (image.stem + ".md")


def _page_report(root: Path):
    rows = []
    for image in _images(root):
        ap = _answer_path(root, image)
        raw = ap.read_text(encoding="utf-8") if ap.is_file() else ""
        body = _strip_template_comments(raw)
        rows.append({
            "page": image.stem,
            "image": image.name,
            "answer": ap.name,
            "exists": ap.is_file(),
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
    print(f"[prepare] {len(images)} images, {created} answer file(s) created in {root / 'answers'}")
    return 0


def cmd_check(root: Path) -> int:
    rows = _page_report(root)
    if not rows:
        print(f"[check] no images in {root / 'images'}")
        return 1
    print("%-34s %-8s %-6s %-5s %-9s %s" % ("page", "answer", "chars", "□", "formulas", "status"))
    bad = 0
    for r in rows:
        problems = []
        if not r["exists"]:
            problems.append("missing")
        elif r["chars"] == 0:
            problems.append("empty")
        if r["todo"]:
            problems.append("todo:" + ",".join(r["todo"]))
        status = "OK" if not problems else " ".join(problems)
        if problems:
            bad += 1
        print("%-34s %-8s %6d %5d %9d %s" % (r["page"], "yes" if r["exists"] else "NO",
                                            r["chars"], r["box"], r["formulas"], status))
    print(f"\n[check] {len(rows)} pages, {len(rows) - bad} ready, {bad} need work")
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
    ap = argparse.ArgumentParser(description="Scaffold / check / build the test2 annotation set.")
    ap.add_argument("action", choices=["prepare", "check", "build"])
    ap.add_argument("--root", default="datasets/handwriting_notes/test2")
    ap.add_argument("--splits-dir", default="datasets/handwriting_notes/splits")
    ap.add_argument("--mark-reviewed", action="store_true",
                    help="set reviewed=true in the built jsonl (only after a human pass)")
    args = ap.parse_args()
    root = Path(args.root)
    if args.action == "prepare":
        return cmd_prepare(root)
    if args.action == "check":
        return cmd_check(root)
    return cmd_build(root, Path(args.splits_dir), args.mark_reviewed)


if __name__ == "__main__":
    sys.exit(main())
