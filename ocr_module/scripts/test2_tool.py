"""One questionnaire for the whole `test2` set. Two commands, one file.

    python scripts/test2_tool.py form     # 生成 test2/问卷.md（每页六题）
    #  → 在 问卷.md 里回答，不需要的题写「无」
    python scripts/test2_tool.py build    # 解析问卷 -> 标注 -> splits/test2.jsonl

`build` 会一次做完：写 `answers/<页>.md`、`splits/test2.jsonl`、`test2_manifest.json`，
并打印每页状态。`check` 只做体检，不写文件。

Free-form is still accepted: if `answers/<页>.md` already exists and the page is
blank in the questionnaire, that file is used as-is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SKIP = {"", "无", "沒有", "没有", "none", "n/a", "-"}
QUESTIONNAIRE = "问卷.md"

PAGE_RE = re.compile(r"^#\s*第\s*(\d+)\s*页\s*[:：]\s*(\S+)\s*$", re.M)
Q_RE = re.compile(r"^Q([1-6])\s*[:：]?\s*(.*)$", re.M)

QUESTIONS = [
    (1, "标题/编号（照抄图上编号，没有写「无」）"),
    (2, "正文（按图上顺序一条一行，看不清的字写 □）"),
    (3, "表格（每行 | 单元格 | 单元格 |，没有写「无」）"),
    (4, "公式（行内 $...$，独立一行 $$...$$，没有写「无」）"),
    (5, "红笔批注/页边注（没有写「无」）"),
    (6, "插图/框图/示波器截图（写 > [图示：…]，没有写「无」）"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _images(root: Path):
    d = root / "images"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _questionnaire_text(root: Path, images) -> str:
    lines = [
        "# test2 标注问卷",
        "",
        f"<!-- 共 {len(images)} 页。每题在 Q 后面直接写答案；一行不够就换行继续写，",
        "     遇到下一个 Q 或下一页标题就自动结束。不需要的题写「无」。",
        "     全部答完后运行:  python scripts/test2_tool.py build  -->",
        "",
    ]
    for i, img in enumerate(images, 1):
        lines.append(f"# 第 {i} 页：{img.name}")
        lines.append("")
        for num, prompt in QUESTIONS:
            lines.append(f"Q{num} {prompt}")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def parse_questionnaire(text: str) -> dict:
    """{image_name: {1..6: answer}} parsed from the single questionnaire file."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    pages = {}
    marks = list(PAGE_RE.finditer(text))
    for idx, m in enumerate(marks):
        name = m.group(2)
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(text)
        block = text[m.end():end]
        answers = {}
        qmarks = list(Q_RE.finditer(block))
        for qi, qm in enumerate(qmarks):
            num = int(qm.group(1))
            qend = qmarks[qi + 1].start() if qi + 1 < len(qmarks) else len(block)
            body = block[qm.end():qend]
            answers[num] = body.strip()
        pages[name] = answers
    return pages


def to_markdown(answers: dict) -> str:
    blocks = []
    title = answers.get(1, "").strip()
    if title.lower() not in SKIP:
        blocks.append("## " + title.lstrip("# ").strip())
    for num in (2, 3, 4, 5, 6):
        body = answers.get(num, "").strip()
        if body.lower() not in SKIP:
            blocks.append(body)
    return "\n\n".join(blocks).strip()


def _report(rows):
    print("%-28s %-6s %-6s %-6s %-5s %-9s %s" % (
        "page", "filled", "answer", "chars", "□", "formulas", "status"))
    bad = 0
    for r in rows:
        problems = []
        if not r["filled"]:
            problems.append("未答")
        elif r["chars"] == 0:
            problems.append("空")
        if r["todo"]:
            problems.append("todo")
        if problems:
            bad += 1
        print("%-28s %-6s %-6s %6d %5d %9d %s" % (
            r["page"][:28], "yes" if r["filled"] else "-", "yes" if r["filled"] else "-",
            r["chars"], r["box"], r["formulas"], "OK" if not problems else " ".join(problems)))
    print(f"\n{len(rows)} 页，{len(rows) - bad} 页就绪，{bad} 页待处理")
    return bad


def collect(root: Path):
    """Merge questionnaire answers with any existing free-form answers/<page>.md."""
    images = _images(root)
    qpath = root / QUESTIONNAIRE
    answers_by_image = {}
    if qpath.is_file():
        answers_by_image = parse_questionnaire(qpath.read_text(encoding="utf-8"))
    rows = []
    for img in images:
        answers = answers_by_image.get(img.name, {})
        md = to_markdown(answers)
        free = root / "answers" / (img.stem + ".md")
        if not md and free.is_file():
            md = re.sub(r"<!--.*?-->", "", free.read_text(encoding="utf-8"), flags=re.S).strip()
        rows.append({
            "page": img.stem,
            "image": img.name,
            "markdown": md,
            "filled": bool(md),
            "has_md": free.is_file(),
            "chars": len(md),
            "box": md.count("□"),
            "formulas": len(re.findall(r"\$\$?.+?\$\$?", md, flags=re.S)),
            "todo": [m for m in ("TODO", "待补", "TBD", "FIXME") if m in md],
        })
    return rows


def cmd_form(root: Path, force: bool) -> int:
    (root / "images").mkdir(parents=True, exist_ok=True)
    images = _images(root)
    if not images:
        print(f"[form] {root / 'images'} 里还没有照片 —— 先把页面放进去")
        return 0
    qpath = root / QUESTIONNAIRE
    if qpath.exists() and not force:
        print(f"[form] {qpath} 已存在，未覆盖（要重写加 --force）")
        return 0
    qpath.write_text(_questionnaire_text(root, images), encoding="utf-8", newline="\n")
    print(f"[form] 已生成 {qpath}（{len(images)} 页 × 6 题）")
    print("[form] 填完后运行: python scripts/test2_tool.py build")
    return 0


def cmd_check(root: Path) -> int:
    rows = collect(root)
    if not rows:
        print(f"[check] {root / 'images'} 里还没有照片")
        return 1
    return 0 if _report(rows) == 0 else 1


def cmd_build(root: Path, splits_dir: Path, reviewed: bool, allow_partial: bool = False) -> int:
    rows = collect(root)
    if not rows:
        print(f"[build] {root / 'images'} 里还没有照片")
        return 1
    missing = [r for r in rows if not r["filled"]]
    if missing:
        if not allow_partial:
            _report(rows)
            print("[build] 有页面未答完，先补 问卷.md 再跑（或加 --allow-partial 只建已答的页）")
            return 1
        print(f"[build] --allow-partial：跳过 {len(missing)} 个未答页 "
              f"({', '.join(r['page'] for r in missing[:5])}{'…' if len(missing) > 5 else ''})")
        rows = [r for r in rows if r["filled"]]
        if not rows:
            print("[build] 没有任何页面答完")
            return 1
    _report(rows)

    answers_dir = root / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    out = splits_dir / "test2.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            (answers_dir / (r["page"] + ".md")).write_text(
                r["markdown"] + "\n", encoding="utf-8", newline="\n")
            f.write(json.dumps({"image": "images/" + r["image"], "markdown": r["markdown"],
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
            "answer_sha256": _sha256(answers_dir / (r["page"] + ".md")),
            "answer_chars": r["chars"],
            "box_placeholders": r["box"],
            "formulas": r["formulas"],
        } for r in rows],
    }
    (root / "test2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"\n[build] 已写出 {out}（{len(rows)} 页）与 {root / 'test2_manifest.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="test2: one questionnaire, two commands.")
    ap.add_argument("action", choices=["form", "check", "build"])
    ap.add_argument("--root", default="datasets/handwriting_notes/test2")
    ap.add_argument("--splits-dir", default="datasets/handwriting_notes/splits")
    ap.add_argument("--mark-reviewed", action="store_true",
                    help="把 reviewed 置为 true（人工过完一遍再加）")
    ap.add_argument("--force", action="store_true", help="覆盖已有问卷")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build 时跳过未答页，只用已答的页建集")
    args = ap.parse_args()
    root = Path(args.root)
    if args.action == "form":
        return cmd_form(root, args.force)
    if args.action == "check":
        return cmd_check(root)
    return cmd_build(root, Path(args.splits_dir), args.mark_reviewed, args.allow_partial)


if __name__ == "__main__":
    sys.exit(main())
