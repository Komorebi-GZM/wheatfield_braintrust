"""Synthetic handwritten course-note page generator (weak supervision).

Renders structured note text (headings / bullets / **formulas**) onto a
paper-like background with jitter, perspective and noise, producing:
  - synthetic page images under `--out/dataset/images/`
  - `--out/manifest.jsonl`: {"image": rel_path, "markdown": source_markdown}

Formulas are rendered with matplotlib mathtext when available (handwriting-like
perturbation applied); otherwise drawn as plain text approximations.

NOTE: synthetic data is for training augmentation only. It never enters the
fixed validation/test split (see prepare_dataset.py).

CLI:
    python scripts/build_synth.py --count 800 --out data/synth
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


# --------------------------------------------------------------------------- #
# Content templates (formula-heavy, few tables — matches the confirmed need)
# --------------------------------------------------------------------------- #
_TITLES = ["电磁感应", "牛顿第二定律", "化学反应速率", "三角函数恒等变换",
           "细胞呼吸", "概率分布", "导数应用", "电路分析", "热力学第一定律"]
_BULLETS = [
    "定义：物理量与其变化率之间的定量关系",
    "条件：在理想模型下近似成立",
    "推论：可用于求解边界问题",
    "注意：单位统一，符号方向约定",
]
_FORMULAS = [
    r"\varepsilon = -\frac{d\Phi}{dt}",
    r"F = m a",
    r"v = v_0 + a t",
    r"\sin^2\theta + \cos^2\theta = 1",
    r"\int_{0}^{1} x^{2}\,dx = \frac{1}{3}",
    r"PV = nRT",
    r"e^{i\pi} + 1 = 0",
    r"\frac{1}{2}mv^{2} = mgh",
    r"\lim_{x\to 0}\frac{\sin x}{x} = 1",
    r"\Delta H = Q - W",
]
_TABLE_ROWS = [["物理量", "符号", "单位"], ["磁通量", "$\\Phi$", "Wb"], ["电动势", "$\\varepsilon$", "V"]]


def _sample_section(rng: random.Random) -> Dict:
    title = rng.choice(_TITLES)
    sub = rng.choice(["定义与原理", "公式推导", "例题分析", "易错点"])
    bullets = rng.sample(_BULLETS, k=rng.randint(1, 3))
    formulas = rng.sample(_FORMULAS, k=rng.randint(1, 3))
    include_table = rng.random() < 0.15  # tables are rare
    return {"title": title, "sub": sub, "bullets": bullets,
            "formulas": formulas, "table": _TABLE_ROWS if include_table else None}


def _to_markdown(sec: Dict) -> str:
    lines = [f"# {sec['title']}", "", f"## {sec['sub']}", ""]
    for b in sec["bullets"]:
        lines.append(f"- {b}")
    lines.append("")
    for f in sec["formulas"]:
        lines.append(f"$$ {f} $$")
        lines.append("")
    if sec["table"]:
        rows = sec["table"]
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
        for r in rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        lines.append("")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    candidates = [path] if path else []
    candidates += [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _render_formula(latex: str, width: int) -> Image.Image:
    """Render a LaTeX formula to a small image (white bg) or text fallback."""
    if _HAS_MPL:
        try:
            fig = plt.figure(figsize=(6, 0.8), dpi=200)
            fig.text(0.01, 0.5, f"${latex}$", fontsize=20, va="center")
            fig.canvas.draw()
            buf = __import__("io").BytesIO()
            fig.savefig(buf, format="png", transparent=False, facecolor="white")
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
            return im
        except Exception:
            pass
    img = Image.new("RGB", (width, 60), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 10), latex, fill="black", font=_load_font(None, 22))
    return img


def _paper_bg(w: int, h: int, rng: random.Random) -> Image.Image:
    bg = Image.new("RGB", (w, h), (252, 251, 246))
    d = ImageDraw.Draw(bg)
    if rng.random() < 0.5:  # ruled lines
        y = 90
        while y < h - 40:
            d.line([(50, y), (w - 50, y)], fill=(200, 210, 225), width=1)
            y += 46
    return bg


def render_page(sec: Dict, rng: random.Random, font_path: str | None,
                size=(1000, 1300)) -> Image.Image:
    w, h = size
    img = _paper_bg(w, h, rng)
    d = ImageDraw.Draw(img)
    x0, y = 70, 60
    jitter = lambda: rng.randint(-2, 2)

    # title
    f_title = _load_font(font_path, 40)
    d.text((x0, y), sec["title"], fill=(20, 20, 30), font=f_title)
    y += 58
    f_sub = _load_font(font_path, 32)
    d.text((x0 + 20, y), sec["sub"], fill=(40, 40, 60), font=f_sub)
    y += 52

    f_body = _load_font(font_path, 28)
    for b in sec["bullets"]:
        d.text((x0 + 20 + jitter(), y), "· " + b, fill=(30, 30, 30), font=f_body)
        y += 44

    y += 10
    for f in sec["formulas"]:
        fim = _render_formula(f, w - 2 * x0)
        fim = fim.resize((min(fim.width, w - 2 * x0), fim.height))
        img.paste(fim, (x0 + 30 + jitter(), y))
        y += fim.height + 20

    if sec["table"] and y < h - 160:
        rows = sec["table"]
        row_h = 46
        col_w = (w - 2 * x0) // len(rows[0])
        for ri, r in enumerate(rows):
            for ci, cell in enumerate(r):
                d.rectangle([x0 + ci * col_w, y + ri * row_h,
                             x0 + (ci + 1) * col_w, y + (ri + 1) * row_h],
                            outline=(90, 90, 90), width=1)
                d.text((x0 + ci * col_w + 8, y + ri * row_h + 8), cell,
                       fill=(20, 20, 20), font=_load_font(font_path, 24))

    # degrade: slight rotate + blur + noise
    img = img.rotate(rng.uniform(-2.0, 2.0), expand=False, fillcolor=(252, 251, 246))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--font", default=None, help="CJK handwriting TTF/TTC path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i in range(args.count):
            sec = _sample_section(rng)
            md = _to_markdown(sec)
            img = render_page(sec, rng, args.font)
            rel = f"images/synth_{i:05d}.png"
            img.save(os.path.join(args.out, rel))
            mf.write(json.dumps({"image": rel, "markdown": md}, ensure_ascii=False) + "\n")
    print(f"[build_synth] wrote {args.count} pages -> {args.out} (manifest: {manifest_path})")
    if not _HAS_MPL:
        print("[build_synth] matplotlib 不可用：公式以纯文本近似渲染；建议安装 matplotlib 以提高逼真度")


if __name__ == "__main__":
    main()
