"""Prompt templates for structured handwriting OCR.

Output protocol is deliberately strict: the model must emit Markdown with
LaTeX formulas and Markdown/OTSL tables, no prose wrappers.
"""

SYSTEM_PROMPT = """你是一名专业的手写课程笔记 OCR 与结构化助手。
请把用户提供的**手写笔记图片**转写为**结构化 Markdown**，并严格遵守：

1. 只输出 Markdown 正文，不要任何解释性前后缀、不要代码围栏包裹整篇。
2. 结构：
   - 标题用 `#`/`##`/`###`，层级依据笔记实际缩进与编号推断；
   - 分点用 `-`（无序）或 `1.`（有序），保留嵌套缩进；
   - 表格用 Markdown 表格语法；
3. 公式：
   - 行内公式用 `$...$`，行间公式用 `$$...$$`；
   - 公式转写为**合法 LaTeX**（如 `\\frac{a}{b}`、`\\int_0^1`）。
4. 只转写**可见内容**：无法辨认的字用 `□` 占位，**不要臆造或补全**。
5. 插图/示意图不识别内容，用 `> [图示：简述]` 占位。
"""

# One compact few-shot example (image token -> structured markdown).
FEW_SHOT = [
    {
        "role": "user",
        "content": "<image>\n请转写这页手写课程笔记为结构化 Markdown（保留标题、列表、表格与公式）。",
    },
    {
        "role": "assistant",
        "content": (
            "# 第三章 电磁感应\n\n"
            "## 3.1 法拉第电磁感应定律\n\n"
            "- 感应电动势与磁通量变化率成正比\n"
            "- 方向由楞次定律判定\n\n"
            "$$ \\varepsilon = -\\frac{d\\Phi}{dt} $$\n\n"
            "| 物理量 | 符号 | 单位 |\n|---|---|---|\n"
            "| 磁通量 | $\\Phi$ | Wb |\n"
            "| 电动势 | $\\varepsilon$ | V |\n"
        ),
    },
]

USER_INSTRUCTION = (
    "<image>\n请转写这页手写课程笔记为结构化 Markdown（保留标题、列表、表格与公式）。"
)


def build_messages(use_few_shot: bool = True):
    """Return chat messages (without system) for the processor chat template."""
    msgs = []
    if use_few_shot:
        msgs.extend(FEW_SHOT)
    msgs.append({"role": "user", "content": USER_INSTRUCTION})
    return msgs
