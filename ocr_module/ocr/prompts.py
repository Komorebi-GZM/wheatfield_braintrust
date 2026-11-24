"""Prompt templates for structured handwriting OCR.

Output protocol is deliberately strict: the model must emit Markdown with
LaTeX formulas and Markdown/OTSL tables, no prose wrappers.

v2 note (inference alignment):
`datasets/handwriting_notes/splits/*_swift.jsonl` trained the model with a single
user turn whose content is `<image>\\n` + INSTRUCTION and *no* system message.
`scripts/probe_protocol.py` shows that rendering
`[{"role": "user", "content": [image, "\\n" + INSTRUCTION]}]` with the HF chat
template reproduces the Swift training prompt token-for-token (prompt_identical).
Adding SYSTEM_PROMPT changes the prompt from token index 1 and appends ~240
tokens, so it is **off by default** and kept only for side experiments.
"""

INSTRUCTION = "请转写这页手写课程笔记为结构化 Markdown（保留标题、列表、表格与公式）。"

# Kept for reference / optional side experiments only (NOT used for the
# base-vs-finetuned comparison, because training never saw a system message).
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

USER_INSTRUCTION = "<image>\n" + INSTRUCTION


def build_messages(image_path: str, use_system_prompt: bool = False,
                   instruction: str | None = None):
    """Return chat messages that match the training rendering by default.

    ``instruction`` overrides the user turn only; the image token and the
    newline that ms-swift produced for ``"<image>\\n"`` are unchanged, so a
    prompt experiment differs from the frozen protocol in exactly one string.
    """
    msgs = []
    if use_system_prompt:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": [
        {"type": "image", "url": image_path},
        # the newline after the image is what ms-swift produced for "<image>\n"
        {"type": "text", "text": "\n" + (instruction or INSTRUCTION)},
    ]})
    return msgs
