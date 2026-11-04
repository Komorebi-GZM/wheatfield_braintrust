"""Output schema for the OCR engine result."""

from typing import TypedDict, List


class OCRResult(TypedDict):
    markdown: str          # 结构化 Markdown（含 LaTeX / 表格）
    structure_ok: bool     # 结构校验是否通过
    latex_issues: List[str]  # 非法/可疑公式列表
    tables: int            # 检测到的表格数量
    formulas: int          # 检测到的公式数量
    model: str             # 模型标识（基座 + adapter 版本）


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "markdown": {"type": "string"},
        "structure_ok": {"type": "boolean"},
        "latex_issues": {"type": "array", "items": {"type": "string"}},
        "tables": {"type": "integer", "minimum": 0},
        "formulas": {"type": "integer", "minimum": 0},
        "model": {"type": "string"},
    },
    "required": ["markdown", "structure_ok", "latex_issues", "tables", "formulas", "model"],
    "additionalProperties": False,
}


def make_result(markdown: str, structure_ok: bool, latex_issues: List[str],
                tables: int, formulas: int, model: str) -> OCRResult:
    return {
        "markdown": markdown,
        "structure_ok": structure_ok,
        "latex_issues": list(latex_issues),
        "tables": int(tables),
        "formulas": int(formulas),
        "model": model,
    }
