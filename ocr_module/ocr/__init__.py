"""OCR module: handwriting course-note OCR via Xiaomi-OCR-0 (QLoRA).

本仓库仅承载 OCR 能力：模型加载 / 推理 / 评测 / 数据管线。
对外主入口见 `ocr.engine.recognize_page`。
"""

from .engine import OCREngine, recognize_page  # noqa: F401
from .schema import make_result, OUTPUT_SCHEMA  # noqa: F401

__all__ = ["OCREngine", "recognize_page", "make_result", "OUTPUT_SCHEMA"]
__version__ = "0.1.0"
