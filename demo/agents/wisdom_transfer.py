"""Demo 智慧传承 Agent —— 使用厂商无关模型工厂，OCR 懒加载。"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from maitian_agent.llm import get_chat_model


class WisdomTransferAgent:
    def __init__(self, llm: BaseChatModel | None = None, ocr_reader=None):
        self._llm = llm
        self._ocr = ocr_reader

        # 结构化模板
        self.structure_template = ChatPromptTemplate.from_template(
            """
            你是一位教育专家，请将以下手写教案的OCR识别结果结构化处理：

            {ocr_text}

            请按照以下格式进行结构化：
            1. 学科：
            2. 年级：
            3. 课题：
            4. 教学目标：
            5. 教学重难点：
            6. 教学过程：
            7. 作业布置：
            8. 教学反思：

            要求：
            - 提取关键信息，忽略无关内容
            - 保持原文意，语言通顺
            - 结构清晰，层次分明
            """
        )

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    @property
    def ocr(self):
        if self._ocr is None:
            from maitian_agent.config.settings import get_settings

            backend = getattr(get_settings(), "ocr_backend", "none")
            if backend in (None, "", "none"):
                raise RuntimeError(
                    "未启用 OCR（OCR_BACKEND=none），无法识别手写教案；"
                    "如需启用请设置 OCR_BACKEND=easyocr。"
                )
            from easyocr import Reader  # type: ignore[import-not-found]

            self._ocr = Reader(['ch_sim', 'en'], gpu=False)
        return self._ocr

    def run(self, image_path):
        """执行手写教案识别与结构化"""
        result = self.ocr.readtext(image_path, detail=1)

        ocr_text = ""
        for detection in result:
            ocr_text += detection[1] + "\n"

        messages = self.structure_template.format_messages(ocr_text=ocr_text)
        structured_result = self.llm.invoke(messages)

        return f"### OCR识别结果\n{ocr_text}\n\n### 结构化结果\n{structured_result.content}"
