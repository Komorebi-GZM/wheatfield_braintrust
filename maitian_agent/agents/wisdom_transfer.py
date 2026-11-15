"""
智慧传承Agent
手写教案OCR识别→结构化→入库→风格学习
性能要求: ≤5秒/张

OCR 复用 ``EasyOCRProcessor`` 的懒加载：构造时不 import easyocr/torch，
首次真正识别时才加载；也可注入已有 reader。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from maitian_agent.agents.base import BaseAgent
from maitian_agent.tools.ocr import EasyOCRProcessor

if TYPE_CHECKING:
    from easyocr import Reader  # type: ignore[import-not-found]


class WisdomTransferAgent(BaseAgent):
    """智慧传承Agent - 手写教案数字化传承"""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        ocr_reader: Reader | None = None,
    ):
        super().__init__(
            llm=llm,
            name="WisdomTransferAgent",
            description="老教师经验一键传承，手写教案识别与结构化"
        )
        if ocr_reader is None:
            self.ocr = EasyOCRProcessor(languages=["ch_sim", "en"], gpu=False)
        else:
            self.ocr = ocr_reader

        self.structure_template = ChatPromptTemplate.from_template(
            """你是一位教育专家，请将以下手写教案的OCR识别结果结构化处理：

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
        self._chain = None

    def build_chain(self):
        """构建结构化处理链"""
        if self._chain is None:
            self._chain = self.structure_template | self.llm | StrOutputParser()
        return self._chain

    def recognize_handwriting(self, image_path: str) -> str:
        """识别手写文字

        Args:
            image_path: 图片路径

        Returns:
            识别出的文本
        """
        try:
            # EasyOCRProcessor 走带锁的 recognize，避免并发重复构造/调用 Reader
            if isinstance(self.ocr, EasyOCRProcessor):
                return self.ocr.recognize(image_path, detail=1).strip()
            result = self.ocr.readtext(image_path, detail=1)
            return "\n".join(detection[1] for detection in result).strip()
        except Exception as e:
            raise RuntimeError(f"OCR识别失败: {e!s}")

    def run(self, image_path: str, **kwargs) -> str:
        """执行手写教案识别与结构化

        Args:
            image_path: 手写教案图片路径

        Returns:
            结构化处理后的教案内容
        """
        ocr_text = self.recognize_handwriting(image_path)
        if not ocr_text:
            return "未能识别出文字，请确保图片清晰"

        chain = self.build_chain()
        structured_result = chain.invoke({"ocr_text": ocr_text})

        return f"### OCR识别结果\n{ocr_text}\n\n### 结构化结果\n{structured_result}"

    def run_batch(self, image_paths: list[str], **kwargs) -> list[dict[str, Any]]:
        """批量处理多张手写教案

        Args:
            image_paths: 图片路径列表

        Returns:
            处理结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.run(image_path)
            except Exception:
                results.append({
                    "image_path": image_path,
                    "result": "处理失败，请稍后重试",
                    "success": False,
                })
            else:
                results.append({
                    "image_path": image_path,
                    "result": result,
                    "success": True,
                })
        return results

    def save_to_knowledge_base(self, structured_content: str, metadata: dict[str, Any]) -> bool:
        """保存到知识库（待集成RAG）

        Args:
            structured_content: 结构化内容
            metadata: 元数据

        Returns:
            是否保存成功
        """
        # TODO: 集成RAG知识库存储
        raise NotImplementedError("知识库存储功能待开发")
