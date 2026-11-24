"""
智慧传承Agent
手写教案OCR识别→结构化→入库→风格学习

OCR 后端可切换（settings.ocr_type）：
- "easyocr"（默认，向后兼容）
- "xiaomi" / "xiaomi_structured"（ocr_module 的 Xiaomi-OCR-0，直接输出结构化 Markdown）
"""
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from .base import BaseAgent

try:
    from ..config.settings import settings
except Exception:  # 允许在缺少 pydantic-settings 的环境下导入
    settings = None

load_dotenv()


def _cfg(name: str, default=None):
    return getattr(settings, name, default) if settings is not None else default


class WisdomTransferAgent(BaseAgent):
    """智慧传承Agent - 手写教案数字化传承"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        ocr_processor=None,
        ocr_reader=None,
    ):
        super().__init__(
            name="WisdomTransferAgent",
            description="老教师经验一键传承，手写教案识别与结构化"
        )
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("MODEL_NAME", "deepseek-v2")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.3
            )
        else:
            self.llm = llm

        # 选择 OCR 后端：优先显式 processor > 旧式 Reader > 按配置创建
        self._legacy_reader = None
        if ocr_processor is not None:
            self.ocr = ocr_processor
        elif ocr_reader is not None:
            self._legacy_reader = ocr_reader
            self.ocr = None
        else:
            from ..tools.ocr import OCRProcessor  # 延迟导入
            self.ocr = OCRProcessor.create(
                ocr_type=_cfg("ocr_type", "easyocr"),
                languages=_cfg("ocr_languages", ["ch_sim", "en"]),
                gpu=_cfg("ocr_gpu", False),
                base_model=_cfg("ocr_base_model", None),
                adapter_dir=_cfg("ocr_adapter_dir", None),
                load_in_4bit=_cfg("ocr_load_in_4bit", True),
            )
        self.is_structured_ocr = bool(getattr(self.ocr, "is_structured", False))

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
        """识别手写文字（文本或结构化 Markdown，取决于 OCR 后端）"""
        try:
            if self._legacy_reader is not None:
                result = self._legacy_reader.readtext(image_path, detail=1)
                return "\n".join(d[1] for d in result).strip()
            return self.ocr.recognize(image_path).strip()
        except Exception as e:
            raise RuntimeError(f"OCR识别失败: {str(e)}")

    def run(self, image_path: str, **kwargs) -> str:
        """执行手写教案识别与结构化

        结构化后端（xiaomi）已直接输出带结构的 Markdown，无需再做 LLM 结构化；
        文本后端（easyocr）走原有 LLM 结构化链。
        """
        try:
            ocr_text = self.recognize_handwriting(image_path)
            if not ocr_text:
                return "未能识别出文字，请确保图片清晰"

            if self.is_structured_ocr:
                return f"### 结构化结果\n{ocr_text}"

            chain = self.build_chain()
            structured_result = chain.invoke({"ocr_text": ocr_text})
            return f"### OCR识别结果\n{ocr_text}\n\n### 结构化结果\n{structured_result}"
        except Exception as e:
            return f"处理手写教案时出错：{str(e)}"

    def run_batch(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量处理多张手写教案"""
        results = []
        for image_path in image_paths:
            result = self.run(image_path)
            results.append({
                "image_path": image_path,
                "result": result,
                "success": not result.startswith("处理手写教案时出错")
            })
        return results

    def save_to_knowledge_base(self, structured_content: str, metadata: Dict[str, Any]) -> bool:
        """保存到知识库（待集成RAG）"""
        # TODO: 集成RAG知识库存储
        raise NotImplementedError("知识库存储功能待开发")
