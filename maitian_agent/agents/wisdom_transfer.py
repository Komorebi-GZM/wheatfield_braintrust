"""
智慧传承Agent
手写教案OCR识别→结构化→入库→风格学习
性能要求: ≤5秒/张
"""
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from easyocr import Reader
import os
from dotenv import load_dotenv

from .base import BaseAgent

load_dotenv()


class WisdomTransferAgent(BaseAgent):
    """智慧传承Agent - 手写教案数字化传承"""

    def __init__(self, llm: Optional[ChatOpenAI] = None, ocr_reader: Optional[Reader] = None):
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

        if ocr_reader is None:
            self.ocr = Reader(['ch_sim', 'en'], gpu=False)
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
            result = self.ocr.readtext(image_path, detail=1)
            ocr_text = ""
            for detection in result:
                ocr_text += detection[1] + "\n"
            return ocr_text.strip()
        except Exception as e:
            raise RuntimeError(f"OCR识别失败: {str(e)}")

    def run(self, image_path: str, **kwargs) -> str:
        """执行手写教案识别与结构化

        Args:
            image_path: 手写教案图片路径

        Returns:
            结构化处理后的教案内容
        """
        try:
            ocr_text = self.recognize_handwriting(image_path)
            if not ocr_text:
                return "未能识别出文字，请确保图片清晰"

            chain = self.build_chain()
            structured_result = chain.invoke({"ocr_text": ocr_text})

            return f"### OCR识别结果\n{ocr_text}\n\n### 结构化结果\n{structured_result}"
        except Exception as e:
            return f"处理手写教案时出错：{str(e)}"

    def run_batch(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量处理多张手写教案

        Args:
            image_paths: 图片路径列表

        Returns:
            处理结果列表
        """
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
        """保存到知识库（待集成RAG）

        Args:
            structured_content: 结构化内容
            metadata: 元数据

        Returns:
            是否保存成功
        """
        # TODO: 集成RAG知识库存储
        raise NotImplementedError("知识库存储功能待开发")
