"""
具象化素材Agent
自动匹配科普视频/3D模型/动画素材
"""
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from .base import BaseAgent

load_dotenv()


class MaterialAgent(BaseAgent):
    """具象化素材Agent"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        super().__init__(
            name="MaterialAgent",
            description="具象化素材匹配，自动推荐教学视频/3D模型/动画"
        )
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("MODEL_NAME", "deepseek-v2")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.7
            )
        else:
            self.llm = llm

        self.material_template = ChatPromptTemplate.from_template(
            """你是一位教育资源专家，请为以下教学内容推荐具象化素材：

学科：{subject}
年级：{grade}
课题：{topic}
知识点：{knowledge_points}
乡村情境：{rural_context}

请推荐适合的素材类型：
1. 科普视频 - 简短有趣的知识讲解视频
2. 3D模型 - 可视化展示抽象概念
3. 动画演示 - 生动有趣的动画说明
4. 图片素材 - 贴近乡村生活的实例图片

要求：
- 素材应贴近乡村学生的生活经验
- 趣味性强，能激发学习兴趣
- 简短精炼，适合课堂使用
- 标注素材来源和获取方式
"""
        )
        self._chain = None

    def build_chain(self):
        """构建素材推荐链"""
        if self._chain is None:
            self._chain = self.material_template | self.llm | StrOutputParser()
        return self._chain

    def recommend_materials(
        self,
        subject: str,
        grade: str,
        topic: str,
        knowledge_points: str = "",
        rural_context: str = "",
        **kwargs
    ) -> str:
        """推荐教学素材

        Args:
            subject: 学科
            grade: 年级
            topic: 课题
            knowledge_points: 知识点
            rural_context: 乡村情境

        Returns:
            素材推荐结果
        """
        try:
            chain = self.build_chain()
            result = chain.invoke({
                "subject": subject,
                "grade": grade,
                "topic": topic,
                "knowledge_points": knowledge_points or "本课主要内容",
                "rural_context": rural_context or "乡村实际情境"
            })
            return result
        except Exception as e:
            return f"推荐素材时出错：{str(e)}"

    def search_material_library(
        self,
        keyword: str,
        material_type: str = "all",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """搜索素材库（待集成素材库）

        Args:
            keyword: 搜索关键词
            material_type: 素材类型

        Returns:
            素材列表
        """
        # TODO: 集成素材库检索
        raise NotImplementedError("素材库检索功能待开发")

    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行素材推荐"""
        result = self.recommend_materials(**input_data)
        return self._format_output(result)
