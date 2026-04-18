"""
课堂实时伴教Agent
课堂极速出题、经典题检索
性能要求: ≤3秒
"""
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from .base import BaseAgent

load_dotenv()


class ClassroomCompanionAgent(BaseAgent):
    """课堂实时伴教Agent"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        super().__init__(
            name="ClassroomCompanionAgent",
            description="课堂实时伴教，3秒极速出题与经典题检索"
        )
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("MODEL_NAME", "deepseek-v2")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.5
            )
        else:
            self.llm = llm

        self.quiz_template = ChatPromptTemplate.from_template(
            """你是一位专业的乡村教师，请根据以下信息快速生成练习题：

学科：{subject}
年级：{grade}
课题：{topic}
知识点：{knowledge_points}
题目数量：{question_count}道
题目类型：{question_types}

要求：
- 题目紧扣知识点，难度适中
- 符合乡村学生的认知水平
- 题目清晰，答案明确
- 融入乡土元素，增加趣味性
"""
        )

        self.retrieval_template = ChatPromptTemplate.from_template(
            """请在知识库中检索与以下内容相关的经典题目和教学资源：

学科：{subject}
年级：{grade}
课题：{topic}
知识点：{knowledge_points}

请返回检索到的相关题目和资源简要说明。
"""
        )
        self._quiz_chain = None
        self._retrieval_chain = None

    def build_chain(self, chain_type: str = "quiz"):
        """构建指定类型的链"""
        if chain_type == "quiz":
            if self._quiz_chain is None:
                self._quiz_chain = self.quiz_template | self.llm | StrOutputParser()
            return self._quiz_chain
        else:
            if self._retrieval_chain is None:
                self._retrieval_chain = self.retrieval_template | self.llm | StrOutputParser()
            return self._retrieval_chain

    def generate_quiz(
        self,
        subject: str,
        grade: str,
        topic: str,
        knowledge_points: str = "",
        question_count: int = 5,
        question_types: str = "选择题、填空题",
        **kwargs
    ) -> str:
        """生成练习题

        Args:
            subject: 学科
            grade: 年级
            topic: 课题
            knowledge_points: 知识点
            question_count: 题目数量
            question_types: 题目类型

        Returns:
            生成的练习题
        """
        try:
            chain = self.build_chain("quiz")
            result = chain.invoke({
                "subject": subject,
                "grade": grade,
                "topic": topic,
                "knowledge_points": knowledge_points or "本课主要内容",
                "question_count": question_count,
                "question_types": question_types
            })
            return result
        except Exception as e:
            return f"生成练习题时出错：{str(e)}"

    def retrieve_classic_questions(
        self,
        subject: str,
        grade: str,
        topic: str,
        knowledge_points: str = "",
        **kwargs
    ) -> str:
        """检索经典题目

        Args:
            subject: 学科
            grade: 年级
            topic: 课题
            knowledge_points: 知识点

        Returns:
            检索到的经典题目
        """
        try:
            chain = self.build_chain("retrieval")
            result = chain.invoke({
                "subject": subject,
                "grade": grade,
                "topic": topic,
                "knowledge_points": knowledge_points or "本课主要内容"
            })
            return result
        except Exception as e:
            return f"检索经典题目时出错：{str(e)}"

    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行伴教功能"""
        action = input_data.get("action", "quiz")

        if action == "quiz":
            result = self.generate_quiz(**input_data)
        elif action == "retrieve":
            result = self.retrieve_classic_questions(**input_data)
        else:
            return {"success": False, "error": f"未知操作: {action}"}

        return self._format_output(result)
