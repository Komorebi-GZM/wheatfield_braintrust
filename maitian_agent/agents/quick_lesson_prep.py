"""
极速备课Agent
语音驱动全流程备课，生成乡土化教案
性能要求: ≤10秒
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from maitian_agent.agents.base import BaseAgent
from maitian_agent.agents.results import AgentResult, LessonPrepInput, TextData


class QuickLessonPrepAgent(BaseAgent):
    """极速备课Agent - 生成乡土化教案"""

    def __init__(self, llm: BaseChatModel | None = None):
        super().__init__(
            llm=llm, name="QuickLessonPrepAgent", description="乡村专属极速备课，10秒生成乡土化教案"
        )

        self.lesson_plan_template = ChatPromptTemplate.from_template(
            """你是一位经验丰富的乡村教师，擅长结合乡村实际情境设计教案。请根据以下信息生成一份详细的教案：

学科：{subject}
年级：{grade}
课题：{topic}
乡村特色情境：{rural_context}

教案应包含以下部分：
1. 教学目标
2. 教学重难点
3. 教学准备
4. 教学过程（导入、新授、练习、总结）
5. 作业布置
6. 板书设计

要求：
- 结合乡村实际情境，使用贴近农村生活的例子
- 语言通俗易懂，符合该年级学生的认知水平
- 教案结构清晰，可直接用于课堂教学
- 融入乡土元素，增强学生的学习兴趣
"""
        )
        self._chain = None

    def build_chain(self):
        """构建备课链"""
        if self._chain is None:
            self._chain = self.lesson_plan_template | self.llm | StrOutputParser()
        return self._chain

    def run(
        self,
        input_data: LessonPrepInput | dict | None = None,
        grade: str | None = None,
        topic: str | None = None,
        rural_context: str | None = "",
        **kwargs,
    ) -> AgentResult[TextData]:
        """执行备课流程

        Args:
            subject: 学科
            grade: 年级
            topic: 课题
            rural_context: 乡村特色情境

        Returns:
            生成的教案内容
        """
        if isinstance(input_data, LessonPrepInput):
            values = input_data.model_dump()
        elif isinstance(input_data, dict):
            values = input_data
        else:
            values = {
                "subject": input_data,
                "grade": grade,
                "topic": topic,
                "rural_context": rural_context,
            }
        values.update(kwargs)
        request = LessonPrepInput.model_validate(values)
        chain = self.build_chain()
        result = chain.invoke(
            {
                "subject": request.subject,
                "grade": request.grade,
                "topic": request.topic,
                "rural_context": request.rural_context or "结合乡村实际教学",
            }
        )
        return AgentResult(status="success", data=TextData(text=result))

    def run_with_voice(self, audio_path: str, **kwargs) -> str:
        """通过语音输入执行备课（待集成Whisper）

        Args:
            audio_path: 语音文件路径

        Returns:
            生成的教案内容
        """
        # TODO: 集成Whisper API进行语音转文本
        raise NotImplementedError("语音备课功能待开发")

    def _build_response(self, lesson_plan: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """构建响应"""
        return {
            "success": True,
            "result": lesson_plan,
            "agent": self.name,
            "metadata": {
                "subject": metadata.get("subject"),
                "grade": metadata.get("grade"),
                "topic": metadata.get("topic"),
                "response_time": metadata.get("response_time", 0),
            },
        }
