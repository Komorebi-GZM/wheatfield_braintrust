"""
教研纪要Agent
录音转写→结构化教研报告→成果入库
性能要求: 转写≥95%准确率
"""
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from .base import BaseAgent

load_dotenv()


class MeetingNotesAgent(BaseAgent):
    """教研纪要Agent"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        super().__init__(
            name="MeetingNotesAgent",
            description="教研纪要生成，录音转写与结构化报告"
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

        self.structure_template = ChatPromptTemplate.from_template(
            """你是一位专业的教研秘书，请将以下教研会议记录整理成结构化报告：

原始记录：
{meeting_transcript}

请按照以下格式整理：
1. 会议基本信息
   - 时间：
   - 地点：
   - 参与人员：
   - 主题：

2. 会议议题
   （列出主要讨论议题）

3. 讨论内容摘要
   （概括性地总结讨论要点）

4. 决议事项
   （列出会议达成的共识和决定）

5. 行动计划
   （列出后续行动项及负责人）

6. 教研成果
   （提取可沉淀的教学经验和成果）

要求：
- 语言精炼，要点清晰
- 保留关键信息和专业术语
- 行动计划具体可执行
"""
        )
        self._chain = None

    def build_chain(self):
        """构建结构化处理链"""
        if self._chain is None:
            self._chain = self.structure_template | self.llm | StrOutputParser()
        return self._chain

    def transcribe_audio(self, audio_path: str) -> str:
        """转写音频（待集成Whisper）

        Args:
            audio_path: 音频文件路径

        Returns:
            转写文本
        """
        # TODO: 集成Whisper API进行音频转写
        raise NotImplementedError("音频转写功能待开发")

    def process_meeting_notes(
        self,
        meeting_transcript: str,
        **kwargs
    ) -> str:
        """处理教研纪要

        Args:
            meeting_transcript: 会议记录文本

        Returns:
            结构化报告
        """
        try:
            chain = self.build_chain()
            result = chain.invoke({"meeting_transcript": meeting_transcript})
            return result
        except Exception as e:
            return f"处理教研纪要时出错：{str(e)}"

    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行教研纪要生成"""
        if "audio_path" in input_data:
            transcript = self.transcribe_audio(input_data["audio_path"])
        elif "transcript" in input_data:
            transcript = input_data["transcript"]
        else:
            return {"success": False, "error": "缺少音频文件或转写文本"}

        result = self.process_meeting_notes(transcript)
        return self._format_output(result)
