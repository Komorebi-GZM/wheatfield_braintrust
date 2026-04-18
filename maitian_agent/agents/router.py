"""
意图路由Agent
LangChain Router 意图路由器
Few-Shot示例校验 + 路由结果二次校验
"""
from typing import Dict, Any, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
import os
from dotenv import load_dotenv

from .base import BaseAgent

load_dotenv()


class RouterAgent(BaseAgent):
    """意图路由Agent"""

    INTENTS = {
        "quick_lesson_prep": "乡村专属极速备课，生成乡土化教案",
        "wisdom_transfer": "老教师经验传承，手写教案识别与结构化",
        "classroom_companion": "课堂实时伴教，练习题生成与经典题检索",
        "material_recommend": "具象化素材推荐，教学视频/3D模型/动画",
        "meeting_notes": "教研纪要生成，录音转写与结构化报告",
        "general_chat": "通用对话，其他教育相关咨询"
    }

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        super().__init__(
            name="RouterAgent",
            description="意图路由器，智能识别用户需求并路由到对应Agent"
        )
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("MODEL_NAME", "deepseek-v2")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.1
            )
        else:
            self.llm = llm

        self.router_template = ChatPromptTemplate.from_template(
            """你是一个意图识别路由器，请分析用户输入并确定其意图。

可用意图：
{intent_list}

用户输入：{user_input}

请只返回一个意图名称，不要其他内容。
"""
        )

        self.validation_template = ChatPromptTemplate.from_template(
            """请验证以下意图识别结果是否正确：

用户输入：{user_input}
识别意图：{intent}
意图描述：{intent_description}

请回答"正确"或"错误"，如果错误请给出正确的意图。
"""
        )
        self._router_chain = None
        self._validation_chain = None

    def build_chain(self, chain_type: str = "router"):
        """构建指定类型的链"""
        if chain_type == "router":
            if self._router_chain is None:
                self._router_chain = self.router_template | self.llm | StrOutputParser()
            return self._router_chain
        else:
            if self._validation_chain is None:
                self._validation_chain = self.validation_template | self.llm | StrOutputParser()
            return self._validation_chain

    def route(self, user_input: str, **kwargs) -> str:
        """路由用户输入到对应Agent

        Args:
            user_input: 用户输入

        Returns:
            识别的意图名称
        """
        try:
            intent_list = "\n".join([f"- {k}: {v}" for k, v in self.INTENTS.items()])

            router_chain = self.build_chain("router")
            intent = router_chain.invoke({
                "intent_list": intent_list,
                "user_input": user_input
            }).strip()

            validated_intent = self._validate_intent(user_input, intent)
            return validated_intent

        except Exception as e:
            return "general_chat"

    def _validate_intent(self, user_input: str, intent: str) -> str:
        """验证意图识别结果

        Args:
            user_input: 用户输入
            intent: 识别的意图

        Returns:
            验证后的意图
        """
        if intent not in self.INTENTS:
            return "general_chat"

        try:
            validation_chain = self.build_chain("validation")
            result = validation_chain.invoke({
                "user_input": user_input,
                "intent": intent,
                "intent_description": self.INTENTS.get(intent, "")
            }).strip()

            if "错误" in result or "不正确" in result:
                return "general_chat"

            return intent
        except Exception:
            return intent

    def get_intent_info(self, intent: str) -> Dict[str, Any]:
        """获取意图信息"""
        return {
            "intent": intent,
            "description": self.INTENTS.get(intent, "未知意图"),
            "available_intents": list(self.INTENTS.keys())
        }

    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行路由"""
        user_input = input_data.get("user_input", "")
        if not user_input:
            return {"success": False, "error": "缺少用户输入"}

        intent = self.route(user_input)
        intent_info = self.get_intent_info(intent)

        return {
            "success": True,
            "intent": intent,
            "agent": self.name,
            "info": intent_info
        }
