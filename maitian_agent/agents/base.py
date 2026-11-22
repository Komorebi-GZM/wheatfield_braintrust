"""
Agent基类
定义所有Agent的公共接口和方法
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel

from maitian_agent.agents.results import AgentResult
from maitian_agent.llm import get_chat_model

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent基类

    构造时不调用 ``get_chat_model()``：未传入 llm 时保留为 None，
    首次真正使用（``self.llm``）时才从工厂懒加载，避免无 Key 时启动崩溃。
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        memory: Any = None,
        name: str = "BaseAgent",
        description: str = "基础Agent",
    ):
        self._llm = llm
        self.memory = memory
        self.name = name
        self.description = description
        self._chain = None
        logger.info(f"初始化Agent: {name}")

    @property
    def llm(self) -> BaseChatModel:
        """懒加载对话模型：注入优先，否则首次使用时从工厂获取。"""
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    @llm.setter
    def llm(self, value: BaseChatModel | None) -> None:
        self._llm = value

    @abstractmethod
    def build_chain(self) -> Any:
        """构建Agent链"""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AgentResult[Any]:
        """执行Agent"""

    def _validate_input(self, input_data: dict[str, Any], required_keys: list) -> None:
        """验证输入数据"""
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"缺少必需参数: {', '.join(missing_keys)}")

    def _format_output(self, result: Any, metadata: dict | None = None) -> dict[str, Any]:
        """格式化输出"""
        return {"success": True, "result": result, "agent": self.name, "metadata": metadata or {}}

    def _handle_error(self, error: Exception) -> dict[str, Any]:
        """处理错误"""
        logger.error("%s执行错误（内部异常已脱敏）", self.name)
        return {"success": False, "error": "服务处理失败，请稍后重试", "agent": self.name}
