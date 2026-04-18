"""
Agent基类
定义所有Agent的公共接口和方法
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.callbacks import CallbackManagerForChainRun
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent基类"""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[BaseMemory] = None,
        name: str = "BaseAgent",
        description: str = "基础Agent",
    ):
        self.llm = llm
        self.memory = memory
        self.name = name
        self.description = description
        self._chain = None
        logger.info(f"初始化Agent: {name}")

    @abstractmethod
    def build_chain(self) -> Any:
        """构建Agent链"""
        pass

    @abstractmethod
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行Agent"""
        pass

    def _validate_input(self, input_data: Dict[str, Any], required_keys: list) -> None:
        """验证输入数据"""
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"缺少必需参数: {', '.join(missing_keys)}")

    def _format_output(self, result: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """格式化输出"""
        return {
            "success": True,
            "result": result,
            "agent": self.name,
            "metadata": metadata or {}
        }

    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """处理错误"""
        logger.error(f"{self.name}执行错误: {str(error)}")
        return {
            "success": False,
            "error": str(error),
            "agent": self.name
        }
