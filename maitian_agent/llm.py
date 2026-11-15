"""
厂商无关的对话模型工厂

唯一入口 ``get_chat_model()``：从 settings 读取 OpenAI 兼容配置构造
``ChatOpenAI``；未配置时抛出 ``LLMNotConfiguredError``。工厂内不做任何网络请求。

为方便测试与依赖注入，提供 ``set_chat_model`` / ``reset_chat_model``：
设置覆盖后 ``get_chat_model()`` 直接返回覆盖模型，不读配置、不构造 ChatOpenAI。
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from maitian_agent.config.settings import get_settings


class LLMNotConfiguredError(RuntimeError):
    """未配置大模型时抛出；``code`` 供 API 层转换为明确失败码。"""

    code = "LLM_NOT_CONFIGURED"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or "未配置大模型，请设置 LLM_BASE_URL、LLM_MODEL、LLM_API_KEY"
        super().__init__(self.message)


# 测试/依赖注入用的进程级覆盖模型
_override: BaseChatModel | None = None


def get_chat_model() -> BaseChatModel:
    """返回对话模型实例。

    - 已设置覆盖模型：直接返回，不读配置、不构造 ChatOpenAI。
    - 未配置三项必填：抛出 ``LLMNotConfiguredError``（带 code）。
    - 已配置：用 ``langchain_openai.ChatOpenAI`` 构造，不写任何厂商默认值。
    """
    if _override is not None:
        return _override

    settings = get_settings()
    base_url = settings.llm_base_url
    model = settings.llm_model
    api_key = settings.get_llm_api_key()
    if not base_url or not model or not api_key:
        raise LLMNotConfiguredError()

    # 延迟导入，避免未配置场景下无谓加载
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def set_chat_model(model: BaseChatModel | None) -> None:
    """设置覆盖模型（传 None 等同于清除覆盖）。"""
    global _override
    _override = model


def reset_chat_model() -> None:
    """清除覆盖模型，恢复从配置构造的行为。"""
    global _override
    _override = None
