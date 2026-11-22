"""
厂商无关的对话模型工厂

唯一入口 ``get_chat_model()``：从 settings 读取 OpenAI 兼容配置构造
``ChatOpenAI``；未配置时抛出 ``LLMNotConfiguredError``。工厂内不做任何网络请求。

为方便测试与依赖注入，提供 ``set_chat_model`` / ``reset_chat_model``：
设置覆盖后 ``get_chat_model()`` 直接返回覆盖模型，不读配置、不构造 ChatOpenAI。
"""

from __future__ import annotations

from contextvars import ContextVar

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from pydantic import SecretStr

from maitian_agent.config.settings import get_settings


class InvocationState:
    def __init__(self):
        self.model: str | None = None


invoked_model: ContextVar[InvocationState | None] = ContextVar("invoked_model", default=None)


class _TrackedChatModel(BaseChatModel):
    wrapped: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return self.wrapped._llm_type

    @property
    def _identifying_params(self) -> dict:
        return self.wrapped._identifying_params

    def _mark_invoked(self) -> None:
        identifier = (
            getattr(self.wrapped, "model", None)
            or getattr(self.wrapped, "model_name", None)
            or self.wrapped.get_name()
        )
        state = invoked_model.get()
        if state is not None:
            state.model = str(identifier)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._mark_invoked()
        return self.wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._mark_invoked()
        return await self.wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._mark_invoked()
        yield from self.wrapped._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._mark_invoked()
        async for chunk in self.wrapped._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk


def track_model_invocation(model: BaseChatModel) -> BaseChatModel:
    return _TrackedChatModel(wrapped=model)


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
