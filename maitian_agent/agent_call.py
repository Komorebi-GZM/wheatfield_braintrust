"""前端调用 Agent 的安全封装：记录异常，不把异常文本展示给用户。"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

GENERIC_AGENT_ERROR = "服务处理失败，请稍后重试"

T = TypeVar("T")


def invoke_agent(
    fn: Callable[..., T],
    *args,
    error_handler: Callable[[str], object],
    log: logging.Logger = logger,
    **kwargs,
) -> T | None:
    """调用 Agent；失败时记日志并通过 error_handler 展示通用中文错误。"""
    try:
        return fn(*args, **kwargs)
    except Exception:
        log.exception("前端调用 Agent 失败")
        error_handler(GENERIC_AGENT_ERROR)
        return None
