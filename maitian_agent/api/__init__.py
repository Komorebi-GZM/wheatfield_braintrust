"""
API模块
FastAPI封装LangChain核心逻辑
"""

from maitian_agent.api.routes import create_app

__all__ = ["create_app"]
