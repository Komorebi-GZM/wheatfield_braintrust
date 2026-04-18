"""
系统配置管理
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """系统配置"""

    # 项目信息
    project_name: str = Field(default="麦田智囊", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    debug: bool = Field(default=False, description="调试模式")

    # API配置
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    openai_api_base: str = Field(default="https://api.deepseek.com", description="API基础URL")
    model_name: str = Field(default="deepseek-v2", description="模型名称")

    # RAG配置
    vector_store_type: str = Field(default="chroma", description="向量存储类型")
    embedding_model: str = Field(default="BAAI/bge-large-zh", description="嵌入模型")
    chroma_persist_directory: str = Field(default="./data/chroma", description="Chroma持久化目录")

    # Agent配置
    max_iterations: int = Field(default=10, description="Agent最大迭代次数")
    max_execution_time: int = Field(default=60, description="Agent最大执行时间(秒)")
    temperature: float = Field(default=0.7, description="LLM温度参数")

    # 记忆配置
    memory_type: str = Field(default="buffer", description="记忆类型")
    conversation_window: int = Field(default=5, description="对话窗口大小")
    memory_persist_directory: str = Field(default="./data/memory", description="记忆持久化目录")

    # OCR配置
    ocr_type: str = Field(default="easyocr", description="OCR类型")
    ocr_languages: list = Field(default=["ch_sim", "en"], description="OCR语言")

    # Whisper配置
    whisper_model: str = Field(default="base", description="Whisper模型大小")

    # 路径配置
    base_dir: str = Field(default="./data", description="基础数据目录")
    upload_dir: str = Field(default="./data/uploads", description="上传文件目录")
    assets_dir: str = Field(default="./data/assets", description="静态资源目录")
    logs_dir: str = Field(default="./logs", description="日志目录")

    # 前端配置
    streamlit_server_port: int = Field(default=8501, description="Streamlit服务端口")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """获取缓存的配置实例"""
    return Settings()


settings = get_settings()
