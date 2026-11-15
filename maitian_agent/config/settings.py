"""
系统配置管理

模型层保持厂商无关：只认 OpenAI 兼容接口，三项必填（LLM_BASE_URL / LLM_MODEL /
LLM_API_KEY）全部从环境变量读取，代码中不写任何厂商默认值。
"""
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 本地嵌入的默认模型（CPU 可跑）
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class Settings(BaseSettings):
    """系统配置（pydantic-settings v2）"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 项目信息
    project_name: str = Field(default="麦田智囊", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    debug: bool = Field(default=False, description="调试模式")

    # 模型层（厂商无关）——三项必填均无默认值，未配置时为 None
    llm_base_url: str | None = Field(default=None, description="LLM 基础 URL（OpenAI 兼容）")
    llm_model: str | None = Field(default=None, description="LLM 模型名称")
    llm_api_key: SecretStr | None = Field(default=None, description="LLM API 密钥")
    llm_temperature: float = Field(default=0.7, description="LLM 温度参数")
    llm_timeout_seconds: int = Field(default=60, description="LLM 请求超时（秒）")
    llm_max_retries: int = Field(default=2, description="LLM 最大重试次数")

    # 嵌入配置
    embedding_backend: str = Field(default="local", description="嵌入后端 local|openai_compatible")
    embedding_model: str | None = Field(
        default=None,
        description="嵌入模型名称；local 未配置时默认 bge-small-zh，openai_compatible 时回退 LLM_MODEL",
    )
    embedding_base_url: str | None = Field(
        default=None, description="嵌入基础 URL；openai_compatible 时未配置回退 LLM_BASE_URL"
    )
    embedding_api_key: SecretStr | None = Field(
        default=None, description="嵌入 API 密钥；openai_compatible 时未配置回退 LLM_API_KEY"
    )

    # RAG配置
    vector_store_type: str = Field(default="chroma", description="向量存储类型")
    chroma_persist_directory: str = Field(default="./data/chroma", description="Chroma持久化目录")

    # Agent配置
    max_iterations: int = Field(default=10, description="Agent最大迭代次数")
    max_execution_time: int = Field(default=60, description="Agent最大执行时间(秒)")
    temperature: float = Field(default=0.7, description="LLM温度参数（兼容旧字段，勿用于模型层）")

    # 记忆配置
    memory_type: str = Field(default="buffer", description="记忆类型")
    conversation_window: int = Field(default=5, description="对话窗口大小")
    memory_persist_directory: str = Field(default="./data/memory", description="记忆持久化目录")

    # OCR配置
    ocr_backend: str = Field(
        default="none", description="OCR 后端 none|easyocr|xiaomi（默认 none，不加载 torch/easyocr）"
    )
    ocr_type: str = Field(default="easyocr", description="OCR类型（兼容旧字段）")
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

    @property
    def llm_configured(self) -> bool:
        """三项必填（BASE_URL / MODEL / API_KEY）都非空才为 True。"""
        if not self.llm_base_url or not self.llm_model or self.llm_api_key is None:
            return False
        return bool(self.llm_api_key.get_secret_value())

    def get_llm_api_key(self) -> str | None:
        """取出明文密钥，供构造 ChatOpenAI 使用；不要写入日志或 API 响应。"""
        return self.llm_api_key.get_secret_value() if self.llm_api_key is not None else None

    @property
    def resolved_embedding_model(self) -> str | None:
        """解析后的嵌入模型名称。"""
        if self.embedding_backend == "local":
            return self.embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL
        # openai_compatible：未显式配置则回退到 LLM_MODEL
        return self.embedding_model or self.llm_model

    @property
    def resolved_embedding_base_url(self) -> str | None:
        """openai_compatible 时的嵌入基础 URL，未配置回退 LLM_BASE_URL。"""
        return self.embedding_base_url or self.llm_base_url

    def get_resolved_embedding_api_key(self) -> str | None:
        """openai_compatible 时的嵌入密钥明文，未配置回退 LLM_API_KEY。"""
        key = self.embedding_api_key or self.llm_api_key
        return key.get_secret_value() if key is not None else None


@lru_cache
def get_settings() -> Settings:
    """获取缓存的配置实例（测试中可 get_settings.cache_clear() 后重建）"""
    return Settings()


settings = get_settings()
