"""
嵌入模型管理（厂商无关，懒加载）

- ``local``：默认 ``BAAI/bge-small-zh-v1.5``，使用 sentence-transformers。
  sentence-transformers / HuggingFaceEmbeddings 必须在函数内部懒加载，
  缺失时抛出清晰错误，提示安装可选 extra：``uv sync --extra rag`` 或
  ``pip install 'maitian-agent[rag]'``。
- ``openai_compatible``：使用 OpenAI 兼容嵌入，凭证未配置时回退到 LLM_*。
"""

from __future__ import annotations

from typing import cast

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from maitian_agent.config.settings import get_settings


class EmbeddingNotConfiguredError(RuntimeError):
    """嵌入凭证未配置时抛出；构造前就报错，绝不把 None 传给 SDK。"""

    code = "EMBEDDING_NOT_CONFIGURED"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or (
            "未配置嵌入模型，请设置 EMBEDDING_BASE_URL、EMBEDDING_MODEL、EMBEDDING_API_KEY"
            "（openai_compatible 时也可回退使用 LLM_BASE_URL、LLM_MODEL、LLM_API_KEY）"
        )
        super().__init__(self.message)


class BGEEmbeddings(Embeddings):
    """BGE 本地嵌入模型封装（HuggingFaceEmbeddings 懒加载）。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        encode_kwargs: dict | None = None,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        passage_instruction: str = "这个段落的主题是：",
    ):
        self.model_name = model_name
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self._embeddings = None

    @property
    def embeddings(self):
        """获取嵌入模型实例（首次访问才 import 重依赖）。"""
        if self._embeddings is None:
            try:
                import sentence_transformers  # type: ignore[import-not-found]  # noqa: F401
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={**self.encode_kwargs, "prompt": self.passage_instruction},
                    query_encode_kwargs={
                        **self.encode_kwargs,
                        "prompt": self.query_instruction,
                    },
                )
            except ImportError as exc:  # 未安装可选 extra
                raise ImportError(
                    "本地嵌入需要可选依赖 sentence-transformers，"
                    "请运行: uv sync --extra rag 或 pip install 'maitian-agent[rag]'"
                ) from exc
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询"""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入多个文档"""
        return self.embeddings.embed_documents(texts)

    def __call__(self, text: str) -> list[float]:
        """调用嵌入"""
        return self.embed_query(text)


def get_embeddings(
    backend: str | None = None,
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """按配置返回嵌入模型。

    Args:
        backend: local|openai_compatible，默认取 settings.embedding_backend
        model: 覆盖嵌入模型名称，默认按配置解析
        **kwargs: 透传参数

    Returns:
        Embeddings 实例
    """
    settings = get_settings()
    backend = backend or settings.embedding_backend

    if backend == "local":
        model_name = model or settings.resolved_embedding_model or "BAAI/bge-small-zh-v1.5"
        return BGEEmbeddings(
            model_name=model_name,
            encode_kwargs=kwargs.get("encode_kwargs", {"normalize_embeddings": True}),
            query_instruction=kwargs.get(
                "query_instruction", "为这个句子生成表示以用于检索相关文章："
            ),
            passage_instruction=kwargs.get("passage_instruction", "这个段落的主题是："),
        )

    if backend == "openai_compatible":
        # 懒加载 OpenAI 兼容嵌入，未走到该分支时不 import
        model_name = model or settings.resolved_embedding_model
        base_url = settings.resolved_embedding_base_url
        api_key_str = settings.get_resolved_embedding_api_key()

        # base_url / model / key 任一缺失都拒绝构造，绝不把 None 传给 SDK
        # （否则 SDK 会回退到官方默认端点）
        missing = [
            name
            for name, value in (
                ("EMBEDDING_BASE_URL/LLM_BASE_URL", base_url),
                ("EMBEDDING_MODEL/LLM_MODEL", model_name),
                ("EMBEDDING_API_KEY/LLM_API_KEY", api_key_str),
            )
            if not value
        ]
        if missing:
            raise EmbeddingNotConfiguredError(
                "openai_compatible 嵌入配置不完整，缺少："
                + "、".join(missing)
                + "。请设置 EMBEDDING_* 或回退使用的 LLM_*。"
            )

        from langchain_openai import OpenAIEmbeddings

        # 经上面的缺失检查后，三者均非空；用 cast 明确类型，显式传入
        # 以避免 SDK 从环境读取 OPENAI_*
        return OpenAIEmbeddings(
            model=cast(str, model_name),
            api_key=SecretStr(cast(str, api_key_str)),
            base_url=cast(str, base_url),
            **kwargs,
        )

    raise ValueError(f"不支持的嵌入后端: {backend}")


def get_embedding_model(
    model_type: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> Embeddings:
    """兼容旧接口：把旧的 model_type 映射到新后端。

    - ``bge`` / ``local`` -> local
    - ``openai`` / ``openai_compatible`` -> openai_compatible
    - None -> 使用 settings.embedding_backend
    """
    if model_type in ("bge", "local"):
        backend = "local"
    elif model_type in ("openai", "openai_compatible"):
        backend = "openai_compatible"
    else:
        backend = None
    return get_embeddings(backend=backend, model=model_name, **kwargs)
