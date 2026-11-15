"""
嵌入模型管理（厂商无关，懒加载）

- ``local``：默认 ``BAAI/bge-small-zh-v1.5``，使用 sentence-transformers。
  sentence-transformers / HuggingFaceBgeEmbeddings 必须在函数内部懒加载，
  缺失时抛出清晰错误，提示安装可选 extra：``uv sync --extra rag`` 或 ``pip install 'maitian-agent[rag]'``。
- ``openai_compatible``：使用 OpenAI 兼容嵌入，凭证未配置时回退到 LLM_*。
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from maitian_agent.config.settings import get_settings


class BGEEmbeddings(Embeddings):
    """BGE 本地嵌入模型封装（HuggingFaceBgeEmbeddings 懒加载）。"""

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
                import sentence_transformers  # noqa: F401
                from langchain_community.embeddings import HuggingFaceBgeEmbeddings

                self._embeddings = HuggingFaceBgeEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs=self.encode_kwargs,
                    query_instruction=self.query_instruction,
                    passage_instruction=self.passage_instruction,
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
        from langchain_openai import OpenAIEmbeddings

        model_name: str = model or settings.resolved_embedding_model or ""
        api_key_str = settings.get_resolved_embedding_api_key()
        api_key_secret: SecretStr | None = SecretStr(api_key_str) if api_key_str else None
        return OpenAIEmbeddings(
            model=model_name,
            api_key=api_key_secret,
            base_url=settings.resolved_embedding_base_url,
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
