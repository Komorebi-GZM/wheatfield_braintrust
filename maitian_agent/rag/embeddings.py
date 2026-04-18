"""
嵌入模型管理
支持BGE和其他嵌入模型
"""
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
import os


class BGEEmbeddings(Embeddings):
    """BGE嵌入模型封装"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh",
        encode_kwargs: dict = None,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        passage_instruction: str = "这个段落的主题是："
    ):
        self.model_name = model_name
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self._embeddings = None

    @property
    def embeddings(self) -> HuggingFaceBgeEmbeddings:
        """获取嵌入模型实例"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceBgeEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs=self.encode_kwargs,
                query_instruction=self.query_instruction,
                passage_instruction=self.passage_instruction
            )
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        return self.embeddings.embed_documents(texts)

    def __call__(self, text: str) -> List[float]:
        """调用嵌入"""
        return self.embed_query(text)


def get_embedding_model(
    model_type: str = "bge",
    model_name: str = "BAAI/bge-large-zh",
    api_key: Optional[str] = None,
    **kwargs
) -> Embeddings:
    """获取嵌入模型

    Args:
        model_type: 模型类型 (bge/openai)
        model_name: 模型名称
        api_key: API密钥
        **kwargs: 其他参数

    Returns:
        嵌入模型实例
    """
    if model_type == "bge":
        return BGEEmbeddings(
            model_name=model_name,
            encode_kwargs=kwargs.get("encode_kwargs", {"normalize_embeddings": True}),
            query_instruction=kwargs.get("query_instruction", "为这个句子生成表示以用于检索相关文章："),
            passage_instruction=kwargs.get("passage_instruction", "这个段落的主题是：")
        )
    elif model_type == "openai":
        return OpenAIEmbeddings(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model_name
        )
    else:
        raise ValueError(f"不支持的嵌入模型类型: {model_type}")
