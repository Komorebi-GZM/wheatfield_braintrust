"""
向量存储管理
支持Chroma和Milvus
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingDimensionMismatchError(RuntimeError):
    """已存向量维度与当前 embedding 维度不一致时抛出。"""

    code = "EMBEDDING_DIMENSION_MISMATCH"

    def __init__(self, stored_dimension: int, current_dimension: int) -> None:
        self.stored_dimension = stored_dimension
        self.current_dimension = current_dimension
        self.message = (
            f"知识库向量维度不一致：已存维度为 {stored_dimension}，"
            f"当前 embedding 维度为 {current_dimension}。"
            "已有索引目录（例如 ./data/chroma）不能与当前嵌入模型混用。"
            "请把 EMBEDDING_MODEL 设回原来的模型（例如 BAAI/bge-large-zh），"
            "或备份/删除索引目录后用当前模型重建。"
        )
        super().__init__(self.message)


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> list[Any]:
        """相似度搜索"""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """删除向量"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Chroma向量存储"""

    # 写入 collection metadata 的维度字段名
    DIMENSION_KEY = "embedding_dimension"

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./data/chroma",
        embedding_function: Embeddings | None = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._vectorstore = None

    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例（打开时校验向量维度）"""
        if self._vectorstore is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            self._validate_dimension(self._vectorstore)
        return self._vectorstore

    def _current_embedding_dimension(self) -> int | None:
        """用当前 embedding 嵌入一小段文本得到维度（fake embedding 同样适用）。"""
        if self.embedding_function is None:
            return None
        probe = self.embedding_function.embed_query("维度校验")
        return len(probe)

    def _extract_dimension(self, result: Any) -> int | None:
        """从 Chroma get/peek 结果中取出首条向量的维度。"""
        if not result:
            return None
        embeddings = result.get("embeddings")
        if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
            return len(embeddings[0])
        return None

    def _probe_collection_dimension(self, collection: Any) -> int | None:
        """从已存向量中探测维度（用于没有 metadata 的遗留 collection）。"""
        for getter in (
            lambda: collection.get(limit=1, include=["embeddings"]),
            lambda: collection.peek(limit=1),
        ):
            try:
                result = getter()
            except Exception:  # noqa: BLE001 - 探测失败时回退到下一种方式
                logger.debug("探测已存向量维度失败", exc_info=True)
                continue
            dimension = self._extract_dimension(result)
            if dimension is not None:
                return dimension
        return None

    def _validate_dimension(self, store: Chroma) -> None:
        """校验已存 collection 维度与当前 embedding 维度是否一致；不一致则报错。

        - 新建/空 collection：把当前维度写入 metadata。
        - 有 metadata：直接比较 metadata 维度。
        - 无 metadata 的遗留 collection：probe 已有向量维度后再比较。
        """
        collection = store._collection
        current_dim = self._current_embedding_dimension()
        if current_dim is None:
            return

        metadata = dict(collection.metadata or {})
        stored_dim = metadata.get(self.DIMENSION_KEY)
        try:
            stored_dim = int(stored_dim) if stored_dim is not None else None
        except (TypeError, ValueError):
            stored_dim = None

        if stored_dim is None:
            stored_dim = self._probe_collection_dimension(collection)

        if stored_dim is None:
            # 新建/空 collection：写入维度
            metadata[self.DIMENSION_KEY] = current_dim
            collection.modify(metadata=metadata)
            return

        if stored_dim != current_dim:
            raise EmbeddingDimensionMismatchError(stored_dim, current_dim)

        if self.DIMENSION_KEY not in metadata:
            # 遗留 collection 维度匹配：补写 metadata，方便下次直接比较
            metadata[self.DIMENSION_KEY] = current_dim
            collection.modify(metadata=metadata)

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本"""
        return self.vectorstore.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[Any]:
        """相似度搜索"""
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None
    ) -> list[tuple]:
        """带分数的相似度搜索"""
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: list[str]) -> None:
        """删除向量"""
        self.vectorstore.delete(ids)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        collection_name: str = "default",
        persist_directory: str = "./data/chroma",
        **kwargs
    ) -> "ChromaVectorStore":
        """从文本创建向量存储（打开时先校验/写入维度）"""
        instance = cls(collection_name, persist_directory, embedding)
        instance.add_texts(texts, metadatas=metadatas)
        return instance


class MilvusVectorStore(BaseVectorStore):
    """Milvus向量存储（待配置）"""

    def __init__(
        self,
        connection_args: dict[str, Any] | None = None,
        collection_name: str = "default",
        embedding_function: Embeddings | None = None
    ):
        self.connection_args = connection_args or {
            "host": "localhost",
            "port": "19530"
        }
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._vectorstore = None

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本"""
        if self._vectorstore is None:
            raise NotImplementedError("Milvus连接未配置")
        return self._vectorstore.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4) -> list[Any]:
        """相似度搜索"""
        if self._vectorstore is None:
            raise NotImplementedError("Milvus连接未配置")
        return self._vectorstore.similarity_search(query, k=k)

    def delete(self, ids: list[str]) -> None:
        """删除向量"""
        if self._vectorstore is None:
            raise NotImplementedError("Milvus连接未配置")
        self._vectorstore.delete(ids)


def get_vector_store(
    store_type: str = "chroma",
    collection_name: str = "default",
    embedding_function: Embeddings | None = None,
    **kwargs
) -> BaseVectorStore:
    """获取向量存储实例

    Args:
        store_type: 存储类型 (chroma/milvus)
        collection_name: 集合名称
        embedding_function: 嵌入函数
        **kwargs: 其他参数

    Returns:
        向量存储实例
    """
    if store_type == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=kwargs.get("persist_directory", "./data/chroma"),
            embedding_function=embedding_function
        )
    elif store_type == "milvus":
        return MilvusVectorStore(
            connection_args=kwargs.get("connection_args"),
            collection_name=collection_name,
            embedding_function=embedding_function
        )
    else:
        raise ValueError(f"不支持的向量存储类型: {store_type}")
