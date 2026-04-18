"""
向量存储管理
支持Chroma和Milvus
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_community.vectorstores import Milvus
import os


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """添加文本"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        """相似度搜索"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Chroma向量存储"""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./data/chroma",
        embedding_function: Optional[Embeddings] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._vectorstore = None

    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例"""
        if self._vectorstore is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        return self._vectorstore

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """添加文本"""
        return self.vectorstore.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None) -> List[Any]:
        """相似度搜索"""
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """带分数的相似度搜索"""
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        self.vectorstore.delete(ids)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict]] = None,
        collection_name: str = "default",
        persist_directory: str = "./data/chroma",
        **kwargs
    ) -> "ChromaVectorStore":
        """从文本创建向量存储"""
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        vectorstore.add_texts(texts, metadatas=metadatas)
        instance = cls(collection_name, persist_directory, embedding)
        instance._vectorstore = vectorstore
        return instance


class MilvusVectorStore(BaseVectorStore):
    """Milvus向量存储（待配置）"""

    def __init__(
        self,
        connection_args: Dict[str, Any] = None,
        collection_name: str = "default",
        embedding_function: Optional[Embeddings] = None
    ):
        self.connection_args = connection_args or {
            "host": "localhost",
            "port": "19530"
        }
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._vectorstore = None

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """添加文本"""
        if self._vectorstore is None:
            raise NotImplementedError("Milvus连接未配置")
        return self._vectorstore.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        """相似度搜索"""
        if self._vectorstore is None:
            raise NotImplementedError("Milvus连接未配置")
        return self._vectorstore.similarity_search(query, k=k)

    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        if self._vectorstore is None:
            raise NotImplementedError("Milvus连接未配置")
        self._vectorstore.delete(ids)


def get_vector_store(
    store_type: str = "chroma",
    collection_name: str = "default",
    embedding_function: Optional[Embeddings] = None,
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
