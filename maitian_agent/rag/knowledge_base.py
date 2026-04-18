"""
知识库管理
三层知识库架构实现
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from .vectorstore import get_vector_store, BaseVectorStore, ChromaVectorStore
from .embeddings import get_embedding_model, BGEEmbeddings


class KnowledgeLevel(str, Enum):
    """知识库层级"""
    UNIVERSAL = "universal"  # 通用基础库（教材）
    SCHOOL = "school"        # 学校共享库（校本教案）
    TEACHER = "teacher"      # 教师专属库（个人风格）


class KnowledgeBase:
    """知识库管理器

    三层知识库架构：
    1. 通用基础库 - 教材内容、标准教案
    2. 学校共享库 - 校本教案、共享资源
    3. 教师专属库 - 个人风格画像、偏好设置
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_model_name: str = "BAAI/bge-large-zh"
    ):
        self.persist_directory = persist_directory
        self.embedding = get_embedding_model("bge", embedding_model_name)
        self.collections: Dict[KnowledgeLevel, BaseVectorStore] = {}

        self._init_collections()

    def _init_collections(self):
        """初始化知识库集合"""
        for level in KnowledgeLevel:
            self.collections[level] = get_vector_store(
                store_type="chroma",
                collection_name=f"knowledge_{level.value}",
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )

    def add_document(
        self,
        text: str,
        level: KnowledgeLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """添加文档到指定层级知识库

        Args:
            text: 文档内容
            level: 知识库层级
            metadata: 元数据

        Returns:
            文档ID列表
        """
        if level not in self.collections:
            raise ValueError(f"未知的知识库层级: {level}")

        doc_metadata = metadata or {}
        doc_metadata["level"] = level.value

        vectorstore = self.collections[level]
        return vectorstore.add_texts([text], metadatas=[doc_metadata])

    def add_documents(
        self,
        texts: List[str],
        level: KnowledgeLevel,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """批量添加文档

        Args:
            texts: 文档内容列表
            level: 知识库层级
            metadatas: 元数据列表

        Returns:
            文档ID列表
        """
        if level not in self.collections:
            raise ValueError(f"未知的知识库层级: {level}")

        docs_metadata = []
        for i, metadata in enumerate(metadatas or [{} for _ in texts]):
            doc_metadata = metadata.copy()
            doc_metadata["level"] = level.value
            docs_metadata.append(doc_metadata)

        vectorstore = self.collections[level]
        return vectorstore.add_texts(texts, metadatas=docs_metadata)

    def search(
        self,
        query: str,
        level: Optional[KnowledgeLevel] = None,
        k: int = 4
    ) -> List[Any]:
        """搜索知识库

        Args:
            query: 查询内容
            level: 指定层级（None表示所有层级）
            k: 返回数量

        Returns:
            搜索结果列表
        """
        if level:
            if level not in self.collections:
                raise ValueError(f"未知的知识库层级: {level}")
            return self.collections[level].similarity_search(query, k=k)

        results = []
        for level_vectorstore in self.collections.values():
            level_results = level_vectorstore.similarity_search(query, k=k)
            results.extend(level_results)
        return results

    def search_with_filter(
        self,
        query: str,
        filter_dict: Dict[str, Any],
        k: int = 4
    ) -> List[Any]:
        """带过滤条件的搜索

        Args:
            query: 查询内容
            filter_dict: 过滤条件
            k: 返回数量

        Returns:
            搜索结果列表
        """
        results = []
        for level_vectorstore in self.collections.values():
            try:
                level_results = level_vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
                results.extend(level_results)
            except Exception:
                continue
        return results

    def delete_by_level(self, level: KnowledgeLevel, ids: List[str]) -> None:
        """删除指定层级的文档

        Args:
            level: 知识库层级
            ids: 文档ID列表
        """
        if level not in self.collections:
            raise ValueError(f"未知的知识库层级: {level}")
        self.collections[level].delete(ids)

    def get_collection_info(self, level: KnowledgeLevel) -> Dict[str, Any]:
        """获取知识库集合信息

        Args:
            level: 知识库层级

        Returns:
            集合信息
        """
        if level not in self.collections:
            raise ValueError(f"未知的知识库层级: {level}")

        vectorstore = self.collections[level]
        return {
            "level": level.value,
            "name": f"knowledge_{level.value}",
            "persist_directory": self.persist_directory
        }


class HybridRetriever:
    """混合检索器

    结合向量检索和关键词检索
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        embedding_model: Optional[BGEEmbeddings] = None
    ):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model

    def vector_search(self, query: str, k: int = 4) -> List[Any]:
        """向量检索"""
        return self.vectorstore.similarity_search(query, k=k)

    def keyword_search(self, query: str, k: int = 4) -> List[Any]:
        """关键词检索（待实现BM25）

        Args:
            query: 查询内容
            k: 返回数量

        Returns:
            检索结果
        """
        # TODO: 实现BM25关键词检索
        raise NotImplementedError("BM25关键词检索待实现")

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Any]:
        """混合检索

        Args:
            query: 查询内容
            k: 返回数量
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重

        Returns:
            混合检索结果
        """
        # TODO: 实现混合检索融合
        # 目前先只返回向量检索结果
        return self.vector_search(query, k=k)
