"""
Reranking service for the RAG pipeline.
"""
from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config.settings import Settings


class BaseReranker:
    """Abstract base class for rerankers."""

    def rerank(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The query string.
            documents: List of documents to rerank.
            top_k: Number of top documents to return after reranking.

        Returns:
            A list of reranked documents (length up to top_k).
        """
        raise NotImplementedError


class EmbeddingReranker(BaseReranker):
    """Reranker that uses embedding similarity to re-rank documents."""

    def __init__(self, embedding_model: str | None = None):
        """
        Args:
            embedding_model: The name of the embedding model to use for similarity.
                Defaults to the embedding model from ``Settings``.
        """
        self.embedding_model = embedding_model or Settings.embedding_model
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

    def rerank(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        """
        Rerank documents by cosine similarity between query and document embeddings.

        Args:
            query: The query string.
            documents: List of documents to rerank.
            top_k: Number of top documents to return after reranking.

        Returns:
            A list of reranked documents (length up to top_k).
        """
        if not documents:
            return []

        query_embedding = np.asarray(self.embeddings.embed_query(query), dtype=float)
        doc_embeddings = np.asarray(
            self.embeddings.embed_documents([doc.page_content for doc in documents]),
            dtype=float,
        )

        # Cosine similarity; the epsilon guards against zero vectors.
        denominator = np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(doc_embeddings, query_embedding) / np.maximum(denominator, 1e-12)

        # stable sort -> documents with equal scores keep their retrieval order
        top_indices = np.argsort(-similarities, kind="stable")[:top_k]
        return [documents[i] for i in top_indices]


def get_reranker(reranker_type: str = "embedding", **kwargs) -> BaseReranker:
    """
    Factory function to get a reranker instance.

    Args:
        reranker_type: Type of reranker to create (currently only "embedding" is supported).
        **kwargs: Arguments to pass to the reranker constructor.

    Returns:
        An instance of a BaseReranker subclass.

    Raises:
        ValueError: If the reranker_type is not recognized.
    """
    if reranker_type == "embedding":
        return EmbeddingReranker(**kwargs)
    raise ValueError(
        f"Unknown reranker type: {reranker_type}. Available types: ['embedding']"
    )
