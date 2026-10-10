"""
Main RAG pipeline that combines chunking, retrieval, and reranking.
"""
from __future__ import annotations

import hashlib
from typing import List

from langchain_core.documents import Document

from chunking.strategies import BaseChunker, get_chunker
from config.settings import Settings
from reranking.reranker import BaseReranker, get_reranker
from retrieval.retriever import Retriever


def _chunk_id(doc: Document, parent_index: int, chunk_index: int, chunk: str) -> str:
    """
    Deterministic ID for a chunk.

    Re-indexing the same documents yields the same IDs (so the vector store
    updates instead of duplicating), while different batches never collide.
    """
    key = f"{doc.metadata.get('source', '')}|{parent_index}|{chunk_index}|{chunk}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline.

    This pipeline handles:
    1. Chunking of input documents.
    2. Storage and retrieval of document chunks from a vector store.
    3. Reranking of retrieved chunks for improved relevance.
    4. Provides the context for an external language model generation step.
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the RAG pipeline.

        Args:
            settings: Configuration settings for the pipeline.
                      If None, default settings are used.
        """
        self.settings = settings or Settings()

        self.chunker: BaseChunker = get_chunker(
            self.settings.chunking_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        self.retriever = Retriever(
            embedding_model=self.settings.embedding_model,
            persist_directory=self.settings.vector_db_path,
            collection_name=self.settings.collection_name,
        )

        self.reranker: BaseReranker | None = None
        if self.settings.use_reranking:
            self.reranker = get_reranker(
                self.settings.reranker_type,
                embedding_model=self.settings.embedding_model,
            )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Chunk the documents and add the chunks to the vector store.

        Args:
            documents: List of Document objects to add.

        Returns:
            The number of chunks that were added.
        """
        chunked_documents: List[Document] = []
        ids: List[str] = []
        for i, doc in enumerate(documents):
            for j, chunk in enumerate(self.chunker.split_text(doc.page_content)):
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk_id": j, "parent_id": i},
                    )
                )
                ids.append(_chunk_id(doc, i, j, chunk))

        if chunked_documents:
            self.retriever.add_documents(chunked_documents, ids)
        return len(chunked_documents)

    def retrieve(self, query: str, k: int | None = None) -> List[Document]:
        """
        Retrieve relevant documents for a query, with optional reranking.

        Args:
            query: The query string.
            k: Number of documents to return. Defaults to ``rerank_top_k`` when
               reranking is enabled, otherwise to ``k`` from the settings.

        Returns:
            At most ``k`` documents.
            - With reranking: ``2 * k`` candidates are retrieved and the best
              ``k`` of them are returned, ordered by relevance.
            - Without reranking: the ``k`` best vector-search hits.
        """
        use_reranker = self.reranker is not None
        if k is None:
            k = self.settings.rerank_top_k if use_reranker else self.settings.k
        if k <= 0:
            raise ValueError("k must be greater than 0")

        # A wider candidate pool gives the reranker something to choose from.
        pool_size = 2 * k if use_reranker else k
        initial_docs = self.retriever.retrieve(query, k=pool_size)

        if use_reranker and initial_docs:
            return self.reranker.rerank(query, initial_docs, top_k=k)
        return initial_docs[:k]

    def get_context(self, query: str, k: int | None = None) -> str:
        """
        Get the context string for a query by retrieving and joining relevant documents.

        Args:
            query: The query string.
            k: Optional number of documents (see ``retrieve``).

        Returns:
            The concatenated page content of the retrieved documents.
        """
        docs = self.retrieve(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
