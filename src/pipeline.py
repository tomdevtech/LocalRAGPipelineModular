"""
Main RAG pipeline that combines chunking, retrieval, and reranking.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from config.settings import Settings
from chunking.strategies import BaseChunker, get_chunker
from retrieval.retriever import Retriever
from reranking.reranker import BaseReranker, get_reranker


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

        # Initialize chunker
        self.chunker: BaseChunker = get_chunker(
            self.settings.chunking_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Initialize retriever
        self.retriever = Retriever(
            embedding_model=self.settings.embedding_model,
            persist_directory=self.settings.vector_db_path,
            collection_name=self.settings.collection_name,
        )

        # Initialize reranker if enabled
        self.reranker: BaseReranker | None = None
        if self.settings.use_reranking:
            self.reranker = get_reranker(
                self.settings.reranker_type,
            )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the pipeline.

        This method chunks the documents and adds the chunks to the vector store.

        Args:
            documents: List of Document objects to add.
        """
        # Chunk the documents
        chunked_documents: List[Document] = []
        ids: List[str] = []
        for i, doc in enumerate(documents):
            chunks = self.chunker.split_text(doc.page_content)
            for j, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_id": j, "parent_id": i},
                )
                chunked_documents.append(chunked_doc)
                ids.append(f"{i}-{j}")

        # Add the chunked documents to the vector store
        self.retriever.add_documents(chunked_documents, ids)

    def retrieve(
        self, query: str, k: int | None = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query, with optional reranking.

        Args:
            query: The query string.
            k: Target number of final documents. If None, uses the pipeline setting.

        Returns:
            A list of retrieved Document objects.
            - If reranking is enabled: Returns top-k documents after reranking.
            - If reranking is disabled: Returns the initial retrieval result, which may contain up to 2*k documents (or more, depending on retriever implementation).
        """
        # Retrieve initial set of documents. We fetch a pool of documents
        # to give the reranker a wider selection. We cap the pool size to k_target + 3.
        # This avoids arbitrary multipliers while still providing buffer for reranking.
        pool_size = k_target + 3 if self.reranker is not None else k_target
        initial_docs = self.retriever.retrieve(query, k=pool_size)

        # If reranking is enabled, rerank the documents
        if self.reranker is not None and len(initial_docs) > 0:
            reranked_docs = self.reranker.rerank(
                query, initial_docs, top_k=k_target
            )
            return reranked_docs
        else:
            # If no reranking, simply return the first k documents from the initial retrieval
            return initial_docs[:k_target]

    def get_context(self, query: str) -> str:
        """
        Get the context string for a query by retrieving and joining relevant documents.

        Args:
            query: The query string.

        Returns:
            A string containing the concatenated page content of the retrieved documents.
        """
        docs = self.retrieve(query)
        return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # Example usage: Initialize settings, create pipeline, and perform a retrieval.
    # Note: For a real CLI, command line argument parsing should replace this dummy setup.
    settings = Settings()
    pipeline = RAGPipeline(settings=settings)

    # Example: Add some documents (assuming some documents are available/mocked)
    dummy_docs = [
        Document(page_content="This is a dummy document content.", metadata={"source": "mock"})
    ]
    pipeline.add_documents(dummy_docs)

    # Example: Perform retrieval
    query = "What is the main topic of the documents?"
    context = pipeline.get_context(query)
    print(f"\n--- Context Retrieved ---\n{context}")