"""
Retrieval service for the RAG pipeline.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from rag_pipeline.config.settings import Settings


class Retriever:
    """Handles retrieval of relevant documents from a vector store."""

    def __init__(
        self,
        vector_store: Chroma | None = None,
        embedding_model: str | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: An existing Chroma vector store instance.
            embedding_model: The name of the embedding model to use.
            persist_directory: Directory where the vector store is persisted.
            collection_name: Name of the collection in the vector store.
        """
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            # Use settings if not provided
            settings = Settings()
            embedding_model = embedding_model or settings.embedding_model
            persist_directory = persist_directory or settings.vector_db_path
            collection_name = collection_name or settings.collection_name

            embeddings = OllamaEmbeddings(model=embedding_model)
            self.vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top k most relevant documents for a query.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            A list of retrieved Document objects.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def add_documents(self, documents: List[Document], ids: List[str] | None = None) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs for the documents.
        """
        self.vector_store.add_documents(documents=documents, ids=ids)

    def is_empty(self) -> bool:
        """
        Check if the vector store collection is empty.

        Returns:
            True if the collection has no documents, False otherwise.
        """
        try:
            # Access the underlying collection to check the count
            return self.vector_store._collection.count() == 0
        except Exception:
            # If there's an error, we assume it's not empty to be safe
            return False