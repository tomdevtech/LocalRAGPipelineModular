from __future__ import annotations

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config.settings import Settings

# Chroma rejects very large single inserts, so documents are added in batches.
_ADD_BATCH_SIZE = 500


class Retriever:
    """
    Handles retrieval of relevant documents from a vector store.

    The retriever is responsible for interacting with the persistent vector
    database, handling both document indexing (adding) and query retrieval.

    Design Note: The embedding function and vector store are created lazily on
    first use when no ``vector_store`` is injected, which keeps startup fast.
    Any argument that is not given falls back to the ``Settings`` defaults.
    """

    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Args:
            vector_store: An existing Chroma instance (for dependency injection).
            embedding_model: Name of the Ollama embedding model.
            persist_directory: Directory where the vector store is persisted.
            collection_name: Name of the collection in the vector store.
        """
        self.vector_store = vector_store
        self._embedding_model = embedding_model
        self._persist_directory = persist_directory
        self._collection_name = collection_name

    def _initialize_vector_store(self) -> Chroma:
        """Create the Chroma vector store and its embedding function."""
        defaults = Settings()
        embeddings = OllamaEmbeddings(
            model=self._embedding_model or defaults.embedding_model
        )
        return Chroma(
            collection_name=self._collection_name or defaults.collection_name,
            persist_directory=self._persist_directory or defaults.vector_db_path,
            embedding_function=embeddings,
        )

    def get_vector_store(self) -> Chroma:
        """Return the vector store, creating it lazily on first use."""
        if self.vector_store is None:
            self.vector_store = self._initialize_vector_store()
        return self.vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top k most relevant documents for a query.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            A list of retrieved Document objects.
        """
        retriever = self.get_vector_store().as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs (same length as ``documents``). Documents
                with an existing ID are updated instead of duplicated.
        """
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")

        store = self.get_vector_store()
        for start in range(0, len(documents), _ADD_BATCH_SIZE):
            end = start + _ADD_BATCH_SIZE
            store.add_documents(
                documents=documents[start:end],
                ids=ids[start:end] if ids is not None else None,
            )

    def is_empty(self) -> bool:
        """Return True if the collection contains no documents."""
        return len(self.get_vector_store().get(limit=1)["ids"]) == 0
