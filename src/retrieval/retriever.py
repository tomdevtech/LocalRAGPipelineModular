from config.settings import Settings
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

class Retriever:
    """
    Handles retrieval of relevant documents from a vector store.

    The retriever is responsible for interacting with the persistent vector database,
    handling both document indexing (adding) and query retrieval.

    Design Note: Initialization of the embedding function and vector store
    is lazy when `vector_store` is not pre-injected, ensuring fast startup time.
    """

    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the retriever.

        If `vector_store` is provided, it uses the existing instance.
        If `vector_store` is None, the retriever initializes the Chroma
        vector store lazily using settings, which requires network/disk I/O.

        Args:
            vector_store: An existing Chroma vector store instance (for dependency injection).
            embedding_model: The name of the embedding model to use.
            persist_directory: Directory where the vector store is persisted.
            collection_name: Name of the collection in the vector store.
        """
        self.vector_store = vector_store
        self._embedding_model = embedding_model
        self._persist_directory = persist_directory
        self._collection_name = collection_name

    def _initialize_vector_store(self, settings: Settings) -> Chroma:
        """Initializes the Chroma vector store and embedding function."""
        embeddings = OllamaEmbeddings(model=settings.embedding_model)
        return Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.vector_db_path,
            embedding_function=embeddings,
        )

    def get_vector_store(self, settings: Settings) -> Chroma:
        """
        Returns the initialized Chroma vector store.
        Initializes it lazily if it hasn't been set up.
        """
        if self.vector_store is None:
            print("INFO: Initializing Chroma vector store lazily...")
            self.vector_store = self._initialize_vector_store(settings)
        return self.vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top k most relevant documents for a query.

        Args:
            query: The query string.
            k: Number of documents to retrieve. Defaults to pipeline settings.

        Returns:
            A list of retrieved Document objects. If the vector store is not
            initialized, it raises an exception.
        """
        settings = Settings()
        if self.vector_store is None:
            self.vector_store = self.get_vector_store(settings)

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs for the documents.
        """
        settings = Settings()
        if self.vector_store is None:
            self.vector_store = self.get_vector_store(settings)
        self.vector_store.add_documents(documents=documents, ids=ids)

    def is_empty(self) -> bool:
        """
        Check if the vector store collection is empty.

        Returns:
            True if the collection has no documents, False otherwise.
        """
        settings = Settings()
        if self.vector_store is None:
            self.vector_store = self.get_vector_store(settings)

        try:
            # Access the underlying collection to check the count
            return self.vector_store._collection.count() == 0
        except Exception:
            # If there's an error (e.g., not connected), we assume it's not empty to be safe.
            return False