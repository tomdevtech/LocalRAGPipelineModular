import os
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config.settings import Settings

class VectorDBManager:
    """
    Manages the connection, indexing, and retrieval from the Chroma Vector Store.

    All operations are encapsulated within this class for reusability across the application.
    """
    def __init__(self, settings: Settings):
        """
        Initialize the VectorDBManager with provided configuration settings.

        Args:
            settings: The configuration object holding paths, model names, etc.
        """
        self.settings = settings
        self.db_location = self.settings.vector_db_path
        self.collection_name = self.settings.collection_name
        self.embeddings = OllamaEmbeddings(model=self.settings.embedding_model)
        self.vector_store: Chroma | None = None

        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes or loads the Chroma vector store.
        If the DB directory does not exist, it is created.
        """
        os.makedirs(self.db_location, exist_ok=True)

        # Attempt to load existing store first
        try:
            # We use persist_directory but don't load documents/metadata yet, just establish connection
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.db_location,
                embedding_function=self.embeddings
            )
            print(f"INFO: Vector Store successfully loaded from {self.db_location}")
        except Exception as e:
            # If loading fails (e.g., collection doesn't exist), we initialize it fresh.
            print(f"ERROR: Failed to load Vector Store from {self.db_location}. Initializing new store. Details: {e}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.db_location,
                embedding_function=self.embeddings
            )


    def index_documents(self, documents: List[Document], file_path: str) -> None:
        """
        Ingests documents from a CSV file into the vector store.

        Args:
            documents: A list of prepared Document objects.
            file_path: Path to the source CSV file for bulk indexing.
        """
        print(f"INFO: Starting indexing process from {file_path}...")

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"ERROR: Source file not found at {file_path}")
            return

        ids: List[str] = []
        prepared_documents: List[Document] = []

        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            prepared_documents.append(document)

        # Add documents to the vector store
        self.vector_store.add_documents(documents=prepared_documents, ids=ids)
        print(f"INFO: Successfully indexed {len(prepared_documents)} documents.")


    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieves the top k most relevant documents for a query.

        Args:
            query: The query string.
            k: Number of documents to retrieve. Defaults to the setting's value.

        Returns:
            A list of retrieved Document objects.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def is_empty(self) -> bool:
        """
        Check if the vector store collection is empty.

        Returns:
            True if the collection has no documents, False otherwise.
        """
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})
            results = retriever.invoke("test")
            return len(results) == 0
        except Exception:
            # If any error occurs during retrieval attempt, assume store is not empty or connection is flawed.
            return False

# Example usage is intentionally omitted here to focus on class structure
# The main application code (in pipeline.py) will handle the usage.