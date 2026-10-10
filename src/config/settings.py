"""
Configuration settings for the RAG pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class Settings:
    """Configuration settings for the RAG pipeline."""

    # Paths
    data_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    vector_db_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "..", "data", "chrome_langchain_db"))
    collection_name: str = "restaurant_reviews"

    # Models
    embedding_model: str = "mxbai-embed-large"
    llm_model: str = "llama3.2"

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    k: int = 5  # Number of documents to retrieve initially

    # Reranking
    use_reranking: bool = True
    reranker_type: str = "embedding"
    rerank_top_k: int = 3  # Number of documents to keep after reranking

    # Additional settings can be added here

    def __post_init__(self) -> None:
        """Ensure paths are absolute and exist if necessary."""
        self.data_path = os.path.abspath(self.data_path)
        self.vector_db_path = os.path.abspath(self.vector_db_path)

        # Create directories if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)

    @classmethod
    def from_yaml(cls, file_path: str) -> "Settings":
        """
        Load settings from a YAML file.

        Args:
            file_path: Path to the YAML file.

        Returns:
            A Settings instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If PyYAML is not installed.
        """
        if yaml is None:  # pragma: no cover
            raise ImportError("PyYAML is not installed. Install it with 'pip install pyyaml'.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}

        return cls(**config)

    def to_yaml(self, file_path: str) -> None:
        """
        Save settings to a YAML file.

        Args:
            file_path: Path to the YAML file.
        """
        if yaml is None:  # pragma: no cover
            raise ImportError("PyYAML is not installed. Install it with 'pip install pyyaml'.")

        # Convert dataclass to dict
        config: Dict[str, Any] = {
            "data_path": self.data_path,
            "vector_db_path": self.vector_db_path,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "k": self.k,
            "use_reranking": self.use_reranking,
            "reranker_type": self.reranker_type,
            "rerank_top_k": self.rerank_top_k,
        }

        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)