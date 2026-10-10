"""
Configuration settings for the RAG pipeline.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

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
    k: int = 5  # Number of documents returned when reranking is disabled

    # Reranking
    use_reranking: bool = True
    reranker_type: str = "embedding"
    rerank_top_k: int = 3  # Number of documents returned when reranking is enabled

    # Additional settings can be added here

    def __post_init__(self) -> None:
        """Validate values, make paths absolute and create directories."""
        # Fail early with a clear message; otherwise the splitters raise cryptic
        # errors much later, e.g. when the first document is indexed.
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.k <= 0:
            raise ValueError("k must be greater than 0")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0")

        # Absolute paths keep the vector store location independent of the
        # current working directory (e.g. `python src/main.py` vs. `local-rag`).
        self.data_path = os.path.abspath(self.data_path)
        self.vector_db_path = os.path.abspath(self.vector_db_path)

        # Create directories if they don't exist (Chroma itself creates the DB folder).
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

        with open(file_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}

        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a mapping: {file_path}")

        return cls(**config)

    def to_yaml(self, file_path: str) -> None:
        """
        Save settings to a YAML file.

        Args:
            file_path: Path to the YAML file.
        """
        if yaml is None:  # pragma: no cover
            raise ImportError("PyYAML is not installed. Install it with 'pip install pyyaml'.")

        config: Dict[str, Any] = asdict(self)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)