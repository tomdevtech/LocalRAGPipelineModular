"""
Chunking strategies for the RAG pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks.
        """
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with optional overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="",
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class RecursiveChunker(BaseChunker):
    """Recursive chunking that tries to split by different separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class DocumentChunker(BaseChunker):
    """Document-based chunking (e.g., by paragraphs)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Split by double newline (paragraphs) then further split if needed
        self._splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n",
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class SemanticChunker(BaseChunker):
    """Semantic chunking (placeholder: splits by sentences and groups)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # We'll use a simple sentence splitter for now
        # In a real implementation, we would use embeddings to group semantically similar sentences.
        self._splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=". ",
        )

    def split_text(self, text: str) -> List[str]:
        # This is a placeholder: real semantic chunking would be more complex.
        return self._splitter.split_text(text)


class PropositionalChunker(BaseChunker):
    """Propositional chunking (placeholder: similar to semantic for now)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=". ",
        )

    def split_text(self, text: str) -> List[str]:
        # Placeholder: real propositional chunking would extract propositions.
        return self._splitter.split_text(text)


class AgenticChunker(BaseChunker):
    """Agentic chunking (placeholder: returns whole text as one chunk)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Note: chunk_size and chunk_overlap are ignored in this placeholder.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # In a real agentic chunker, an agent would decide the splits.
        # For now, we return the whole text as a single chunk.
        return [text] if text.strip() else []


def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    """Factory function to get a chunker instance by strategy name.

    Args:
        strategy: Name of the chunking strategy.
        **kwargs: Arguments to pass to the chunker constructor.

    Returns:
        An instance of a BaseChunker subclass.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    strategies = {
        "fixed_size": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "document": DocumentChunker,
        "semantic": SemanticChunker,
        "propositional": PropositionalChunker,
        "agentic": AgenticChunker,
    }
    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available strategies: {list(strategies.keys())}"
        )
    return strategies[strategy](**kwargs)