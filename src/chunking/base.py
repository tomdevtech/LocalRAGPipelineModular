"""Abstract base class for all chunking strategies.

Lives in its own module so that every concrete chunker can import it without
going through ``strategies.py`` (which imports the concrete chunkers).
That is what avoids the circular import.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


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
