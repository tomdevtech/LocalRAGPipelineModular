from __future__ import annotations

from .base import BaseChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

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
        """Split text using a recursive splitting approach."""
        return self._splitter.split_text(text)