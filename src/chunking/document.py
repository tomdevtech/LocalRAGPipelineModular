"""Document chunking: splits text along paragraph boundaries."""
from __future__ import annotations

from .base import BaseChunker
from langchain_text_splitters import CharacterTextSplitter
from typing import List

class DocumentChunker(BaseChunker):
    """Document-based chunking (e.g., by paragraphs)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Maximum number of characters per chunk. Neighbouring
                paragraphs are merged as long as they fit into it.
            chunk_overlap: Number of characters shared by consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Split by double newline (paragraphs) then further split if needed
        self._splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n",
        )

    def split_text(self, text: str) -> List[str]:
        """Split text based on paragraph separators."""
        return self._splitter.split_text(text)