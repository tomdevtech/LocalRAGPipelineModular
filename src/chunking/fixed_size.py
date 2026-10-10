from .strategies import BaseChunker
from langchain_text_splitters import CharacterTextSplitter

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
        """Split text into fixed-size chunks."""
        return self._splitter.split_text(text)