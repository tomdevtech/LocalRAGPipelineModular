from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings

from .base import BaseChunker
from .utils import split_sentences


class SemanticChunker(BaseChunker):
    """
    Semantic chunking: starts a new chunk where the topic changes.

    Algorithm:
    1. Split the text into sentences.
    2. Embed every sentence.
    3. Walk through the sentences in order. A new chunk is started when the
       cosine similarity to the previous sentence drops below
       ``similarity_threshold`` or when the chunk would exceed ``chunk_size``.

    Sentences longer than ``chunk_size`` are hard-split with a recursive
    splitter. ``chunk_overlap`` is accepted for API compatibility with the other
    chunkers but is not used, because semantic boundaries are the whole point.

    Args:
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Unused (see above).
        embeddings: Any object with ``embed_documents(list[str])``. If omitted,
            ``OllamaEmbeddings`` is created lazily on first use.
        embedding_model: Ollama model name used when ``embeddings`` is omitted.
        similarity_threshold: Cosine similarity below which a new chunk starts.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: Optional[Any] = None,
        embedding_model: Optional[str] = None,
        similarity_threshold: float = 0.6,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self._embedding_model = embedding_model or Settings.embedding_model
        self._embeddings = embeddings
        self._hard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(chunk_overlap, max(chunk_size - 1, 0)),
        )

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(model=self._embedding_model)
        return self._embeddings

    def _similarities(self, sentences: List[str]) -> np.ndarray:
        """Cosine similarity between each sentence and the one before it."""
        vectors = np.asarray(self.embeddings.embed_documents(sentences), dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1e-12
        unit = vectors / norms[:, None]
        return np.sum(unit[1:] * unit[:-1], axis=1)

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        sentences: List[str] = []
        for sentence in split_sentences(text):
            if len(sentence) > self.chunk_size:
                sentences.extend(self._hard_splitter.split_text(sentence))
            else:
                sentences.append(sentence)

        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        similarities = self._similarities(sentences)

        chunks: List[str] = []
        current = sentences[0]
        for sentence, similarity in zip(sentences[1:], similarities):
            too_long = len(current) + 1 + len(sentence) > self.chunk_size
            if similarity < self.similarity_threshold or too_long:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}"
        chunks.append(current)
        return chunks
