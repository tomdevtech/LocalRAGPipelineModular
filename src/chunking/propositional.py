"""Propositional chunking: one chunk per self-contained statement."""
from __future__ import annotations

from typing import Callable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker
from .utils import split_sentences


class PropositionalChunker(BaseChunker):
    """
    Propositional chunking: one chunk per self-contained statement.

    By default a *sentence* is used as the proposition (a heuristic baseline
    that works without any model). For true propositional chunking, pass an
    ``extractor``: a callable ``text -> list[str]`` that returns atomic,
    self-contained statements (e.g. backed by an LLM or a dependency parser).

    Statements longer than ``chunk_size`` are hard-split so that the size limit
    always holds. ``chunk_overlap`` only applies to that hard split.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extractor: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Overlap, only used when a statement must be hard-split.
            extractor: Optional callable ``text -> list[str]`` returning atomic
                statements. Without it, sentences are used.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extractor = extractor
        # Safety net for statements longer than chunk_size; the overlap is
        # clamped because the splitter rejects overlap >= chunk_size.
        self._hard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(chunk_overlap, max(chunk_size - 1, 0)),
        )

    def _extract_propositions(self, text: str) -> List[str]:
        """Return the propositions of ``text``. Override or inject ``extractor``."""
        # An injected extractor (e.g. an LLM) wins over the sentence heuristic.
        # Blank results are dropped so that no empty chunk reaches the vector store.
        if self.extractor is not None:
            return [p.strip() for p in self.extractor(text) if p and p.strip()]
        return split_sentences(text)

    def split_text(self, text: str) -> List[str]:
        """Split text into one chunk per proposition."""
        chunks: List[str] = []
        for proposition in self._extract_propositions(text):
            if len(proposition) > self.chunk_size:
                chunks.extend(self._hard_splitter.split_text(proposition))
            else:
                chunks.append(proposition)
        return chunks
