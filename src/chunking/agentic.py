"""Agentic chunking: lets an external LLM agent decide where to split."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker

logger = logging.getLogger(__name__)

# An agent takes (text, chunk_size) and returns the list of chunks.
Agent = Callable[[str, int], List[str]]


class AgenticChunker(BaseChunker):
    """
    Agentic chunking: delegates the splitting decision to an LLM agent.

    Plug in any callable ``agent(text, chunk_size) -> list[str]`` via the
    constructor or ``set_agent_orchestrator``. If no agent is set, or the agent
    returns something unusable, the chunker falls back to recursive character
    splitting, so it always honours ``chunk_size``.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        agent: Optional[Agent] = None,
    ):
        """
        Args:
            chunk_size: Maximum chunk size, passed to the agent and used by the fallback.
            chunk_overlap: Overlap used by the fallback splitter.
            agent: Callable ``(text, chunk_size) -> list[str]``. Optional.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.agent_orchestrator: Optional[Agent] = agent
        # Used whenever no agent is set or the agent misbehaves, so ingestion
        # never fails and the size limit always holds.
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._warned = False  # log the fallback only once, not for every document

    def set_agent_orchestrator(self, orchestrator: Agent) -> None:
        """Set the callable responsible for interacting with the agent."""
        self.agent_orchestrator = orchestrator

    def _warn_once(self, message: str) -> None:
        """Log ``message`` as a warning, but only the first time per instance."""
        if not self._warned:
            logger.warning(message)
            self._warned = True

    def split_text(self, text: str) -> List[str]:
        """Split text with the agent, falling back to recursive splitting."""
        if self.agent_orchestrator is None:
            self._warn_once("AgenticChunker has no agent set; using recursive splitting.")
            return self._fallback.split_text(text)

        # The agent is external code (often an LLM call over the network), so
        # any exception is treated as "no usable answer", not as a fatal error.
        try:
            chunks = self.agent_orchestrator(text, self.chunk_size)
        except Exception as exc:  # the agent is external code, don't crash ingestion
            self._warn_once(f"Agent failed ({exc}); using recursive splitting.")
            return self._fallback.split_text(text)

        # Never trust the agent blindly: it must return a non-empty list of
        # non-blank strings, otherwise empty chunks would end up in the index.
        valid = (
            isinstance(chunks, list)
            and bool(chunks)
            and all(isinstance(c, str) and c.strip() for c in chunks)
        )
        if not valid:
            self._warn_once("Agent returned an invalid chunk list; using recursive splitting.")
            return self._fallback.split_text(text)
        return chunks
