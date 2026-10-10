"""Factory for chunking strategies.

Dependency direction (no cycles):

    base.py  <-  fixed_size.py / recursive.py / ...  <-  strategies.py
"""
from __future__ import annotations

from typing import Dict, Type

from .agentic import AgenticChunker
from .base import BaseChunker
from .document import DocumentChunker
from .fixed_size import FixedSizeChunker
from .propositional import PropositionalChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

__all__ = [
    "BaseChunker",
    "AgenticChunker",
    "DocumentChunker",
    "FixedSizeChunker",
    "PropositionalChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "STRATEGIES",
    "get_chunker",
]

STRATEGIES: Dict[str, Type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "document": DocumentChunker,
    "semantic": SemanticChunker,
    "propositional": PropositionalChunker,
    "agentic": AgenticChunker,
}


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
    try:
        chunker_cls = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available strategies: {list(STRATEGIES)}"
        ) from None
    return chunker_cls(**kwargs)
