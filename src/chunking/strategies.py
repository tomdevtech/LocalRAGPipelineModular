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
        pass

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
    # Imports from new modules
    from .fixed_size import FixedSizeChunker
    from .recursive import RecursiveChunker
    from .document import DocumentChunker
    from .semantic import SemanticChunker
    from .propositional import PropositionalChunker
    from .agentic import AgenticChunker

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