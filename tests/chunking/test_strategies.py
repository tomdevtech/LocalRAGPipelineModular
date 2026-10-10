"""Tests for all chunking strategies and the chunker factory."""
import os
import subprocess
import sys

import pytest

from chunking.base import BaseChunker
from chunking.strategies import (
    AgenticChunker,
    DocumentChunker,
    FixedSizeChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)

SAMPLE = (
    "This is sentence one. This is sentence two. "
    "And this is the final sentence three.\n\nThis is a new paragraph."
)


@pytest.mark.parametrize(
    "module", ["fixed_size", "recursive", "document", "semantic", "propositional", "agentic", "strategies"]
)
def test_every_module_imports_first_without_circular_import(module):
    """Importing any chunking module first must work (the original ImportError).

    Runs in a fresh interpreter so no previously imported module can hide a cycle.
    """
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    result = subprocess.run(
        [sys.executable, "-c", f"import chunking.{module}"],
        env={**os.environ, "PYTHONPATH": src},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name, cls",
    [
        ("fixed_size", FixedSizeChunker),
        ("recursive", RecursiveChunker),
        ("document", DocumentChunker),
        ("semantic", SemanticChunker),
        ("propositional", PropositionalChunker),
        ("agentic", AgenticChunker),
    ],
)
def test_factory_returns_requested_type(name, cls):
    """The factory returns an instance of the class registered for the strategy name."""
    chunker = get_chunker(name, chunk_size=500, chunk_overlap=100)
    assert isinstance(chunker, cls)
    assert isinstance(chunker, BaseChunker)


def test_factory_unknown_strategy():
    """An unknown strategy name raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("unknown_strategy", chunk_size=100)


def test_fixed_size_chunker():
    """Fixed-size chunking yields many chunks that never exceed chunk_size."""
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=5).split_text("A" * 5000)
    assert len(chunks) > 100
    assert all(len(c) <= 20 for c in chunks)


def test_recursive_chunker():
    """Recursive chunking splits long text and respects chunk_size."""
    chunks = RecursiveChunker(chunk_size=50, chunk_overlap=10).split_text(SAMPLE)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_document_chunker_splits_paragraphs():
    """Paragraphs that do not fit into one chunk end up in separate chunks."""
    chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).split_text(SAMPLE)
    assert len(chunks) == 2
    assert chunks[1] == "This is a new paragraph."


def test_propositional_chunker_one_chunk_per_sentence():
    """Each sentence becomes its own chunk, in the original order."""
    chunks = PropositionalChunker(chunk_size=200, chunk_overlap=10).split_text(SAMPLE)
    assert chunks == [
        "This is sentence one.",
        "This is sentence two.",
        "And this is the final sentence three.",
        "This is a new paragraph.",
    ]


def test_propositional_chunker_respects_chunk_size_and_extractor():
    """Oversized statements are hard-split, and a custom extractor replaces the sentence heuristic (blank results dropped)."""
    long_sentence = "word " * 100
    chunks = PropositionalChunker(chunk_size=50, chunk_overlap=0).split_text(long_sentence)
    assert len(chunks) > 1 and all(len(c) <= 50 for c in chunks)

    custom = PropositionalChunker(extractor=lambda text: ["fact A", " ", "fact B"])
    assert custom.split_text("anything") == ["fact A", "fact B"]


class TopicEmbeddings:
    """Sentences mentioning 'pizza' get one vector, everything else another."""

    def embed_documents(self, texts):
        """Return [1, 0] for pizza sentences and [0, 1] for all others."""
        return [[1.0, 0.0] if "pizza" in t.lower() else [0.0, 1.0] for t in texts]


def test_semantic_chunker_splits_on_topic_change_and_keeps_all_text():
    """A topic change starts a new chunk and no text is lost."""
    text = "I love pizza. The pizza was great. The service was slow. The waiter was rude."
    chunks = SemanticChunker(chunk_size=500, embeddings=TopicEmbeddings()).split_text(text)
    assert chunks == [
        "I love pizza. The pizza was great.",
        "The service was slow. The waiter was rude.",
    ]


@pytest.mark.parametrize("text", ["", "   ", "One sentence only."])
def test_semantic_chunker_edge_cases(text):
    """Empty input yields no chunks; a single sentence is returned unchanged."""
    chunks = SemanticChunker(embeddings=TopicEmbeddings()).split_text(text)
    assert chunks == ([text] if text.strip() else [])


def test_semantic_chunker_respects_chunk_size():
    """Even for one topic, chunks stay within chunk_size and together rebuild the text."""
    text = " ".join(f"Pizza fact number {i}." for i in range(30))
    chunks = SemanticChunker(chunk_size=80, embeddings=TopicEmbeddings()).split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    assert " ".join(chunks) == text


def test_agentic_chunker_without_agent_falls_back_to_size_limited_split():
    """Without an agent the chunker still honours chunk_size via the fallback."""
    chunks = AgenticChunker(chunk_size=50, chunk_overlap=10).split_text(SAMPLE)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_agentic_chunker_uses_agent_and_validates_output():
    """The agent's chunks are used; invalid output or an exception triggers the fallback."""
    chunker = AgenticChunker(chunk_size=50, chunk_overlap=10)
    chunker.set_agent_orchestrator(lambda text, size: ["part 1", "part 2"])
    assert chunker.split_text(SAMPLE) == ["part 1", "part 2"]

    chunker.set_agent_orchestrator(lambda text, size: "not a list")
    assert len(chunker.split_text(SAMPLE)) > 1  # fallback

    def broken(text, size):
        """Simulate an agent that crashes."""
        raise RuntimeError("boom")

    chunker.set_agent_orchestrator(broken)
    assert len(chunker.split_text(SAMPLE)) > 1  # fallback
