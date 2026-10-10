"""Tests for EmbeddingReranker using fixed vectors with known similarities."""
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from reranking.reranker import BaseReranker, EmbeddingReranker, get_reranker


class VectorEmbeddings:
    """Maps texts to fixed vectors so similarities are known in advance."""

    VECTORS = {
        "query": [1.0, 0.0],
        "Doc A": [1.0, 0.1],   # most similar
        "Doc B": [1.0, 1.0],   # medium
        "Doc C": [0.0, 1.0],   # orthogonal
        "Doc Zero": [0.0, 0.0],  # zero vector must not cause NaN / crash
    }

    def embed_query(self, text):
        """Look up the predefined vector for the query text."""
        return self.VECTORS[text]

    def embed_documents(self, texts):
        """Look up the predefined vectors for the document texts."""
        return [self.VECTORS[t] for t in texts]


@pytest.fixture
def reranker():
    """An EmbeddingReranker whose embedding model is replaced by VectorEmbeddings."""
    with patch("reranking.reranker.OllamaEmbeddings") as mock_ollama:
        instance = EmbeddingReranker()
        mock_ollama.assert_called_once_with(model="mxbai-embed-large")
    instance.embeddings = VectorEmbeddings()
    return instance


def docs(*names):
    """Create Document objects from the given page contents."""
    return [Document(page_content=n) for n in names]


def test_rerank_empty_documents(reranker):
    """No documents in, no documents out."""
    assert reranker.rerank("query", [], top_k=3) == []


def test_rerank_orders_by_cosine_similarity(reranker):
    """Documents are returned best match first, cut off at top_k."""
    result = reranker.rerank("query", docs("Doc C", "Doc B", "Doc A"), top_k=2)
    assert [d.page_content for d in result] == ["Doc A", "Doc B"]


def test_rerank_top_k_larger_than_input(reranker):
    """A top_k larger than the input returns all documents, sorted."""
    result = reranker.rerank("query", docs("Doc C", "Doc A"), top_k=10)
    assert [d.page_content for d in result] == ["Doc A", "Doc C"]


def test_rerank_handles_zero_vector(reranker):
    """A zero vector must not produce NaN or a crash; it ranks last."""
    result = reranker.rerank("query", docs("Doc Zero", "Doc A"), top_k=2)
    assert [d.page_content for d in result] == ["Doc A", "Doc Zero"]


def test_get_reranker_factory():
    """The factory builds the known type and rejects unknown ones."""
    with patch("reranking.reranker.OllamaEmbeddings"):
        assert isinstance(get_reranker("embedding"), BaseReranker)
    with pytest.raises(ValueError, match="Unknown reranker type"):
        get_reranker("magic")
