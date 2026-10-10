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
        return self.VECTORS[text]

    def embed_documents(self, texts):
        return [self.VECTORS[t] for t in texts]


@pytest.fixture
def reranker():
    with patch("reranking.reranker.OllamaEmbeddings") as mock_ollama:
        instance = EmbeddingReranker()
        mock_ollama.assert_called_once_with(model="mxbai-embed-large")
    instance.embeddings = VectorEmbeddings()
    return instance


def docs(*names):
    return [Document(page_content=n) for n in names]


def test_rerank_empty_documents(reranker):
    assert reranker.rerank("query", [], top_k=3) == []


def test_rerank_orders_by_cosine_similarity(reranker):
    result = reranker.rerank("query", docs("Doc C", "Doc B", "Doc A"), top_k=2)
    assert [d.page_content for d in result] == ["Doc A", "Doc B"]


def test_rerank_top_k_larger_than_input(reranker):
    result = reranker.rerank("query", docs("Doc C", "Doc A"), top_k=10)
    assert [d.page_content for d in result] == ["Doc A", "Doc C"]


def test_rerank_handles_zero_vector(reranker):
    result = reranker.rerank("query", docs("Doc Zero", "Doc A"), top_k=2)
    assert [d.page_content for d in result] == ["Doc A", "Doc Zero"]


def test_get_reranker_factory():
    with patch("reranking.reranker.OllamaEmbeddings"):
        assert isinstance(get_reranker("embedding"), BaseReranker)
    with pytest.raises(ValueError, match="Unknown reranker type"):
        get_reranker("magic")
