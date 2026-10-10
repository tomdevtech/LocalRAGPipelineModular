"""Unit tests for RAGPipeline orchestration, with all services mocked."""
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from chunking.base import BaseChunker
from config.settings import Settings
from pipeline import RAGPipeline
from reranking.reranker import BaseReranker
from retrieval.retriever import Retriever


def make_pipeline(tmp_path, **overrides):
    """Build a real pipeline whose paths point into ``tmp_path``."""
    settings = Settings(
        data_path=str(tmp_path), vector_db_path=str(tmp_path / "db"), **overrides
    )
    return RAGPipeline(settings=settings)


@pytest.fixture
def pipeline(tmp_path):
    """Real pipeline whose services are replaced by mocks."""
    p = make_pipeline(tmp_path)
    p.chunker = Mock(spec=BaseChunker)
    p.retriever = Mock(spec=Retriever)
    p.reranker = Mock(spec=BaseReranker)
    return p


def test_initialization_creates_all_components(tmp_path):
    """All components are created, and reranking can be switched off."""
    p = make_pipeline(tmp_path)
    assert isinstance(p.chunker, BaseChunker)
    assert isinstance(p.retriever, Retriever)
    assert isinstance(p.reranker, BaseReranker)
    assert make_pipeline(tmp_path, use_reranking=False).reranker is None


def test_retriever_and_reranker_receive_configured_models(tmp_path):
    """The configured models and collection reach the retriever and reranker."""
    p = make_pipeline(tmp_path, embedding_model="other-model", collection_name="coll")
    assert p.reranker.embedding_model == "other-model"
    assert p.retriever._embedding_model == "other-model"
    assert p.retriever._collection_name == "coll"


def test_add_documents_flow(pipeline):
    """Documents are chunked and passed on with metadata and unique IDs."""
    pipeline.chunker.split_text.return_value = ["chunk A", "chunk B"]
    doc = Document(page_content="long content one", metadata={"source": "doc1"})

    added = pipeline.add_documents([doc])

    assert added == 2
    pipeline.chunker.split_text.assert_called_once_with("long content one")
    chunks, ids = pipeline.retriever.add_documents.call_args.args
    assert chunks == [
        Document(page_content="chunk A", metadata={"source": "doc1", "chunk_id": 0, "parent_id": 0}),
        Document(page_content="chunk B", metadata={"source": "doc1", "chunk_id": 1, "parent_id": 0}),
    ]
    assert len(ids) == len(set(ids)) == 2


def test_chunk_ids_are_stable_and_content_based(pipeline):
    """Same input gives the same IDs (idempotent); different content gives different IDs."""
    pipeline.chunker.split_text.return_value = ["chunk A"]
    doc = Document(page_content="x", metadata={"source": "s"})

    pipeline.add_documents([doc])
    first = pipeline.retriever.add_documents.call_args.args[1]
    pipeline.add_documents([doc])
    second = pipeline.retriever.add_documents.call_args.args[1]
    assert first == second  # re-indexing updates instead of duplicating

    pipeline.chunker.split_text.return_value = ["different chunk"]
    pipeline.add_documents([doc])
    assert pipeline.retriever.add_documents.call_args.args[1] != first  # no collisions


def test_add_documents_with_no_chunks_does_not_touch_store(pipeline):
    """Empty chunker output must not trigger a call to the vector store."""
    pipeline.chunker.split_text.return_value = []
    assert pipeline.add_documents([Document(page_content="")]) == 0
    pipeline.retriever.add_documents.assert_not_called()


def test_retrieve_without_reranker(pipeline):
    """Without reranking exactly k documents are requested and returned."""
    pipeline.reranker = None
    pipeline.retriever.retrieve.return_value = [Document(page_content=f"R{i}") for i in range(1, 3)]

    results = pipeline.retrieve("test query", k=2)

    pipeline.retriever.retrieve.assert_called_once_with("test query", k=2)
    assert [d.page_content for d in results] == ["R1", "R2"]


def test_retrieve_with_reranker_fetches_wider_pool(pipeline):
    """With reranking, 2*k candidates are fetched and the reranked top-k is returned."""
    pool = [Document(page_content=f"R{i}") for i in range(1, 5)]
    reranked = [Document(page_content="Top 1"), Document(page_content="Top 2")]
    pipeline.retriever.retrieve.return_value = pool
    pipeline.reranker.rerank.return_value = reranked

    results = pipeline.retrieve("test query", k=2)

    pipeline.retriever.retrieve.assert_called_once_with("test query", k=4)
    pipeline.reranker.rerank.assert_called_once_with("test query", pool, top_k=2)
    assert results == reranked


def test_retrieve_default_k_comes_from_settings(pipeline):
    """Without an explicit k, the settings decide (rerank_top_k or k)."""
    pipeline.retriever.retrieve.return_value = []

    pipeline.retrieve("q")  # reranker on -> rerank_top_k (3) -> pool 6
    pipeline.retriever.retrieve.assert_called_with("q", k=2 * pipeline.settings.rerank_top_k)

    pipeline.reranker = None
    pipeline.retrieve("q")  # reranker off -> k (5)
    pipeline.retriever.retrieve.assert_called_with("q", k=pipeline.settings.k)


def test_retrieve_with_empty_pool_skips_reranker(pipeline):
    """An empty search result is returned as is, without calling the reranker."""
    pipeline.retriever.retrieve.return_value = []
    assert pipeline.retrieve("q", k=2) == []
    pipeline.reranker.rerank.assert_not_called()


def test_retrieve_rejects_invalid_k(pipeline):
    """k must be positive."""
    with pytest.raises(ValueError):
        pipeline.retrieve("q", k=0)


def test_get_context_joins_documents(pipeline):
    """The context is the page contents joined by blank lines."""
    pipeline.reranker = None
    pipeline.retriever.retrieve.return_value = [
        Document(page_content="Context snippet 1"),
        Document(page_content="Context snippet 2"),
    ]
    assert pipeline.get_context("q", k=2) == "Context snippet 1\n\nContext snippet 2"
