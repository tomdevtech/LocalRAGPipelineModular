"""End-to-end test with a real Chroma store and fake embeddings (no Ollama needed)."""
import os

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import Settings
from loaders import load_reviews_csv
from pipeline import RAGPipeline
from reranking.reranker import EmbeddingReranker
from retrieval.retriever import Retriever

REVIEWS_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "realistic_restaurent_reviews.csv")


@pytest.fixture
def pipeline(tmp_path, fake_embeddings):
    """Pipeline backed by a real, temporary Chroma store and fake embeddings."""
    settings = Settings(data_path=str(tmp_path), vector_db_path=str(tmp_path / "db"))
    p = RAGPipeline(settings=settings)
    p.retriever = Retriever(
        vector_store=Chroma(
            collection_name="e2e",
            persist_directory=str(tmp_path / "db"),
            embedding_function=fake_embeddings,
        )
    )
    p.reranker = EmbeddingReranker()
    p.reranker.embeddings = fake_embeddings
    return p


def test_index_and_retrieve(pipeline):
    """Indexed documents can be found again by a related query."""
    assert pipeline.retriever.is_empty()

    pipeline.add_documents(
        [
            Document(page_content="Wood-fired pizza with a crispy crust", metadata={"source": "a"}),
            Document(page_content="Slow service and cold soup", metadata={"source": "b"}),
            Document(page_content="Great sushi and fresh fish", metadata={"source": "c"}),
        ]
    )
    assert not pipeline.retriever.is_empty()

    context = pipeline.get_context("crispy pizza crust", k=1)
    assert context == "Wood-fired pizza with a crispy crust"


def test_reindexing_is_idempotent_and_batches_do_not_overwrite(pipeline):
    """Re-adding identical data creates no duplicates; new data does not overwrite old."""
    store = pipeline.retriever.vector_store
    docs = [Document(page_content=f"review number {i}", metadata={"source": "x"}) for i in range(5)]

    pipeline.add_documents(docs)
    pipeline.add_documents(docs)  # same data again
    assert store._collection.count() == 5

    pipeline.add_documents([Document(page_content="a brand new review", metadata={"source": "y"})])
    assert store._collection.count() == 6  # 2nd batch must not overwrite the 1st


def test_real_csv_loads_and_indexes(pipeline):
    """The bundled review CSV can be loaded, indexed and queried."""
    documents = load_reviews_csv(REVIEWS_CSV)
    assert len(documents) == 123

    assert pipeline.add_documents(documents) >= 123
    assert pipeline.get_context("pizza", k=3)
