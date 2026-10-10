from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from retrieval.retriever import Retriever


def test_constructor_is_lazy():
    assert Retriever().vector_store is None


def test_lazy_initialization_uses_constructor_arguments():
    """The values given to the constructor must reach Chroma/Ollama (they were ignored before)."""
    with patch("retrieval.retriever.OllamaEmbeddings") as mock_emb, patch(
        "retrieval.retriever.Chroma"
    ) as mock_chroma:
        retriever = Retriever(
            embedding_model="my-model", persist_directory="/tmp/x", collection_name="my_coll"
        )
        retriever.retrieve("q", k=3)

        mock_emb.assert_called_once_with(model="my-model")
        mock_chroma.assert_called_once_with(
            collection_name="my_coll",
            persist_directory="/tmp/x",
            embedding_function=mock_emb.return_value,
        )
        # created once, then reused
        retriever.retrieve("q2", k=3)
        mock_chroma.assert_called_once()


def test_retrieve():
    store = MagicMock()
    store.as_retriever.return_value.invoke.return_value = [
        Document(page_content="Result 1"),
        Document(page_content="Result 2"),
    ]
    results = Retriever(vector_store=store).retrieve("test query", k=2)

    assert [d.page_content for d in results] == ["Result 1", "Result 2"]
    store.as_retriever.assert_called_once_with(search_kwargs={"k": 2})


def test_add_documents_passes_ids():
    store = MagicMock()
    docs = [Document(page_content="chunk 1")]
    Retriever(vector_store=store).add_documents(docs, ["id_1"])
    store.add_documents.assert_called_once_with(documents=docs, ids=["id_1"])


def test_add_documents_batches_large_inputs():
    store = MagicMock()
    docs = [Document(page_content=str(i)) for i in range(1201)]
    ids = [str(i) for i in range(1201)]
    Retriever(vector_store=store).add_documents(docs, ids)

    assert store.add_documents.call_count == 3
    sent = [d for call in store.add_documents.call_args_list for d in call.kwargs["documents"]]
    assert sent == docs


def test_add_documents_rejects_mismatched_ids():
    with pytest.raises(ValueError):
        Retriever(vector_store=MagicMock()).add_documents([Document(page_content="a")], ["1", "2"])


@pytest.mark.parametrize("ids, expected", [([], True), (["a"], False)])
def test_is_empty(ids, expected):
    store = MagicMock()
    store.get.return_value = {"ids": ids}
    assert Retriever(vector_store=store).is_empty() is expected
