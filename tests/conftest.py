"""Shared test helpers: deterministic fake embeddings (no Ollama needed)."""
import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing embeddings: texts sharing words get similar vectors."""

    DIM = 64

    def _vector(self, text: str) -> list:
        vec = [0.0] * self.DIM
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode("utf-8")) % self.DIM] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()
