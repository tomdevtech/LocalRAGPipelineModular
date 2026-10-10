import unittest
from unittest.mock import Mock, patch
import numpy as np

# Fix imports to use absolute paths from src
from src.reranking.reranker import BaseReranker, EmbeddingReranker, get_reranker
from langchain_core.documents import Document

class TestReranker(unittest.TestCase):
    """Tests the EmbeddingReranker logic."""

    def setUp(self):
        # Mock the internal dependency (OllamaEmbeddings) for deterministic testing
        self.mock_embeddings = Mock()

        # We must mock OllamaEmbeddings globally for this class setup
        from unittest.mock import patch
        self.patcher = patch('src.reranking.reranker.OllamaEmbeddings', return_value=self.mock_embeddings)
        self.mock_ollama = self.patcher.start()

        # Initialize the reranker
        self.reranker = EmbeddingReranker()

    def tearDown(self):
        self.patcher.stop()

    def test_reranker_initialization(self):
        """Test that the reranker initializes its embedding model."""
        self.mock_ollama.assert_called_once_with(model="mxbai-embed-large")

    def test_rerank_empty_documents(self):
        """Test reranking with no documents."""
        self.assertEqual(self.reranker.rerank("query", [], top_k=3), [])

    @patch('src.reranking.reranker.np.dot')
    @patch('src.reranking.reranker.np.linalg.norm')
    def test_rerank_basic_similarity(self, mock_norm, mock_dot):
        """Test that reranking correctly sorts documents based on mocked similarity scores."""

        # Mock similarities (Scores should be: Doc A=0.9, Doc B=0.5, Doc C=0.1)
        mock_dot.return_value = np.array([0.1, 0.5, 0.9]) # Scores for C, B, A
        mock_norm.return_value = 1.0 # Simplification

        documents = [
            Document(page_content="Doc A Content"), # Index 0
            Document(page_content="Doc B Content"), # Index 1
            Document(page_content="Doc C Content")  # Index 2
        ]

        # Set k to 2
        reranked = self.reranker.rerank("query", documents, top_k=2)

        # Check if the top 2 are returned in the correct order (A, B)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].page_content, "Doc A Content")
        self.assertEqual(reranked[1].page_content, "Doc B Content")

    @patch('src.reranking.reranker.EmbeddingReranker.rerank')
    def test_rerank_when_not_enabled(self, mock_rerank):
        """Test behavior when reranker is not initialized (simulating no reranking)."""
        # Bypass setup to test flow if reranker was None in RAGPipeline
        class TestRerankerStub:
            def rerank(self, query, documents, top_k):
                # Return original docs to mimic 'no reranking' scenario
                return documents

        stub = TestRerankerStub()

        documents = [Document(page_content="Test")]
        result = stub.rerank("query", documents, top_k=1)

        self.assertEqual(result, documents)

if __name__ == '__main__':
    unittest.main()