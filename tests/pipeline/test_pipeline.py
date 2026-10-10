import unittest
from src.pipeline import RAGPipeline
from src.config.settings import Settings
from src.chunking.strategies import BaseChunker
from src.retrieval.retriever import Retriever
from src.reranking.reranker import BaseReranker

class TestRAGPipeline(unittest.TestCase):
    """Tests the core RAGPipeline orchestration logic."""

    def setUp(self):
        # Use mock settings for predictable testing
        self.settings = Settings()
        self.pipeline = RAGPipeline(settings=self.settings)

        # Mock services for isolation testing
        self.mock_chunker = unittest.mock.Mock(spec=BaseChunker)
        self.pipeline.chunker = self.mock_chunker

        self.mock_retriever = unittest.mock.Mock(spec=Retriever)
        self.pipeline.retriever = self.mock_retriever

        self.mock_reranker = unittest.mock.Mock(spec=BaseReranker)
        self.pipeline.reranker = self.mock_reranker

    def test_pipeline_initialization_success(self):
        """Test that the RAGPipeline initializes all components successfully."""
        self.assertIsInstance(self.pipeline.chunker, BaseChunker)
        self.assertIsInstance(self.pipeline.retriever, Retriever)
        if self.settings.use_reranking:
            self.assertIsNotNone(self.pipeline.reranker)

    def test_add_documents_flow(self):
        """Test the document ingestion and chunking flow."""
        doc1 = Document(page_content="long content one", metadata={"source": "doc1"})
        documents = [doc1]

        # Mock chunker output: returns two chunks for one document
        mock_chunks = ["chunk A", "chunk B"]
        self.mock_chunker.split_text.return_value = mock_chunks

        self.pipeline.add_documents(documents)

        # Verify chunker was called
        self.mock_chunker.split_text.assert_called_once_with("long content one")

        # Verify retriever was called with correctly formatted chunks and IDs
        expected_chunks = [
            Document(page_content="chunk A", metadata={"source": "doc1", "chunk_id": 0, "parent_id": 0}),
            Document(page_content="chunk B", metadata={"source": "doc1", "chunk_id": 1, "parent_id": 0}),
        ]
        self.mock_retriever.add_documents.assert_called_once_with(
            expected_chunks,
            ["0-0", "0-1"]
        )

    def test_retrieve_no_reranker(self):
        """Test retrieval flow when reranking is disabled."""
        self.pipeline.settings.use_reranking = False

        mock_initial_docs = [Document(page_content="Raw doc 1"), Document(page_content="Raw doc 2")]

        # Mock retriever to return more than K*2 (e.g., 4)
        self.mock_retriever.retrieve.return_value = [
            Document(page_content="R1"), Document(page_content="R2"),
            Document(page_content="R3"), Document(page_content="R4")
        ]

        # We request k=2, so we expect the first 2 to be returned.
        results = self.pipeline.retrieve("test query", k=2)

        # Verify retriever was called with k*2 (4)
        self.mock_retriever.retrieve.assert_called_once_with("test query", k=4)

        # Verify the pipeline returned the top K (2) results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "R1")

    def test_retrieve_with_reranker(self):
        """Test retrieval flow when reranking is enabled."""
        self.pipeline.settings.use_reranking = True

        mock_initial_docs = [Document(page_content="R1"), Document(page_content="R2"), Document(page_content="R3"), Document(page_content="R4")]
        mock_reranked_docs = [Document(page_content="Top Reranked 1"), Document(page_content="Top Reranked 2")]

        # Mock retriever to return pool of 4
        self.mock_retriever.retrieve.return_value = mock_initial_docs

        # Mock reranker to return final pool of 2
        self.mock_reranker.rerank.return_value = mock_reranked_docs

        results = self.pipeline.retrieve("test query", k=2)

        # Verify retriever was called with k*2 (4)
        self.mock_retriever.retrieve.assert_called_once_with("test query", k=4)

        # Verify reranker was called with the initial pool and target K
        self.mock_reranker.rerank.assert_called_once_with(
            "test query", mock_initial_docs, top_k=2
        )

        # Verify pipeline returned the final reranked list (2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "Top Reranked 1")

    def test_get_context_flow(self):
        """Test the end-to-end context generation."""
        # Setup mock to return 2 documents during retrieval
        mock_docs = [
            Document(page_content="Context snippet 1"),
            Document(page_content="Context snippet 2")
        ]
        self.mock_retriever.retrieve.return_value = mock_docs

        context = self.pipeline.get_context("test query")

        # Verify retriever was called
        self.mock_retriever.retrieve.assert_called_once_with("test query")

        # Verify context formatting is correct
        expected_context = "Context snippet 1\n\nContext snippet 2"
        self.assertEqual(context, expected_context)

if __name__ == '__main__':
    unittest.main()