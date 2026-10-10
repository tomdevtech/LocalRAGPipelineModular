import unittest
from unittest.mock import Mock, patch
from src.retrieval.retriever import Retriever
from src.config.settings import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

class TestRetriever(unittest.TestCase):
    """Tests the Retriever class, focusing on lazy initialization and retrieval."""

    def setUp(self):
        self.settings = Settings()
        # Ensure we mock OllamaEmbeddings globally if we were to test it, but here we focus on Chroma interaction.

    @patch('src.retrieval.retriever.OllamaEmbeddings')
    def test_init_with_settings(self, MockEmbeddings):
        """Test initialization when vector_store is None, verifying reliance on Settings."""
        # This test mainly checks if the structure correctly points to using settings when no store is provided.
        retriever = Retriever()
        # We expect the private initialization to have occurred when the first method is called,
        # but we can check the constructor's behavior.
        self.assertIsNone(retriever.vector_store)

    @patch('src.retrieval.retriever.OllamaEmbeddings')
    def test_lazy_initialization_on_first_call(self, MockEmbeddings):
        """Test that the vector store is initialized only upon the first call to retrieve/add_documents."""
        retriever = Retriever()

        # Check that the store is None before the call
        self.assertIsNone(retriever.vector_store)

        # Simulate a call that triggers initialization
        # We mock Chroma to track calls
        mock_chroma = Mock()
        MockEmbeddings.return_value = mock_chroma

        # Call a method that triggers initialization (e.g., retrieve)
        # Note: This is challenging to test perfectly without modifying the service to expose the store for inspection,
        # but we check if the internal state reflects a non-None store after use.
        with patch('src.retrieval.retriever.Chroma', return_value=mock_chroma) as MockChroma:
            retriever.retrieve("test query", k=5)
            self.assertIsNotNone(retriever.vector_store)
            MockChroma.assert_called_once()


    @patch('src.retrieval.retriever.Chroma')
    def test_add_documents(self, MockChroma):
        """Test adding documents to the mocked vector store."""
        mock_chroma_instance = MockChroma.return_value
        retriever = Retriever()

        docs = [Document(page_content="chunk 1", metadata={})]
        ids = ["id_1"]

        retriever.add_documents(docs, ids)

        # Verify the underlying Chroma instance received the calls correctly
        mock_chroma_instance.add_documents.assert_called_once_with(documents=docs, ids=ids)

    @patch('src.retrieval.retriever.Chroma')
    def test_retrieve(self, MockChroma):
        """Test retrieving documents from the mocked vector store."""
        mock_chroma_instance = MockChroma.return_value
        retriever = Retriever()

        mock_docs = [
            Document(page_content="Result 1"),
            Document(page_content="Result 2")
        ]

        mock_chroma_instance.as_retriever.return_value.invoke.return_value = mock_docs

        results = retriever.retrieve("test query", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "Result 1")
        mock_chroma_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 2})

    @patch('src.retrieval.retriever.Chroma')
    def test_is_empty_when_not_empty(self, MockChroma):
        """Test that is_empty returns False when documents exist."""
        mock_chroma_instance = MockChroma.return_value
        # Mock the private method to return count > 0
        mock_chroma_instance._collection.count.return_value = 1
        retriever = Retriever()

        self.assertFalse(retriever.is_empty())

    @patch('src.retrieval.retriever.Chroma')
    def test_is_empty_when_empty(self, MockChroma):
        """Test that is_empty returns True when no documents exist."""
        mock_chroma_instance = MockChroma.return_value
        # Mock the private method to return count == 0
        mock_chroma_instance._collection.count.return_value = 0
        retriever = Retriever()

        self.assertTrue(retriever.is_empty())

if __name__ == '__main__':
    unittest.main()