import unittest
from src.chunking.strategies import (
    BaseChunker, FixedSizeChunker, RecursiveChunker, DocumentChunker,
    SemanticChunker, PropositionalChunker, AgenticChunker, get_chunker
)
from langchain_core.documents import Document

class TestChunkingStrategies(unittest.TestCase):
    """Tests all implemented chunking strategies."""

    def setUp(self):
        self.sample_text = "This is sentence one. This is sentence two. And this is the final sentence three.\n\nThis is a new paragraph."
        self.long_text = "A" * 5000

    def test_get_chunker_factory(self):
        """Test that the factory function correctly returns the specified chunker type."""
        chunker = get_chunker("fixed_size", chunk_size=500, chunk_overlap=100)
        self.assertIsInstance(chunker, FixedSizeChunker)

        chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=100)
        self.assertIsInstance(chunker, RecursiveChunker)

        with self.assertRaises(ValueError):
            get_chunker("unknown_strategy", chunk_size=100)

    def test_fixed_size_chunker(self):
        """Test fixed size chunking."""
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.split_text(self.long_text)
        # Since it's simple character splitter, we can assert against size
        self.assertGreater(len(chunks), 100) # Should yield many chunks
        self.assertTrue(len(chunks[0]) <= 20 + 5) # Check size constraint

    def test_recursive_chunker(self):
        """Test recursive chunking."""
        # Use smaller size for predictable testing of separators
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split_text(self.sample_text)
        # Should split based on newlines/periods if possible
        self.assertGreater(len(chunks), 1)
        self.assertTrue(len(chunks[0]) <= 50)

    def test_document_chunker(self):
        """Test document chunking (using double newline separator)."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_text(self.sample_text)
        # Should split based on paragraph breaks
        self.assertEqual(len(chunks), 2)
        self.assertTrue(len(chunks[0]) > 10)

    def test_semantic_and_propositional_chunker_executability(self):
        """Test executable baseline for advanced chunkers."""
        # Test SemanticChunker (functional baseline)
        semantic_chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        chunks_sem = semantic_chunker.split_text(self.sample_text)
        self.assertGreater(len(chunks_sem), 1)

        # Test PropositionalChunker (functional baseline)
        propositional_chunker = PropositionalChunker(chunk_size=50, chunk_overlap=10)
        chunks_prop = propositional_chunker.split_text(self.sample_text)
        self.assertGreater(len(chunks_prop), 1)

    def test_agentic_chunker_executability(self):
        """Test AgenticChunker execution path."""
        agentic_chunker = AgenticChunker(chunk_size=100, chunk_overlap=10)
        chunks = agentic_chunker.split_text(self.sample_text)
        self.assertEqual(len(chunks), 1) # Should return one chunk as per the current mock
        self.assertEqual(chunks[0], self.sample_text)


if __name__ == '__main__':
    unittest.main()