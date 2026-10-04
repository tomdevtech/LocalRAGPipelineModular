"""
Command-line interface for the RAG pipeline.
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from langchain_core.documents import Document

from rag_pipeline.config.settings import Settings
from rag_pipeline.pipeline import RAGPipeline


def _load_documents_from_csv(file_path: str) -> List[Document]:
    """
    Load documents from a CSV file (expects columns: Title, Review, etc.).

    This is a helper function for the example data format.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A list of Document objects.
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    documents: List[Document] = []
    for i, row in df.iterrows():
        # Combine Title and Review for the page content
        content = f"{row['Title']} {row['Review']}"
        # Metadata can include other columns
        metadata = {
            key: row[key]
            for key in df.columns
            if key not in ["Title", "Review"]
        }
        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
                id=str(i),
            )
        )
    return documents


def main() -> None:
    """Run the RAG pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Run a Retrieval-Augmented Generation pipeline for question answering."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data file (CSV) to ingest. If not provided, the pipeline will use the existing vector store.",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask the pipeline. If not provided, the pipeline will enter interactive mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML configuration file. If not provided, default settings are used.",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of documents to retrieve (overrides config).",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        help="Chunking strategy to use (overrides config).",
    )
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        help="Enable reranking (overrides config).",
    )
    parser.add_argument(
        "--no-reranking",
        dest="use_reranking",
        action="store_false",
        help="Disable reranking (overrides config).",
    )
    parser.set_defaults(use_reranking=None)  # None means use config value

    args = parser.parse_args()

    # Load settings
    if args.config:
        try:
            settings = Settings.from_yaml(args.config)
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        settings = Settings()

    # Override settings with command-line arguments
    if args.k is not None:
        settings.k = args.k
    if args.chunking_strategy is not None:
        settings.chunking_strategy = args.chunking_strategy
    if args.use_reranking is not None:
        settings.use_reranking = args.use_reranking

    # Initialize the pipeline
    pipeline = RAGPipeline(settings=settings)

    # If a data file is provided, ingest it
    if args.data:
        try:
            documents = _load_documents_from_csv(args.data)
            pipeline.add_documents(documents)
            print(f"Ingested {len(documents)} documents from {args.data}")
        except Exception as e:
            print(f"Error ingesting data: {e}", file=sys.stderr)
            sys.exit(1)

    # If a question is provided, answer it and exit
    if args.question:
        try:
            context = pipeline.get_context(args.question)
            print(f"Question: {args.question}")
            print(f"Context:\n{context}")
            # In a real application, you would now pass the context to an LLM.
            # For this example, we just print the context.
        except Exception as e:
            print(f"Error processing question: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Otherwise, enter interactive mode
    print("RAG Pipeline Interactive Mode")
    print("Type your question and press Enter to get the relevant context.")
    print("Type 'q' or 'quit' to exit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if question.lower() in ("q", "quit"):
            print("Exiting...")
            break

        if not question:
            continue

        try:
            context = pipeline.get_context(question)
            print(f"\nContext:\n{context}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()