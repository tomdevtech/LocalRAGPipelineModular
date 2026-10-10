"""Command line interface: index review data and ask questions about it."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from config.settings import Settings
from loaders import load_reviews_csv
from pipeline import RAGPipeline

DEFAULT_DATA_FILE = "realistic_restaurent_reviews.csv"
OLLAMA_HINT = "\nIs Ollama running and are the models pulled (ollama pull <model>)?"

PROMPT_TEMPLATE = """
You are a perfect summarizer for restaurant reviews.

The details you can find here: {reviews}

And here is the question to answer: {question}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-rag",
        description="Index restaurant reviews and ask questions about them (local RAG).",
    )
    parser.add_argument("--config", help="Path to a YAML settings file.")
    parser.add_argument("--data", help="CSV file to index before answering.")
    parser.add_argument("--question", "-q", help="Ask a single question and exit.")
    parser.add_argument("-k", type=int, help="Number of context documents to use.")
    return parser


def index_csv(pipeline: RAGPipeline, path: str) -> None:
    documents = load_reviews_csv(path)
    count = pipeline.add_documents(documents)
    print(f"Indexed {len(documents)} reviews ({count} chunks) from {path}.")


def answer(pipeline: RAGPipeline, chain, question: str, k: Optional[int]) -> str:
    context = pipeline.get_context(question, k=k)
    return chain.invoke({"reviews": context, "question": question})


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = Settings.from_yaml(args.config) if args.config else Settings()
        pipeline = RAGPipeline(settings=settings)

        if args.data:
            index_csv(pipeline, args.data)
        elif pipeline.retriever.is_empty():
            default_csv = os.path.join(settings.data_path, DEFAULT_DATA_FILE)
            if os.path.isfile(default_csv):
                print("Vector store is empty - indexing the default data set ...")
                index_csv(pipeline, default_csv)
            else:
                print(
                    "Vector store is empty. Index data first: local-rag --data path/to/reviews.csv",
                    file=sys.stderr,
                )
                return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # e.g. Ollama not running while embedding
        print(f"Error: {exc}{OLLAMA_HINT}", file=sys.stderr)
        return 1

    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | OllamaLLM(model=settings.llm_model)

    if args.question:
        try:
            print(answer(pipeline, chain, args.question, args.k))
        except Exception as exc:  # connection problems, missing models, ...
            print(f"Error: {exc}{OLLAMA_HINT}", file=sys.stderr)
            return 1
        return 0

    while True:
        try:
            question = input("\nAsk your question (or type 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() == "q":
            break
        if not question:
            continue
        try:
            print(f"\n{answer(pipeline, chain, question, args.k)}")
        except Exception as exc:
            print(f"Error: {exc}{OLLAMA_HINT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
