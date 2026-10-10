"""Helpers for loading source data into LangChain ``Document`` objects."""
from __future__ import annotations

import os
from typing import List

import pandas as pd
from langchain_core.documents import Document

REVIEW_COLUMNS = ("Title", "Date", "Rating", "Review")


def load_reviews_csv(file_path: str) -> List[Document]:
    """
    Load restaurant reviews from a CSV file.

    The CSV must have the columns ``Title``, ``Date``, ``Rating`` and ``Review``.
    Rows without any text are skipped.

    Args:
        file_path: Path to the CSV file.

    Returns:
        One Document per review (title and review text joined).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_csv(file_path)
    missing = [col for col in REVIEW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path} is missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REVIEW_COLUMNS)}"
        )

    source = os.path.basename(file_path)
    documents: List[Document] = []
    for row_number, row in df.iterrows():
        title = "" if pd.isna(row["Title"]) else str(row["Title"]).strip()
        review = "" if pd.isna(row["Review"]) else str(row["Review"]).strip()
        text = f"{title} {review}".strip()
        if not text:
            continue

        # Chroma only accepts str/int/float/bool metadata (no numpy types, no NaN).
        metadata = {"source": source, "row": int(row_number)}
        if pd.notna(row["Date"]):
            metadata["date"] = str(row["Date"])
        if pd.notna(row["Rating"]):
            rating = float(row["Rating"])
            metadata["rating"] = int(rating) if rating.is_integer() else rating

        documents.append(Document(page_content=text, metadata=metadata))
    return documents
