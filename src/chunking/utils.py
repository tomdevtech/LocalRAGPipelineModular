"""Small text helpers shared by the chunkers."""
from __future__ import annotations

import re
from typing import List

# Sentence end (. ! ?) followed by whitespace, or a paragraph break.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> List[str]:
    """Split text into stripped, non-empty sentences."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s and s.strip()]
