"""Keyword search over the same chunks as the vector store.

BM25 is there for exact tokens: section numbers (``s 5D``), case names, and
court codes. It searches the full collection, not the vector-store hits.

Install later: rank-bm25
"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Split text into tokens for BM25.

    Keep section markers such as ``5D`` intact. A naive split on punctuation
    will break those.
    """
    raise NotImplementedError


def build_index(texts: list[str], chunk_ids: list[str]):
    """Build a BM25 index. ``texts`` and ``chunk_ids`` must be aligned."""
    raise NotImplementedError


def search_bm25(index, query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Return ``(chunk_id, score)`` pairs. Run this beside vector search, not after it."""
    raise NotImplementedError
