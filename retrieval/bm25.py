"""Keyword search over the same chunks as the vector store.

BM25 is there for exact tokens: section numbers (``s 5D``), case names, and
court codes. It searches the full collection, not the vector-store hits.
"""

from __future__ import annotations

import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into tokens while keeping terms such as 5D intact."""

    return re.findall(r"[A-Za-z0-9]+", text.lower())


def build_index(texts: list[str], chunk_ids: list[str]):
    """Build a BM25 index. texts and chunk_ids must be aligned."""

    if len(texts) != len(chunk_ids):
        raise ValueError("texts and chunk_ids must have the same length")

    tokenized_texts = [tokenize(text) for text in texts]

    bm25 = BM25Okapi(tokenized_texts)

    return {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
    }


def search_bm25(
    index,
    query: str,
    top_k: int = 20
) -> list[tuple[str, float]]:
    """Return (chunk_id, score) pairs."""

    query_tokens = tokenize(query)

    scores = index["bm25"].get_scores(query_tokens)

    ranked = sorted(
        zip(index["chunk_ids"], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        (chunk_id, float(score))
        for chunk_id, score in ranked[:top_k]
    ]