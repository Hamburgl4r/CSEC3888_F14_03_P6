"""Load processed chunks, run both retrievers, merge, then filter.

Input files (produced by scripts/process_corpus.py):

    data/processed/judgment_chunks.jsonl
    data/processed/legislation_chunks.jsonl

Filters required by the brief: court, date, and provision. Apply them to the
merged list so a filter cannot hide a hit from only one retriever.

Citation rule: if ``citation_available`` is false, the UI may show the passage
but must not present a paragraph number as a pinpoint citation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

JUDGMENT_CHUNKS = Path("data/processed/judgment_chunks.jsonl")
LEGISLATION_CHUNKS = Path("data/processed/legislation_chunks.jsonl")


def load_chunks(
    judgment_path: Path = JUDGMENT_CHUNKS,
    legislation_path: Path = LEGISLATION_CHUNKS,
) -> list[dict[str, Any]]:
    """Read the processed jsonl files. Do not go back to the raw corpus here."""
    raise NotImplementedError


def reciprocal_rank_fusion(
    bm25_hits: list[tuple[str, float]],
    vector_hits: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge the two ranked lists.

    Reciprocal rank fusion is a reasonable default: a chunk that ranked well
    on either list rises. You do not need to compare raw BM25 scores with
    cosine scores.
    """
    raise NotImplementedError


def apply_filters(
    chunks: list[dict[str, Any]],
    *,
    court: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    provision: str | None = None,
) -> list[dict[str, Any]]:
    """Keep chunks that match the selected court, date range, and provision."""
    raise NotImplementedError


def search(
    query: str,
    *,
    court: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    provision: str | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Run BM25 and vector search in parallel, merge, filter, and return chunks.

    Each result should still carry the original metadata: citation, court,
    date, paragraph_numbers, provision_id, url, and citation_available.
    """
    raise NotImplementedError
