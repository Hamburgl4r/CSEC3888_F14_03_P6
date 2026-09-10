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
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import json

from bm25 import build_index as build_bm25_index
from bm25 import search_bm25
from embed import embed_query, load_encoder
from vector_store import load_index as load_vector_index
from vector_store import search_vectors

JUDGMENT_CHUNKS = Path("data/processed/judgment_chunks.jsonl")
LEGISLATION_CHUNKS = Path("data/processed/legislation_chunks.jsonl")


@lru_cache(maxsize=1)
def _load_search_resources():
    """Load chunks and indexes once, then reuse them for later searches."""
    chunks = load_chunks()
    chunks_by_id: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("every chunk must have a non-empty string chunk_id")

        if chunk_id in chunks_by_id:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")

        chunks_by_id[chunk_id] = chunk

    chunk_ids = list(chunks_by_id)
    texts = [str(chunks_by_id[chunk_id].get("text", "")) for chunk_id in chunk_ids]

    return (
        chunks_by_id,
        build_bm25_index(texts, chunk_ids),
        load_vector_index(),
        load_encoder(),
    )


def load_chunks(
    judgment_path: Path = JUDGMENT_CHUNKS,
    legislation_path: Path = LEGISLATION_CHUNKS,
) -> list[dict[str, Any]]:
    """Read the processed jsonl files. Do not go back to the raw corpus here."""
    chunks = []

    for path in (judgment_path, legislation_path):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} on line {line_number}"
                    ) from error

                chunks.append(chunk)

    return chunks


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
    if k < 0:
        raise ValueError("k must not be negative")

    fused_scores: dict[str, float] = {}

    for ranked_hits in (bm25_hits, vector_hits):
        seen: set[str] = set()

        for rank, (chunk_id, _) in enumerate(ranked_hits, start=1):
            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused_scores.items(),
        key=lambda hit: hit[1],
        reverse=True,
    )


def apply_filters(
    chunks: list[dict[str, Any]],
    *,
    court: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    provision: str | None = None,
) -> list[dict[str, Any]]:
    """Keep chunks that match the selected court, date range, and provision."""
    start_date = date.fromisoformat(date_from) if date_from else None
    end_date = date.fromisoformat(date_to) if date_to else None

    if start_date and end_date and start_date > end_date:
        raise ValueError("date_from must not be later than date_to")

    filtered_chunks = []

    for chunk in chunks:
        if court is not None:
            chunk_court = chunk.get("court")

            if (
                not isinstance(chunk_court, str)
                or chunk_court.casefold() != court.casefold()
            ):
                continue

        if start_date is not None or end_date is not None:
            chunk_date_value = chunk.get("date")

            if not chunk_date_value:
                continue

            try:
                chunk_date = date.fromisoformat(chunk_date_value)
            except (TypeError, ValueError):
                continue

            if start_date is not None and chunk_date < start_date:
                continue

            if end_date is not None and chunk_date > end_date:
                continue

        if provision is not None:
            chunk_provisions = chunk.get("provision_id")

            if isinstance(chunk_provisions, str):
                chunk_provisions = [chunk_provisions]
            elif not isinstance(chunk_provisions, list):
                chunk_provisions = []

            if not any(
                str(item).casefold() == provision.casefold()
                for item in chunk_provisions
            ):
                continue

        filtered_chunks.append(chunk)

    return filtered_chunks


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
    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        return []

    chunks_by_id, bm25_index, vector_index, encoder = _load_search_resources()

    # Retrieve extra candidates because the selected filters may remove hits.
    candidate_count = min(len(chunks_by_id), max(top_k * 5, 50))

    def run_vector_search() -> list[tuple[str, float]]:
        query_vector = embed_query(query.strip(), encoder)
        return search_vectors(vector_index, query_vector, candidate_count)

    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(
            search_bm25,
            bm25_index,
            query.strip(),
            candidate_count,
        )
        vector_future = executor.submit(run_vector_search)

        bm25_hits = bm25_future.result()
        vector_hits = vector_future.result()

    fused_hits = reciprocal_rank_fusion(bm25_hits, vector_hits)

    ranked_chunks = []

    for chunk_id, fusion_score in fused_hits:
        chunk = chunks_by_id.get(chunk_id)

        # Ignore an index entry if its source chunk no longer exists.
        if chunk is None:
            continue

        result = chunk.copy()
        result["score"] = fusion_score
        ranked_chunks.append(result)

    filtered_chunks = apply_filters(
        ranked_chunks,
        court=court,
        date_from=date_from,
        date_to=date_to,
        provision=provision,
    )

    return filtered_chunks[:top_k]
