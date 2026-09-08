# Imports
import numpy as np
import pytest

from retrieval.vector_store import (
    build_index,
    save_index,
    load_index,
    search_vectors,
)


def test_build_index_creates_correct_size():
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
        "chunk_b",
    ]

    index = build_index(embeddings, chunk_ids)

    assert index["faiss"].ntotal == 2
    assert index["chunk_ids"] == chunk_ids


def test_build_index_rejects_mismatched_lengths():
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
    ]

    with pytest.raises(ValueError):
        build_index(embeddings, chunk_ids)


def test_build_index_rejects_empty_embeddings():
    with pytest.raises(ValueError):
        build_index([], [])


def test_build_index_rejects_duplicate_chunk_ids():
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
        "chunk_a",
    ]

    with pytest.raises(ValueError):
        build_index(embeddings, chunk_ids)


def test_search_vectors_returns_nearest_chunk():
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    chunk_ids = [
        "east",
        "north",
        "west",
    ]

    index = build_index(embeddings, chunk_ids)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = search_vectors(index, query, top_k=1)

    assert len(results) == 1
    assert results[0][0] == "east"
    assert results[0][1] == pytest.approx(1.0)


def test_search_vectors_returns_results_in_similarity_order():
    embeddings = [
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "best_match",
        "second_match",
        "third_match",
    ]

    index = build_index(embeddings, chunk_ids)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = search_vectors(index, query, top_k=3)

    assert [chunk_id for chunk_id, _ in results] == [
        "best_match",
        "second_match",
        "third_match",
    ]


def test_search_vectors_limits_top_k():
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
        "chunk_b",
    ]

    index = build_index(embeddings, chunk_ids)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = search_vectors(index, query, top_k=10)

    assert len(results) == 2


def test_search_vectors_returns_empty_for_non_positive_top_k():
    embeddings = [
        [1.0, 0.0],
    ]

    chunk_ids = [
        "chunk_a",
    ]

    index = build_index(embeddings, chunk_ids)

    query = np.array([1.0, 0.0], dtype=np.float32)

    assert search_vectors(index, query, top_k=0) == []
    assert search_vectors(index, query, top_k=-1) == []


def test_search_vectors_rejects_wrong_dimension():
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
        "chunk_b",
    ]

    index = build_index(embeddings, chunk_ids)

    wrong_dimension_query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        search_vectors(
            index,
            wrong_dimension_query,
        )


def test_save_and_load_index(tmp_path):
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
        "chunk_b",
    ]

    index = build_index(embeddings, chunk_ids)

    path = tmp_path / "test_vectors"

    save_index(index, path)

    loaded_index = load_index(path)

    assert loaded_index["faiss"].ntotal == 2
    assert loaded_index["chunk_ids"] == chunk_ids


def test_loaded_index_can_be_searched(tmp_path):
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunk_ids = [
        "chunk_a",
        "chunk_b",
    ]

    index = build_index(embeddings, chunk_ids)

    path = tmp_path / "test_vectors"

    save_index(index, path)

    loaded_index = load_index(path)

    query = np.array([0.0, 1.0], dtype=np.float32)

    results = search_vectors(
        loaded_index,
        query,
        top_k=1,
    )

    assert results[0][0] == "chunk_b"