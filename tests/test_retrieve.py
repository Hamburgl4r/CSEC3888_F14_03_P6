import pytest

from retrieval import retrieve


CHUNKS = {
    "chunk_1": {
        "chunk_id": "chunk_1",
        "text": "The defendant owed the plaintiff a duty of care.",
        "court": "NSWSC",
        "date": "2020-01-10",
        "provision_id": [],
        "citation": "Smith v Jones [2020] NSWSC 1",
    },
    "chunk_2": {
        "chunk_id": "chunk_2",
        "text": "Section 5D concerns causation.",
        "court": "NSWCA",
        "date": "2021-05-20",
        "provision_id": ["5D"],
        "citation": "Brown v State [2021] NSWCA 10",
    },
    "chunk_3": {
        "chunk_id": "chunk_3",
        "text": "The court considered damages.",
        "court": "NSWDC",
        "date": "2018-03-15",
        "provision_id": [],
        "citation": "Green v Council [2018] NSWDC 5",
    },
}


def configure_fake_resources(monkeypatch):
    """Replace the real files, model and indexes with small test objects."""

    monkeypatch.setattr(
        retrieve,
        "_load_search_resources",
        lambda: (
            CHUNKS,
            "fake_bm25_index",
            "fake_vector_index",
            "fake_encoder",
        ),
    )

    monkeypatch.setattr(
        retrieve,
        "embed_query",
        lambda query, encoder: "fake_query_vector",
    )


def test_reciprocal_rank_fusion():
    bm25_hits = [
        ("chunk_1", 5.0),
        ("chunk_2", 3.0),
    ]

    vector_hits = [
        ("chunk_2", 0.9),
        ("chunk_3", 0.8),
    ]

    results = retrieve.reciprocal_rank_fusion(
        bm25_hits,
        vector_hits,
    )

    # chunk_2 appears in both result lists, so it should rank first.
    assert results[0][0] == "chunk_2"


def test_filter_by_court():
    results = retrieve.apply_filters(
        list(CHUNKS.values()),
        court="NSWCA",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_2"


def test_filter_by_date():
    results = retrieve.apply_filters(
        list(CHUNKS.values()),
        date_from="2020-01-01",
        date_to="2022-01-01",
    )

    assert [result["chunk_id"] for result in results] == [
        "chunk_1",
        "chunk_2",
    ]


def test_filter_by_provision():
    results = retrieve.apply_filters(
        list(CHUNKS.values()),
        provision="5d",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_2"


def test_hybrid_search_returns_best_chunk(monkeypatch):
    configure_fake_resources(monkeypatch)

    monkeypatch.setattr(
        retrieve,
        "search_bm25",
        lambda index, query, top_k: [
            ("chunk_2", 5.0),
            ("chunk_1", 3.0),
        ],
    )

    monkeypatch.setattr(
        retrieve,
        "search_vectors",
        lambda index, query_vector, top_k: [
            ("chunk_2", 0.95),
            ("chunk_3", 0.80),
        ],
    )

    results = retrieve.search(
        "section 5D causation",
        top_k=3,
    )

    assert results[0]["chunk_id"] == "chunk_2"
    assert "score" in results[0]


def test_search_applies_court_filter(monkeypatch):
    configure_fake_resources(monkeypatch)

    monkeypatch.setattr(
        retrieve,
        "search_bm25",
        lambda index, query, top_k: [
            ("chunk_1", 5.0),
            ("chunk_2", 4.0),
            ("chunk_3", 3.0),
        ],
    )

    monkeypatch.setattr(
        retrieve,
        "search_vectors",
        lambda index, query_vector, top_k: [
            ("chunk_2", 0.95),
            ("chunk_1", 0.85),
            ("chunk_3", 0.75),
        ],
    )

    results = retrieve.search(
        "negligence",
        court="NSWCA",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_2"


def test_empty_query_raises_error():
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieve.search("   ")


def test_non_positive_top_k_returns_empty_list():
    assert retrieve.search("causation", top_k=0) == []


def test_invalid_date_range_raises_error():
    with pytest.raises(
        ValueError,
        match="date_from must not be later than date_to",
    ):
        retrieve.apply_filters(
            list(CHUNKS.values()),
            date_from="2022-01-01",
            date_to="2020-01-01",
        )