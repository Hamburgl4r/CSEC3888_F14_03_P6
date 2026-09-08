from retrieval.bm25 import build_index, search_bm25


texts = [
    "The defendant owed the plaintiff a duty of care.",
    "Section 5D of the Civil Liability Act concerns causation.",
    "The Court of Appeal considered damages.",
    "Negligence requires consideration of foreseeable risk."
]

chunk_ids = [
    "chunk_1",
    "chunk_2",
    "chunk_3",
    "chunk_4"
]


def test_section_5d():
    index = build_index(texts, chunk_ids)

    results = search_bm25(index, "section 5D", top_k=3)

    assert results[0][0] == "chunk_2"


def test_duty_of_care():
    index = build_index(texts, chunk_ids)

    results = search_bm25(index, "duty of care", top_k=3)

    assert results[0][0] == "chunk_1"


def test_foreseeable_risk():
    index = build_index(texts, chunk_ids)

    results = search_bm25(index, "foreseeable risk", top_k=3)

    assert results[0][0] == "chunk_4"


def test_court_of_appeal():
    index = build_index(texts, chunk_ids)

    results = search_bm25(index, "Court of Appeal", top_k=3)

    assert results[0][0] == "chunk_3"