from retrieval.bm25 import build_index, search_bm25


texts = [
    "The defendant owed the plaintiff a duty of care.",
    "Section 5D of the Civil Liability Act concerns causation.",
    "The Court of Appeal considered damages.",
    "Negligence requires consideration of foreseeable risk.",
]

chunk_ids = [
    "chunk_1",
    "chunk_2",
    "chunk_3",
    "chunk_4",
]


index = build_index(texts, chunk_ids)


queries = [
    "section 5D",
    "duty of care",
    "foreseeable risk",
    "Court of Appeal"
]

for query in queries:
    print(f"\nQuery: {query}")

    results = search_bm25(index, query, top_k=3)

    for chunk_id, score in results:
        print(chunk_id, score)