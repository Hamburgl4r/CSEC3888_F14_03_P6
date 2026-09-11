"""Build and save the FAISS vector index from processed chunks."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embed import embed_passages, load_encoder
from retrieval.retrieve import load_chunks
from retrieval.vector_store import build_index, save_index


def main() -> None:
    print("Loading processed chunks...")
    chunks = load_chunks()

    if not chunks:
        raise ValueError("No processed chunks were found")

    chunk_ids: list[str] = []
    texts: list[str] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every chunk must have a non-empty chunk_id")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {chunk_id!r} does not contain valid text"
            )

        chunk_ids.append(chunk_id)
        texts.append(text)

    print(f"Loaded {len(chunks)} chunks")

    print("Loading the sentence-transformer model...")
    encoder = load_encoder()
    print(f"Embedding device: {encoder.device}")

    print(f"Creating embeddings for {len(texts):,} passages...")
    embeddings = embed_passages(texts, encoder)

    print("Building the FAISS index...")
    vector_index = build_index(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
    )

    print("Saving the FAISS index...")
    save_index(vector_index)

    print("Index successfully created:")
    print("  data/indexes/vectors.faiss")
    print("  data/indexes/vectors_ids.json")
    print("  data/indexes/vectors.npy")


if __name__ == "__main__":
    main()