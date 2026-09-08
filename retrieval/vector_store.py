"""Store passage embeddings and look up the nearest chunks.

FAISS or Chroma are both fine. Pick one and keep the functions below as the
public interface so the rest of the app does not care which library it is.

Build the index on a team machine and save it to disk. The demo should load
that saved index. It should not embed the full corpus when the client opens
the page.

Install later: faiss-cpu  (or chromadb)
"""
## Imports
import faiss
import numpy as np
import json

from __future__ import annotations
from pathlib import Path

INDEX_DIR = Path("data/indexes")

## Functions
def build_index(embeddings, chunk_ids: list[str]) -> dict: 
    """Create a FAISS index from passage embedings.

    ``chunk_ids`` must stay in the same order as ``embeddings``.
    """
    # default checks
    if len(embeddings) != len(chunk_ids):
        raise ValueError("embeddings and chunk_ids must have the same length")

    if len(embeddings) == 0:
        raise ValueError("cannot build an index with no embeddings")

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_ids must be unique")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    dimension = vectors.shape[1]

    # create faiss index and append vectors. 
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(vectors)

    return {
        "faiss": faiss_index,
        "chunk_ids": list(chunk_ids),
    }



def save_index(index, path: Path = INDEX_DIR / "vectors") -> None:
    """Write the index to disk so the demo can start without rebuilding."""

    # ensure dir exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # save index
    faiss.write_index(
        index[faiss],
        str(path.with_suffix(".faiss"))
    )

    # save chunk id mappings
    ids_path = path.with_name(path.name, + "_ids.json")

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(index["chunk_ids"], f)



def load_index(path: Path = INDEX_DIR / "vectors"):
    """Load a previously saved index."""
    path = Path(path)

    # read the saved iundex
    faiss_index = faiss.read_index(
        str(path.with_suffix(".faiss"))
    )

    # retireve chunk ids
    ids_path = path.with_name(path.name + "_ids.json")

    with open(ids_path, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    
    return {
        "faiss": faiss_index,
        "chunk_ids": chunk_ids,
    }


def search_vectors(index, query_embedding, top_k: int = 20) -> list[tuple[str, float]]:
    """Return ``(chunk_id, score)`` pairs for the nearest vectors."""
    raise NotImplementedError
