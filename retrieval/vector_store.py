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

from typing import NotRequired, TypedDict
from pathlib import Path

# Setup 
INDEX_DIR = Path("data/indexes")

class VectorIndex(TypedDict):
    faiss: faiss.Index
    chunk_ids: list[str]
    vectors: NotRequired[np.ndarray]



## Functions
def build_index(embeddings, chunk_ids: list[str]) -> VectorIndex: 
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
        index["faiss"],
        str(path.with_suffix(".faiss"))
    )

    # save chunk id mappings
    ids_path = path.with_name(path.name + "_ids.json")

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(index["chunk_ids"], f)

    # Keep a NumPy copy for the application runtime. On macOS, running
    # PyTorch inference and a large FAISS search in the same process can
    # crash because their native runtimes conflict.
    vectors = index["faiss"].reconstruct_n(0, index["faiss"].ntotal)
    np.save(path.with_suffix(".npy"), vectors)



def load_index(path: Path = INDEX_DIR / "vectors") -> VectorIndex:
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

    vectors_path = path.with_suffix(".npy")
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"{vectors_path} is missing; rebuild the search index"
        )

    return {
        "faiss": faiss_index,
        "chunk_ids": chunk_ids,
        "vectors": np.load(vectors_path, mmap_mode="r"),
    }


def search_vectors(index: VectorIndex, 
                   query_embedding: list[float] | np.ndarray, 
                   top_k: int = 20
                   ) -> list[tuple[str, float]]:
    """Return ``(chunk_id, score)`` pairs for the nearest vectors."""

    if top_k <= 0: return []

    if len(index["chunk_ids"]) == 0: return []

    # convert embeddings into FAISS safe format
    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    if query_vector.shape[1] != index["faiss"].d:
        raise ValueError (
            "query embedding dimmension does not math FAISS index"
        )

    k = min(top_k, len(index["chunk_ids"]))

    if "vectors" in index:
        scores = np.asarray(index["vectors"] @ query_vector[0])
        candidate_positions = np.argpartition(scores, -k)[-k:]
        positions = candidate_positions[
            np.argsort(scores[candidate_positions])[::-1]
        ]
        return [
            (index["chunk_ids"][int(position)], float(scores[position]))
            for position in positions
        ]

    # perform vector similarity search
    scores, positions = index["faiss"].search(
        query_vector,
        k,
    )
    results: list[tuple[str, float]] = []

    # map the FAISS positions into respective chunks
    for position, score in zip(positions[0], scores[0]):
        if position == -1: continue

        chunk_id = index["chunk_ids"][position]

        results.append(
            (chunk_id, float(score))
        )

    return results