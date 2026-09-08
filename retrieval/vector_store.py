"""Store passage embeddings and look up the nearest chunks.

FAISS or Chroma are both fine. Pick one and keep the functions below as the
public interface so the rest of the app does not care which library it is.

Build the index on a team machine and save it to disk. The demo should load
that saved index. It should not embed the full corpus when the client opens
the page.

Install later: faiss-cpu  (or chromadb)
"""

from __future__ import annotations

from pathlib import Path

INDEX_DIR = Path("data/indexes")


def build_index(embeddings, chunk_ids: list[str]):
    """Create a vector index from passage embeddings.

    ``chunk_ids`` must stay in the same order as ``embeddings``.
    """
    raise NotImplementedError


def save_index(index, path: Path = INDEX_DIR / "vectors") -> None:
    """Write the index to disk so the demo can start without rebuilding."""
    raise NotImplementedError


def load_index(path: Path = INDEX_DIR / "vectors"):
    """Load a previously saved index."""
    raise NotImplementedError


def search_vectors(index, query_embedding, top_k: int = 20) -> list[tuple[str, float]]:
    """Return ``(chunk_id, score)`` pairs for the nearest vectors."""
    raise NotImplementedError
