"""Turn queries and chunks into vectors with a sentence transformer.

The September demo should use a small or base BGE model so a normal laptop
can load it. The large models are an upgrade path, not the starting point.

    BAAI/bge-small-en-v1.5
    BAAI/bge-base-en-v1.5

BGE expects an instruction prefix on the query only, not on the passages.
E5 models use different prefixes (``query:`` / ``passage:``). Match the
prefix to the model or search quality will drop.

Install later: sentence-transformers
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer

# Default for the client demo. Swap the string if the team later tries large.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_encoder(model_name: str = DEFAULT_MODEL_NAME):
    """Load the sentence transformer once and reuse it.

    Keep the returned object in memory. Do not reload the model on every query.
    """
    encoder = SentenceTransformer(model_name)
    return encoder


def embed_passages(texts: list[str], encoder) -> list:
    """Embed chunk text. Do not add the BGE query prefix here."""
    embeddings = encoder.encode (
        texts, 
        normalize_embeddings=True, # scales output vectors to unit length 1, allow fast comparison
    )
    return embeddings


def embed_query(text: str, encoder):
    """Embed a user query. Add the BGE query prefix before encoding."""
    raise NotImplementedError