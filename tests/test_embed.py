import numpy as np
from unittest.mock import patch

from retrieval.embed import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL_NAME,
    embed_passages,
    embed_query,
    load_encoder,
)

texts = [
    "The defendant owed the plaintiff a duty of care.",
    "Section 5D of the Civil Liability Act concerns causation.",
    "The Court of Appeal considered damages.",
    "Negligence requires consideration of foreseeable risk.",
]

class SimpleEncoder:
    def __init__(self):
        self.last_input = None
        self.last_normalize = None

    def encode(self, value, normalize_embeddings=False):
        self.last_input = value
        self.last_normalize = normalize_embeddings

        if isinstance(value, list):
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )

        return np.array(
            [0.5, 0.5, 0.5],
            dtype=np.float32,
        )


def test_load_default_encoder():
    with patch("retrieval.embed.SentenceTransformer") as mock_transformer:
        simple_encoder = object()
        mock_transformer.return_value = simple_encoder
        encoder = load_encoder()
        mock_transformer.assert_called_once_with(DEFAULT_MODEL_NAME)
        assert encoder is simple_encoder

def test_load_custom_encoder():
    with patch("retrieval.embed.SentenceTransformer") as mock_transformer:
        simple_encoder = object()
        mock_transformer.return_value = simple_encoder

        encoder = load_encoder("BAAI/bge-base-en-v1.5")
        mock_transformer.assert_called_once_with(
            "BAAI/bge-base-en-v1.5"
        )
        assert encoder is simple_encoder


def test_embed_query_adds_bge_prefix():
    encoder = SimpleEncoder()
    embed_query(
        "Section 5D concerns causation.",
        encoder,
    )
    assert encoder.last_input == (
        BGE_QUERY_PREFIX
        + "Section 5D concerns causation."
    )


def test_embed_query_normalizes_embedding():
    encoder = SimpleEncoder()
    embed_query(
        "Negligence and foreseeable risk",
        encoder,
    )
    assert encoder.last_normalize is True


def test_embed_query_returns_embedding():
    encoder = SimpleEncoder()
    result = embed_query(
        "Civil Liability Act",
        encoder,
    )

    expected = np.array(
        [0.5, 0.5, 0.5],
        dtype=np.float32,
    )
    assert np.array_equal(result, expected)


def test_embed_query_returns_numpy_array():
    encoder = SimpleEncoder()
    result = embed_query(
        "Court of Appeal damages",
        encoder,
    )
    assert isinstance(result, np.ndarray)


def test_embed_passages_uses_original_texts():
    encoder = SimpleEncoder()
    embed_passages(
        texts,
        encoder,
    )
    assert encoder.last_input == texts


def test_embed_passages_does_not_add_query_prefix():
    encoder = SimpleEncoder()
    embed_passages(
        texts,
        encoder,
    )

    for text in encoder.last_input:
        assert not text.startswith(BGE_QUERY_PREFIX)


def test_embed_passages_normalizes_embeddings():
    encoder = SimpleEncoder()
    embed_passages(
        texts,
        encoder,
    )
    assert encoder.last_normalize is True


def test_embed_passages_returns_list():
    encoder = SimpleEncoder()
    result = embed_passages(
        texts,
        encoder,
    )
    assert isinstance(result, list)


def test_embed_passages_returns_expected_embeddings():
    encoder = SimpleEncoder()
    result = embed_passages(
        texts,
        encoder,
    )

    expected = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    assert result == expected


def test_embed_passages_returns_one_embedding_per_text():
    encoder = SimpleEncoder()
    result = embed_passages(
        texts,
        encoder,
    )
    assert len(result) == len(texts)


def test_embed_query_does_not_modify_original_text():
    encoder = SimpleEncoder()
    query = "Section 5D causation"
    original_query = query

    embed_query(
        query,
        encoder,
    )
    assert query == original_query


def test_embed_passages_does_not_modify_original_texts():
    encoder = SimpleEncoder()
    original_texts = texts.copy()

    embed_passages(
        texts,
        encoder,
    )
    assert texts == original_texts