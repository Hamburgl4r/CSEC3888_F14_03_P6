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

class TestEncoder:
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
        test_encoder = object()
        mock_transformer.return_value = test_encoder

        encoder = load_encoder()

        mock_transformer.assert_called_once_with(DEFAULT_MODEL_NAME)
        assert encoder is test_encoder