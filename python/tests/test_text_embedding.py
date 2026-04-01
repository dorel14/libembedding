"""Tests for TextEmbedding."""

import numpy as np
import pytest


def test_list_text_models():
    from libembedding import list_text_models

    models = list_text_models()
    assert len(models) > 0
    assert any("bge-small" in m.model_name for m in models)
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_text_embedding_basic():
    from libembedding import TextEmbedding

    with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
        assert model.dim == 384
        result = model.embed(["Hello world", "How are you?"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)
        assert result.dtype == np.float32
        # Embeddings should be L2-normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_text_embedding_empty():
    from libembedding import TextEmbedding

    model = TextEmbedding()
    result = model.embed([])
    assert result.shape == (0, model.dim)
    model.close()


def test_text_embedding_cosine_similarity():
    from libembedding import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    result = model.embed([
        "The cat sat on the mat",
        "A kitten was sitting on a rug",
        "Quantum physics is complex",
    ])
    # Similar texts should have higher similarity
    sim_similar = np.dot(result[0], result[1])
    sim_different = np.dot(result[0], result[2])
    assert sim_similar > sim_different
    model.close()
