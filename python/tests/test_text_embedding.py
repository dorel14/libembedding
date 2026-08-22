"""Tests for TextEmbedding."""

import warnings

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


def test_text_embedding_threads_param():
    from libembedding import TextEmbedding

    with TextEmbedding("BAAI/bge-small-en-v1.5", threads=2) as model:
        info = model.info()
        assert info.num_threads == 2


def test_text_embedding_num_threads_deprecated():
    from libembedding import TextEmbedding

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TextEmbedding("BAAI/bge-small-en-v1.5", num_threads=2)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "num_threads" in str(w[0].message)


def test_text_embedding_batch_size():
    from libembedding import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=64)
    assert model.batch_size == 64
    info = model.info()
    assert info.batch_size == 64
    result = model.embed(["Hello", "World"], batch_size=1)
    assert result.shape == (2, 384)
    # Default batch_size (None) uses constructor default
    result2 = model.embed(["Hello", "World"])
    assert result2.shape == (2, 384)
    model.close()


def test_text_embedding_info():
    from libembedding import TextEmbedding

    with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
        info = model.info()
        assert info.dimension == 384
        assert info.dimension > 0
        assert info.max_length > 0
        assert info.batch_size > 0


def test_text_embedding_name():
    from libembedding import TextEmbedding

    with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
        assert "bge-small" in model.name


def test_text_embedding_offline_param():
    """Verify offline parameter is accepted and mapped to options."""
    from libembedding import TextEmbedding

    # offline=False (default) should work when model is cached or downloadable
    with TextEmbedding("BAAI/bge-small-en-v1.5", offline=False) as model:
        assert model.dim == 384
        info = model.info()
        assert info.batch_size == 256


def test_text_embedding_offline_missing_model():
    """offline=True with a model not in cache should raise an error."""
    from libembedding import TextEmbedding
    from libembedding.exceptions import LembedError

    # BGE large is not used by other tests, so likely not cached
    with pytest.raises(LembedError):
        TextEmbedding("BAAI/bge-large-en-v1.5", offline=True)
