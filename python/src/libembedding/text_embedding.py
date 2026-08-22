"""High-level text embedding API."""

from __future__ import annotations

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .models import resolve_text_model, list_text_models, _PROVIDER_MAP


class TextEmbedding:
    """Generate dense vector embeddings from text.

    Args:
        model_name: HuggingFace model name or repo code (e.g. "BAAI/bge-small-en-v1.5").
        provider: Execution provider ("cpu", "cuda", "coreml", "directml", "tensorrt").
        device_id: Device index for GPU providers.
        cache_dir: Model cache directory (None = default).
        max_length: Max token length (0 = model default).
        num_threads: Number of threads (0 = auto).
        show_download_progress: Show download progress bar.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        num_threads: int = 0,
        show_download_progress: bool = True,
    ):
        opts = lib.lembed_text_options_default()
        opts.model = resolve_text_model(model_name)
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = ffi.new("char[]", cache_dir.encode()) if cache_dir else ffi.NULL
        opts.cache_dir = self._cache_dir_buf
        opts.max_length = max_length
        opts.num_threads = num_threads
        opts.show_download_progress = int(show_download_progress)

        ctx_ptr = ffi.new("lembed_text_embedding_t **")
        check_status(lib.lembed_text_embedding_create(ffi.addressof(opts), ctx_ptr))
        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_text_embedding_free)
        self._dim = lib.lembed_text_embedding_dim(self._ctx)

    @staticmethod
    def list_supported_models():
        """Return a list of all supported text embedding models."""
        return list_text_models()

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def embed(self, texts: list[str], *, batch_size: int = 0) -> np.ndarray:
        """Embed texts into dense vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Internal batch size (0 = default).

        Returns:
            numpy array of shape (len(texts), dim) with dtype float32.
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        encoded = [t.encode("utf-8") for t in texts]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_texts = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_embeddings_t *")
        check_status(
            lib.lembed_text_embedding_embed(self._ctx, c_texts, n, batch_size, result)
        )

        try:
            buf = ffi.buffer(result.data, result.num_embeddings * result.dim * 4)
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            return arr.reshape(result.num_embeddings, result.dim)
        finally:
            lib.lembed_embeddings_free(result)

    def close(self) -> None:
        """Release the underlying C resources."""
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"TextEmbedding(dim={self._dim})"
