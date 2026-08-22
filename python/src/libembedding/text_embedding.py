"""High-level text embedding API."""

from __future__ import annotations

import warnings

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .models import (
    resolve_text_model,
    list_text_models,
    _PROVIDER_MAP,
    _POOLING_ENUM,
    _desc_from_c,
    _is_local_path,
)
from .types import ModelDesc
from .exceptions import ModelNotFoundError


class TextEmbedding:
    """Generate dense vector embeddings from text.

    Args:
        model_name: HuggingFace model name (e.g. "BAAI/bge-small-en-v1.5") or a
            path to a local directory containing model.onnx + tokenizer.json.
        provider: Execution provider ("cpu", "cuda", "coreml", "directml", "tensorrt").
        device_id: Device index for GPU providers.
        cache_dir: Model cache directory (None = default).
        max_length: Max token length (0 = model default).
        threads: Number of threads (0 = auto).
        batch_size: Internal batch size for embedding (default 256).
        offline: If True, use cached models only; never download (default False).
        show_download_progress: Show download progress bar.
        dim: Embedding dimension for local models without config.json (0 = auto).
        pooling: Pooling strategy for local models ("cls" or "mean").
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        threads: int = 0,
        batch_size: int = 256,
        offline: bool = False,
        show_download_progress: bool = True,
        dim: int = 0,
        pooling: str = "mean",
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        opts = lib.lembed_text_options_default()
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = (
            ffi.new("char[]", cache_dir.encode("utf-8")) if cache_dir else ffi.NULL
        )
        opts.cache_dir = self._cache_dir_buf
        opts.max_length = max_length
        opts.num_threads = threads
        opts.batch_size = batch_size
        opts.offline = int(offline)
        opts.show_download_progress = int(show_download_progress)
        opts.dim = dim
        opts.pooling = _POOLING_ENUM.get(pooling.lower(), 1)  # default MEAN

        ctx_ptr = ffi.new("lembed_text_embedding_t **")

        try:
            model_idx = resolve_text_model(model_name)
            opts.model = model_idx
            check_status(lib.lembed_text_embedding_create(
                ffi.addressof(opts), ctx_ptr))
        except ModelNotFoundError:
            if _is_local_path(model_name):
                check_status(lib.lembed_text_embedding_create_from_path(
                    model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))
            else:
                raise

        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_text_embedding_free)
        self._dim = lib.lembed_text_embedding_dim(self._ctx)
        self._batch_size = batch_size

    @staticmethod
    def list_supported_models():
        """Return a list of all supported text embedding models."""
        return list_text_models()

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def batch_size(self) -> int:
        """Configured internal batch size."""
        return self._batch_size

    def info(self) -> ModelDesc:
        """Return runtime model descriptor."""
        desc_ptr = lib.lembed_text_embedding_desc(self._ctx)
        return _desc_from_c(desc_ptr)

    @property
    def name(self) -> str:
        """Model name or local path."""
        name_ptr = lib.lembed_text_embedding_model_name(self._ctx)
        return ffi.string(name_ptr).decode("utf-8", errors="replace") if name_ptr else ""

    def embed(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """Embed texts into dense vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Internal batch size (None = use constructor default).

        Returns:
            numpy array of shape (len(texts), dim) with dtype float32.
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        bs = 0 if batch_size is None else batch_size

        encoded = [t.encode("utf-8") for t in texts]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_texts = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_embeddings_t *")
        check_status(
            lib.lembed_text_embedding_embed(self._ctx, c_texts, n, bs, result)
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
