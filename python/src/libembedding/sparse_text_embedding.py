"""High-level sparse text embedding API."""

from __future__ import annotations

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .models import resolve_sparse_model, _PROVIDER_MAP
from .types import SparseEmbedding


class SparseTextEmbedding:
    """Generate sparse vector embeddings (term weights) from text.

    Args:
        model_name: HuggingFace model name (e.g. "prithvida/SPLADE_PP_en_v1").
        provider: Execution provider.
        cache_dir: Model cache directory.
        max_length: Max token length (0 = model default).
        num_threads: Number of threads (0 = auto).
        show_download_progress: Show download progress bar.
    """

    def __init__(
        self,
        model_name: str = "prithvida/SPLADE_PP_en_v1",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        num_threads: int = 0,
        show_download_progress: bool = True,
    ):
        opts = lib.lembed_sparse_options_default()
        opts.model = resolve_sparse_model(model_name)
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = ffi.new("char[]", cache_dir.encode()) if cache_dir else ffi.NULL
        opts.cache_dir = self._cache_dir_buf
        opts.max_length = max_length
        opts.num_threads = num_threads
        opts.show_download_progress = int(show_download_progress)

        ctx_ptr = ffi.new("lembed_sparse_embedding_ctx_t **")
        check_status(lib.lembed_sparse_text_embedding_create(ffi.addressof(opts), ctx_ptr))
        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_sparse_text_embedding_free)

    def embed(self, texts: list[str], *, batch_size: int = 0) -> list[SparseEmbedding]:
        """Embed texts into sparse vectors.

        Returns:
            List of SparseEmbedding with indices (int32) and values (float32).
        """
        n = len(texts)
        if n == 0:
            return []

        encoded = [t.encode("utf-8") for t in texts]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_texts = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_sparse_embeddings_t *")
        check_status(
            lib.lembed_sparse_text_embedding_embed(self._ctx, c_texts, n, batch_size, result)
        )

        try:
            embeddings = []
            for i in range(result.count):
                item = result.items[i]
                indices = np.frombuffer(
                    ffi.buffer(item.indices, item.length * 4), dtype=np.int32
                ).copy()
                values = np.frombuffer(
                    ffi.buffer(item.values, item.length * 4), dtype=np.float32
                ).copy()
                embeddings.append(SparseEmbedding(indices=indices, values=values))
            return embeddings
        finally:
            lib.lembed_sparse_embeddings_free(result)

    def close(self) -> None:
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
