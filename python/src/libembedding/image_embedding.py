"""High-level image embedding API."""

from __future__ import annotations

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .models import resolve_image_model, _PROVIDER_MAP


class ImageEmbedding:
    """Generate dense vector embeddings from images.

    Args:
        model_name: Model name (e.g. "Qdrant/clip-ViT-B-32-vision").
        provider: Execution provider.
        cache_dir: Model cache directory.
        num_threads: Number of threads (0 = auto).
        show_download_progress: Show download progress bar.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/clip-ViT-B-32-vision",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        num_threads: int = 0,
        show_download_progress: bool = True,
    ):
        opts = lib.lembed_image_options_default()
        opts.model = resolve_image_model(model_name)
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = ffi.new("char[]", cache_dir.encode()) if cache_dir else ffi.NULL
        opts.cache_dir = self._cache_dir_buf
        opts.num_threads = num_threads
        opts.show_download_progress = int(show_download_progress)

        ctx_ptr = ffi.new("lembed_image_embedding_t **")
        check_status(lib.lembed_image_embedding_create(ffi.addressof(opts), ctx_ptr))
        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_image_embedding_free)
        self._dim = lib.lembed_image_embedding_dim(self._ctx)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_files(self, paths: list[str], *, batch_size: int = 0) -> np.ndarray:
        """Embed images from file paths.

        Returns:
            numpy array of shape (len(paths), dim) with dtype float32.
        """
        n = len(paths)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        encoded = [p.encode("utf-8") for p in paths]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_paths = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_embeddings_t *")
        check_status(
            lib.lembed_image_embedding_embed_files(self._ctx, c_paths, n, batch_size, result)
        )

        try:
            buf = ffi.buffer(result.data, result.num_embeddings * result.dim * 4)
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            return arr.reshape(result.num_embeddings, result.dim)
        finally:
            lib.lembed_embeddings_free(result)

    def embed_bytes(self, images: list[bytes], *, batch_size: int = 0) -> np.ndarray:
        """Embed images from raw bytes (JPEG, PNG, etc.).

        Returns:
            numpy array of shape (len(images), dim) with dtype float32.
        """
        n = len(images)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        c_data = ffi.new("unsigned char*[]", n)
        c_sizes = ffi.new("int[]", n)
        kept = []
        for i, img in enumerate(images):
            buf = ffi.new("unsigned char[]", img)
            kept.append(buf)
            c_data[i] = buf
            c_sizes[i] = len(img)

        result = ffi.new("lembed_embeddings_t *")
        check_status(
            lib.lembed_image_embedding_embed_bytes(
                self._ctx, c_data, c_sizes, n, batch_size, result
            )
        )

        try:
            buf = ffi.buffer(result.data, result.num_embeddings * result.dim * 4)
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            return arr.reshape(result.num_embeddings, result.dim)
        finally:
            lib.lembed_embeddings_free(result)

    def close(self) -> None:
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
