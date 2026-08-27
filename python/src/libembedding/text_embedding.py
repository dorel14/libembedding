"""High-level text embedding API with multi-worker support."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from .types import ModelDesc, Stats
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
        threads: Number of threads per worker (0 = auto).
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
        self._model_name = model_name

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

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_t *")
        lib.lembed_text_embedding_stats(self._ctx, s)
        return Stats(
            texts_embedded=s.texts_embedded,
            batches_run=s.batches_run,
            avg_latency_ms=s.avg_latency_ms,
        )

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

    def embed_stream(self, texts: list[str], *, batch_size: int | None = None):
        """Embed texts as a generator, yielding one embedding at a time.

        Processes in batches internally but yields each embedding individually,
        avoiding large memory allocation for large document sets.

        Args:
            texts: List of text strings to embed.
            batch_size: Internal batch size (None = use constructor default).

        Yields:
            numpy array of shape (dim,) with dtype float32.
        """
        n = len(texts)
        if n == 0:
            return

        bs = 0 if batch_size is None else batch_size
        actual_bs = bs if bs > 0 else self._batch_size
        if actual_bs <= 0:
            actual_bs = 32

        for i in range(0, n, actual_bs):
            batch = texts[i:i + actual_bs]
            embeddings = self.embed(batch, batch_size=actual_bs)
            for emb in embeddings:
                yield emb

    def close(self) -> None:
        """Release the underlying C resources."""
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"TextEmbedding(dim={self._dim})"


class TextEmbeddingPool:
    """Multi-worker text embedding pool for maximum throughput.

    Creates multiple TextEmbedding instances and distributes work across
    them using threads. This achieves request-level parallelism which is
    more efficient than ORT intra-op parallelism for small Transformers.

    Args:
        model_name: Same as TextEmbedding.
        workers: Number of worker sessions (0 = auto, based on CPU cores).
        threads_per_worker: Threads per worker (default 1).
        batch_size: Internal batch size per worker.
        provider: Same as TextEmbedding.
        offline: Same as TextEmbedding.
        show_download_progress: Same as TextEmbedding.

    Example:
        >>> pool = TextEmbeddingPool("sentence-transformers/all-MiniLM-L6-v2", workers=8)
        >>> embeddings = pool.embed(texts)  # ~8x faster than single session
        >>> pool.close()
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        workers: int = 0,
        threads_per_worker: int = 1,
        batch_size: int = 256,
        provider: str = "cpu",
        offline: bool = False,
        show_download_progress: bool = True,
        cache_dir: str | None = None,
        max_length: int = 0,
        dim: int = 0,
        pooling: str = "mean",
    ):
        import os

        # Auto-detect workers
        n_workers = workers
        if n_workers <= 0:
            n_workers = min(os.cpu_count() or 4, 8)

        self._n_workers = n_workers
        self._dim = 0

        # Create worker sessions
        self._workers: list[TextEmbedding] = []
        for i in range(n_workers):
            worker = TextEmbedding(
                model_name,
                provider=provider,
                threads=threads_per_worker,
                batch_size=batch_size,
                offline=offline,
                show_download_progress=(show_download_progress and i == 0),
                cache_dir=cache_dir,
                max_length=max_length,
                dim=dim,
                pooling=pooling,
            )
            self._workers.append(worker)

        self._dim = self._workers[0].dim

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def num_workers(self) -> int:
        """Number of worker sessions."""
        return self._n_workers

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using all workers.

        Distributes texts across workers in parallel for maximum throughput.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dim) with dtype float32.
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        # Split texts across workers
        per_worker = (n + self._n_workers - 1) // self._n_workers
        chunks = []
        for i in range(self._n_workers):
            start = i * per_worker
            end = min(start + per_worker, n)
            if start < end:
                chunks.append(texts[start:end])

        # Embed in parallel
        results: list[np.ndarray] = [None] * len(chunks)

        def embed_chunk(idx: int, chunk: list[str]) -> None:
            results[idx] = self._workers[idx].embed(chunk)

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(embed_chunk, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Concatenate results in order
        return np.concatenate(results, axis=0)

    def close(self) -> None:
        """Release all worker sessions."""
        for worker in self._workers:
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"TextEmbeddingPool(workers={self._n_workers}, dim={self._dim})"
