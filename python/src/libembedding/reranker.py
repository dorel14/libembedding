"""High-level document reranker API."""

from __future__ import annotations

import warnings

from ._binding import ffi, lib
from ._status import check_status
from .models import (
    resolve_reranker_model,
    list_reranker_models,
    _PROVIDER_MAP,
    _desc_from_c,
    _is_local_path,
)
from .types import RerankResult, ModelDesc, Stats
from .exceptions import ModelNotFoundError


class Reranker:
    """Score and sort documents by relevance to a query.

    Args:
        model_name: Model name (e.g. "BAAI/bge-reranker-base") or local
            directory path.
        provider: Execution provider.
        cache_dir: Model cache directory.
        max_length: Max token length (0 = model default).
        threads: Number of threads (0 = auto).
        batch_size: Internal batch size (default 256).
        offline: If True, use cached models only (default False).
        show_download_progress: Show download progress bar.
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        threads: int = 0,
        batch_size: int = 256,
        offline: bool = False,
        show_download_progress: bool = True,
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        opts = lib.lembed_reranker_options_default()
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

        ctx_ptr = ffi.new("lembed_reranker_t **")

        try:
            model_idx = resolve_reranker_model(model_name)
            opts.model = model_idx
            check_status(lib.lembed_reranker_create(ffi.addressof(opts), ctx_ptr))
        except ModelNotFoundError:
            if _is_local_path(model_name):
                check_status(lib.lembed_reranker_create_from_path(
                    model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))
            else:
                raise

        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_reranker_free)
        self._batch_size = batch_size

    @staticmethod
    def list_supported_models():
        """Return a list of all supported reranker models."""
        return list_reranker_models()

    def rerank(
        self, query: str, documents: list[str], *, batch_size: int | None = None
    ) -> list[RerankResult]:
        """Score and sort documents by relevance.

        Args:
            query: The query text.
            documents: List of document texts.
            batch_size: Internal batch size (None = use constructor default).

        Returns:
            List of RerankResult sorted by score (descending).
        """
        n = len(documents)
        if n == 0:
            return []

        bs = 0 if batch_size is None else batch_size

        c_query = ffi.new("char[]", query.encode("utf-8"))
        encoded = [d.encode("utf-8") for d in documents]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_docs = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_rerank_results_t *")
        check_status(
            lib.lembed_reranker_rerank(self._ctx, c_query, c_docs, n, bs, result)
        )

        try:
            return [
                RerankResult(index=result.items[i].index, score=result.items[i].score)
                for i in range(result.count)
            ]
        finally:
            lib.lembed_rerank_results_free(result)

    def info(self) -> ModelDesc:
        """Return runtime model descriptor."""
        desc_ptr = lib.lembed_reranker_desc(self._ctx)
        return _desc_from_c(desc_ptr)

    @property
    def name(self) -> str:
        """Model name or local path."""
        name_ptr = lib.lembed_reranker_model_name(self._ctx)
        return ffi.string(name_ptr).decode("utf-8", errors="replace") if name_ptr else ""

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_t *")
        lib.lembed_reranker_stats(self._ctx, s)
        return Stats(
            texts_embedded=s.texts_embedded,
            batches_run=s.batches_run,
            avg_latency_ms=s.avg_latency_ms,
        )

    def close(self) -> None:
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
