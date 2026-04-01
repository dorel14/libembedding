"""High-level document reranker API."""

from __future__ import annotations

from ._binding import ffi, lib
from ._status import check_status
from .models import resolve_reranker_model, _PROVIDER_MAP
from .types import RerankResult


class Reranker:
    """Score and sort documents by relevance to a query.

    Args:
        model_name: Model name (e.g. "BAAI/bge-reranker-base").
        provider: Execution provider.
        cache_dir: Model cache directory.
        max_length: Max token length (0 = model default).
        num_threads: Number of threads (0 = auto).
        show_download_progress: Show download progress bar.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        num_threads: int = 0,
        show_download_progress: bool = True,
    ):
        opts = lib.lembed_reranker_options_default()
        opts.model = resolve_reranker_model(model_name)
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = ffi.new("char[]", cache_dir.encode()) if cache_dir else ffi.NULL
        opts.cache_dir = self._cache_dir_buf
        opts.max_length = max_length
        opts.num_threads = num_threads
        opts.show_download_progress = int(show_download_progress)

        ctx_ptr = ffi.new("lembed_reranker_t **")
        check_status(lib.lembed_reranker_create(ffi.addressof(opts), ctx_ptr))
        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_reranker_free)

    def rerank(
        self, query: str, documents: list[str], *, batch_size: int = 0
    ) -> list[RerankResult]:
        """Score and sort documents by relevance.

        Args:
            query: The query text.
            documents: List of document texts.
            batch_size: Internal batch size (0 = default).

        Returns:
            List of RerankResult sorted by score (descending).
        """
        n = len(documents)
        if n == 0:
            return []

        c_query = ffi.new("char[]", query.encode("utf-8"))
        encoded = [d.encode("utf-8") for d in documents]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_docs = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_rerank_results_t *")
        check_status(
            lib.lembed_reranker_rerank(self._ctx, c_query, c_docs, n, batch_size, result)
        )

        try:
            return [
                RerankResult(index=result.items[i].index, score=result.items[i].score)
                for i in range(result.count)
            ]
        finally:
            lib.lembed_rerank_results_free(result)

    def close(self) -> None:
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
