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
from .types import ModelDesc, Stats, TuningResult, ModelSelectionResult, UnifiedTuningResult
from .exceptions import ModelNotFoundError


def _sample_corpus(texts: list[str], max_size: int = 100) -> list[str]:
    """Sample a representative subset of texts for autotune benchmarking.

    Uses stratified sampling by text length to ensure the sample includes
    a mix of short, medium, and long texts. This gives stable benchmark
    results without processing the entire corpus.

    Args:
        texts: Full corpus (can be millions of texts)
        max_size: Maximum number of texts to sample (default 100)

    Returns:
        Sampled texts representative of the corpus distribution
    """
    n = len(texts)
    if n <= max_size:
        return texts

    # Stratified sampling: divide into buckets by length
    # This ensures we get a representative mix
    n_buckets = 10
    buckets: list[list[str]] = [[] for _ in range(n_buckets)]

    for text in texts:
        # Use word count as proxy for token count
        word_count = len(text.split())
        # Log-scale bucketing: more granular for short texts
        import math
        if word_count <= 0:
            bucket_idx = 0
        else:
            log_count = math.log10(word_count)
            bucket_idx = min(int(log_count * 2.5), n_buckets - 1)
        buckets[bucket_idx].append(text)

    # Sample proportionally from each bucket
    sampled = []
    per_bucket = max_size // n_buckets

    for bucket in buckets:
        if not bucket:
            continue
        # Sample from this bucket
        if len(bucket) <= per_bucket:
            sampled.extend(bucket)
        else:
            # Evenly spaced sampling
            step = len(bucket) / per_bucket
            for i in range(per_bucket):
                idx = int(i * step)
                sampled.append(bucket[idx])

    # Fill remaining slots if needed
    if len(sampled) < max_size:
        import random
        remaining = [t for t in texts if t not in sampled]
        if remaining:
            sampled.extend(random.sample(remaining, min(max_size - len(sampled), len(remaining))))

    return sampled[:max_size]


def autotune(
    model_name: str = "BAAI/bge-small-en-v1.5",
    *,
    full: bool = False,
) -> TuningResult:
    """Run auto-tuning to find optimal configuration for a model.

    Args:
        model_name: HuggingFace model code (e.g. "Qdrant/all-MiniLM-L6-v2-onnx")
        full: If True, run exhaustive tuning (30-120s). Otherwise quick (5-15s).

    Returns:
        TuningResult with optimal workers, threads, batch_size.

    Example:
        >>> result = autotune("Qdrant/all-MiniLM-L6-v2-onnx")
        >>> print(f"Optimal: {result.workers} workers, {result.threads} threads")
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_tuning_result_t *")

    # Resolve model code to get the correct code
    model_idx = resolve_text_model(model_name)
    if model_idx >= 0:
        # Use the model_code from registry
        info = ffi.new("lembed_model_info_t *")
        lib.lembed_get_text_model_info(model_idx, info)
        code = ffi.string(info.model_code).decode("utf-8")
    else:
        code = model_name

    check_status(lib.lembed_autotune(code.encode("utf-8"), mode, result))

    return TuningResult(
        workers=result.workers,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
    )


def auto_select_model(
    use_case: str = "balanced",
) -> "ModelSelectionResult":
    """Automatically select the best model and configuration for your hardware.

    Benchmarks multiple models and configurations to find the optimal
    combination for your specific use case.

    Results are cached - subsequent calls return instantly.

    Args:
        use_case: One of "speed", "quality", or "balanced" (default).
            - "speed": Prioritize throughput (e.g. real-time inference)
            - "quality": Prioritize embedding quality (e.g. semantic search)
            - "balanced": Good balance of speed and quality

    Returns:
        ModelSelectionResult with optimal model and configuration.

    Example:
        >>> result = auto_select_model("balanced")
        >>> print(f"Best model: {result.model_name}")
        >>> print(f"Config: {result.workers} workers, {result.threads} threads")
        >>> print(f"Throughput: {result.throughput_docs_sec:.0f} docs/s")
    """
    result = ffi.new("lembed_model_selection_t *")
    check_status(lib.lembed_auto_select_model(use_case.encode("utf-8"), result))

    return ModelSelectionResult(
        model_code=ffi.string(result.model_code).decode("utf-8"),
        model_name=ffi.string(result.model_name).decode("utf-8"),
        dim=result.dim,
        workers=result.workers,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
        score=result.score,
    )


def clear_autotune_cache(model_name: str = None) -> None:
    """Clear autotune cache for a model (or all models if None).

    Args:
        model_name: Model code to clear, or None to clear all cache.
    """
    lib.lembed_autotune_clear_cache(
        model_name.encode("utf-8") if model_name else ffi.NULL
    )


# Task types for unified auto-tuner
_TASK_EMBEDDING = 0
_TASK_RERANKING = 1
_TASK_IMAGE = 2
_TASK_SPARSE = 3


def autotune_unified(
    task: str = "embedding",
    model_name: str = None,
    *,
    full: bool = False,
) -> "UnifiedTuningResult":
    """Unified auto-tune entry point for all task types.

    Args:
        task: "embedding", "reranking", "image", or "sparse"
        model_name: Model name (default depends on task)
        full: If True, run FULL mode (30-120s), else QUICK (5-15s)

    Returns:
        UnifiedTuningResult with optimal configuration.

    Example:
        >>> result = autotune_unified("reranking", "jinaai/jina-reranker-v1-turbo-en-quantized")
        >>> print(f"Config: {result.threads} threads, batch={result.batch_size}")
    """
    task_map = {
        "embedding": _TASK_EMBEDDING,
        "reranking": _TASK_RERANKING,
        "image": _TASK_IMAGE,
        "sparse": _TASK_SPARSE,
    }
    if task not in task_map:
        raise ValueError(f"Unknown task '{task}'. Use: {list(task_map.keys())}")

    if model_name is None:
        # Default models per task
        defaults = {
            "embedding": "BAAI/bge-small-en-v1.5",
            "reranking": "jinaai/jina-reranker-v1-turbo-en-quantized",
            "image": None,
            "sparse": None,
        }
        model_name = defaults.get(task)
        if model_name is None:
            raise ValueError(f"No default model for task '{task}'. Please specify model_name.")

    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_unified_tuning_result_t *")

    # Resolve model name to code for embedding
    resolved_name = model_name
    if task == "embedding":
        from .models import resolve_text_model
        try:
            idx = resolve_text_model(model_name)
            info = ffi.new("lembed_model_info_t *")
            lib.lembed_get_text_model_info(idx, info)
            resolved_name = ffi.string(info.model_code).decode("utf-8")
        except Exception:
            pass  # Use original name

    check_status(lib.lembed_autotune_unified(task_map[task], resolved_name.encode("utf-8"), mode, result))

    return UnifiedTuningResult(
        task=task,
        threads=result.threads,
        batch_size=result.batch_size,
        workers=result.workers,
        max_tokens=result.max_tokens,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        memory_mb=result.memory_mb,
    )


class TextEmbedding:
    """Generate dense vector embeddings from text.

    Args:
        model_name: HuggingFace model name (e.g. "BAAI/bge-small-en-v1.5") or a
            path to a local directory containing model.onnx + tokenizer.json.
        provider: Execution provider ("cpu", "cuda", "coreml", "directml", "tensorrt").
        device_id: Device index for GPU providers.
        cache_dir: Model cache directory (None = default).
        max_length: Max token length (0 = model default).
        threads: Number of threads per worker (0 = auto with autotune).
        batch_size: Internal batch size for embedding (default 256).
        offline: If True, use cached models only; never download (default False).
        show_download_progress: Show download progress bar.
        dim: Embedding dimension for local models without config.json (0 = auto).
        pooling: Pooling strategy for local models ("cls" or "mean").
        autotune: If True, auto-tune threads/batch_size for best performance.
            Results are cached per machine/model so subsequent calls are instant.
        autotune_texts: Optional list of texts to use for autotune benchmark.
            If provided, autotune will benchmark using these texts instead of
            synthetic corpus. This gives more accurate tuning for your specific
            use case (e.g. if you process long documents vs short queries).
        autotune_max_samples: Maximum number of texts to sample from your corpus
            for benchmarking. Default 100. Only used when autotune_texts is provided.
            If your corpus has fewer texts, all are used.
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
        autotune: bool = False,
        autotune_texts: list[str] = None,
        autotune_max_samples: int = 100,
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        # Auto-tune if requested and threads not explicitly set
        if autotune and threads == 0:
            tuned = self._do_autotune(model_name, provider,
                                      texts=autotune_texts,
                                      max_sample_size=autotune_max_samples)
            threads = tuned.threads
            batch_size = tuned.batch_size  # always use autotuned batch_size
            if batch_size > 0:
                print(f"Autotune: threads={threads}, batch_size={batch_size}")

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
        self._threads = threads
        self._model_name = model_name

    @staticmethod
    def _do_autotune(model_name: str, provider: str = "cpu",
                     texts: list[str] = None, max_sample_size: int = 100) -> TuningResult:
        """Run autotune with cache lookup. Returns optimal config.

        Args:
            model_name: Model code or name
            provider: Execution provider
            texts: Optional custom corpus for benchmarking. If None, uses
                internal synthetic corpus. Providing your own texts gives
                more accurate tuning for your specific use case.
            max_sample_size: Maximum number of texts to sample from user's
                corpus for benchmarking. Default 100 is sufficient for
                stable results. Only used when texts is provided.
        """
        # Resolve model code
        model_idx = resolve_text_model(model_name)
        if model_idx >= 0:
            info_ptr = ffi.new("lembed_model_info_t *")
            lib.lembed_get_text_model_info(model_idx, info_ptr)
            code = ffi.string(info_ptr.model_code).decode("utf-8")
        else:
            code = model_name

        if texts:
            # Sample if corpus is too large
            sampled = _sample_corpus(texts, max_sample_size)
            n = len(sampled)
            result = ffi.new("lembed_tuning_result_t *")
            c_texts = ffi.new("char*[]", n)
            c_strs = []
            for i, t in enumerate(sampled):
                c_strs.append(ffi.new("char[]", t.encode("utf-8")))
                c_texts[i] = c_strs[i]
            check_status(lib.lembed_autotune_custom(
                code.encode("utf-8"), c_texts, n,
                lib.LEMBED_AUTOTUNE_QUICK, result))
        else:
            # Use internal synthetic corpus
            result = ffi.new("lembed_tuning_result_t *")
            check_status(lib.lembed_autotune(
                code.encode("utf-8"), lib.LEMBED_AUTOTUNE_QUICK, result))

        return TuningResult(
            workers=result.workers,
            threads=result.threads,
            batch_size=result.batch_size,
            throughput_docs_sec=result.throughput_docs_sec,
            latency_ms=result.latency_ms,
            memory_mb=result.memory_mb,
        )

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
        workers: Number of worker sessions (0 = auto with autotune).
        threads_per_worker: Threads per worker (default 1).
        batch_size: Internal batch size per worker.
        provider: Same as TextEmbedding.
        offline: Same as TextEmbedding.
        show_download_progress: Same as TextEmbedding.
        autotune: If True, auto-tune workers/threads/batch for best performance.
            Results are cached per machine/model so subsequent calls are instant.
        autotune_texts: Optional list of texts to use for autotune benchmark.
            If provided, autotune will benchmark using these texts instead of
            synthetic corpus. Recommended for production use.

    Example:
        >>> # Auto-tune: finds optimal config automatically
        >>> pool = TextEmbeddingPool("sentence-transformers/all-MiniLM-L6-v2", autotune=True)
        >>> embeddings = pool.embed(texts)
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
        autotune: bool = False,
        autotune_texts: list[str] = None,
        autotune_max_samples: int = 100,
    ):
        import os

        # Auto-tune if requested
        if autotune:
            tuned = TextEmbedding._do_autotune(model_name, provider,
                                               texts=autotune_texts,
                                               max_sample_size=autotune_max_samples)
            n_workers = tuned.workers
            threads_per_worker = tuned.threads
            batch_size = tuned.batch_size
            print(f"Autotune: {n_workers} workers × {threads_per_worker} threads, batch={batch_size}")
        else:
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
