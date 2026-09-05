---
nav_exclude: true
---

# Performance Tuning

This guide covers libembedding's advanced performance features: session pooling, auto-tuning, automatic model selection, llama.cpp/GGUF tuning, bucketing, LRU cache, and FAST/BALANCED/QUALITY modes.

## Table of Contents

1. [Session Pool (EmbeddingPool)](#session-pool-embeddingpool)
2. [Auto-Tuning](#auto-tuning)
3. [Automatic Model Selection](#automatic-model-selection)
4. [llama.cpp / GGUF Performance](#llamacpp--gguf-performance)
5. [Benchmarks](#benchmarks)
6. [Best Practices](#best-practices)

---

## Session Pool (EmbeddingPool)

For small Transformer architectures (MiniLM, BGE-small, E5-small), **inter-session** parallelism (multiple independent ONNX sessions) is more efficient than **intra-session** parallelism (ORT threading).

### When to use it

| Scenario | Recommendation |
|----------|----------------|
| < 100 embeddings | Single `TextEmbedding` is sufficient |
| > 100 embeddings | `TextEmbeddingPool` recommended |
| Production / high throughput | `TextEmbeddingPool` + `autotune=True` |

### Usage

```python
from libembedding import TextEmbeddingPool

# Pool with 8 workers (independent ONNX sessions)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    workers=8,               # number of parallel sessions
    threads_per_worker=1,    # 1 thread per session (avoids contention)
    batch_size=64,
    offline=True,
)

embeddings = pool.embed(texts)
pool.close()
```

### Performance gain

| Configuration | Docs/s (short texts) | Speedup |
|---------------|------------------------|---------|
| 1 session × 4 threads | ~100 | 1.0x |
| 4 workers × 1 thread | ~265 | 2.6x |
| **8 workers × 1 thread** | **~360** | **3.6x** |

> **Golden rule**: `workers × threads ≤ CPU core count`

---

## Auto-Tuning

Auto-tuning automatically finds the optimal configuration (workers, threads, batch_size) for your hardware and corpus.

### Simple usage

```python
from libembedding import TextEmbeddingPool

# Autotune with synthetic corpus (default)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,           # enable auto-tuning
    offline=True,
)
# → benchmark ~5-15s first time, then instant cache
```

### Autotune with your corpus (recommended)

For more accurate results, provide a sample of your actual texts:

```python
# Use a representative sample of your data
sample_texts = your_csv["text_column"].head(1000).tolist()

pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,    # your corpus
    autotune_max_samples=100,        # samples 100 representative texts
    offline=True,
)
```

### Autotune cache

Results are cached by machine + model:

```
    %LOCALAPPDATA%\libembedding\autotune\8x4_Intel_i7-1065G7_model_ort1.29_v1.4.0.json
```

| Event | Behavior |
|-------|----------|
| First call | Benchmark + save cache |
| Same machine + model | Cache hit (< 1ms) |
| Change CPU/ORT/model | Cache miss → re-benchmark |

```python
from libembedding import clear_autotune_cache

# Clear cache for a model
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Clear all cache
clear_autotune_cache()
```

### Full API

```python
# TextEmbedding with autotune
model = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)

# TextEmbeddingPool with autotune
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)
```

### Unified Auto-Tuning API

The unified auto-tuner provides a single entry point for all task types. It routes to the appropriate backend implementation and caches results using a hardware fingerprint.

```python
from libembedding import (
    autotune_unified,
    LEMBED_TASK_EMBEDDING,    # text embedding
    LEMBED_TASK_SPARSE,      # sparse embedding
    LEMBED_TASK_IMAGE,       # image embedding
    LEMBED_TASK_RERANKING,   # reranking
    LEMBED_AUTOTUNE_QUICK,   # 5-15s
    LEMBED_AUTOTUNE_FULL,    # 30-120s
)

# Tune a reranker model
result = autotune_unified(
    task=LEMBED_TASK_RERANKING,
    model_name="BAAI/bge-reranker-base",
    mode=LEMBED_AUTOTUNE_QUICK,
)
# result: UnifiedTuningResult with threads, batch_size, max_tokens, latency, etc.
```

### llama.cpp / GGUF Auto-Tuning

When building with llama.cpp support, the unified benchmark and autotuner can compare ONNX vs llama.cpp:

```python
from libembedding import Benchmark, CorpusType, Objective

bench = Benchmark()
comparison = bench.compare_all(
    onnx_path="/path/to/model.onnx",
    gguf_path="/path/to/model.Q4_K_M.gguf",
    corpus=CorpusType.MIXED,
    objective=Objective.BALANCED,
)
print(comparison.recommendation.backend)  # "onnx" or "llama.cpp"
```

---

## Automatic Model Selection

To automatically choose the best model for your hardware and use case:

```python
from libembedding import auto_select_model, TextEmbeddingPool

# Auto-select best model for your hardware + use case
result = auto_select_model("balanced")  # "speed", "quality", or "balanced"

print(f"Best model: {result.model_name}")
print(f"Config: {result.workers} workers × {result.threads} threads")
print(f"Throughput: {result.throughput_docs_sec:.0f} docs/s")

# Create pool with optimal config
pool = TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
)
embeddings = pool.embed(texts)
```

### Use cases

| Use case | Recommendation |
|----------|----------------|
| Real-time, latency critical | `"speed"` |
| Semantic search, max quality | `"quality"` |
| General production | `"balanced"` (default) |

---

## llama.cpp / GGUF Performance

For CPU-bound embedding workloads with small BERT-style models, the optimal configuration is:

### Recommendations

| Parameter | Recommended value | Reason |
|-----------|-------------------|--------|
| `threads` | **1** per session | Avoids contention on small BERT models |
| `workers` / `sessions` | **physical_cores × 2** (max 8) | Near-linear scaling until saturation |
| `batch_size` | 8-32 | Depends on average text length |
| `batch_strategy` | `LENGTH_BUCKET` | Reduces padding for heterogeneous corpora |

### Auto-tuning workers

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,      # auto-detect optimal session count
    cache_size=4096,        # optional LRU cache
)
```

### Preset modes

```python
# FAST : speed priority
model = TextEmbedding.from_mode("fast")

# BALANCED : speed/quality compromise (default)
model = TextEmbedding.from_mode("balanced")

# QUALITY : quality priority
model = TextEmbedding.from_mode("quality")
```

### LRU Cache

```python
# Enable LRU cache with 4096 entries
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    cache_size=4096,
)
```

### Baseline i7-1065G7 (MiniLM-L6-v2 Q4_K_M)

| Sessions | Threads | Docs/s |
|----------|---------|--------|
| 1 | 1 | 41.8 |
| 1 | 4 | 72.1 |
| 4 | 1 | 105.9 |
| **6** | **1** | **125.9** |
| 8 | 1 | 129.4 |
| 6 | 2 | 87.6 |

**Conclusion**: beyond 2 sessions, `threads=1` is always faster. Intra-session multithreading degrades performance on BERT embedding models.

---

## Benchmarks

### Test configuration

- CPU: Intel i7-1065G7 (4c/8t)
- OS: Windows 11
- Model: all-MiniLM-L6-v2 (384-dim)
- Backends: ONNX Runtime and llama.cpp (GGUF Q4_K_M)

### Impact of text length (ONNX)

| Tokens/text | Docs/s (8 workers) | ms/text |
|-------------|---------------------|---------|
| 16 | 696 | 6.0 |
| 64 | 150 | 16.9 |
| 128 | 62 | 32.6 |
| 256 | 15 | 68.6 |

> **Note**: Throughput is strongly dependent on text length. Benchmarks with short texts do not predict performance with long texts.

### ONNX vs llama.cpp comparison

| Backend | Config | Docs/s | RAM |
|---------|--------|--------|-----|
| ONNX | 8 workers × 1 thread | ~128 | ~500 MB |
| llama.cpp GGUF Q4_K_M | 6 sessions × 1 thread | ~126 | ~20 MB |

### llama.cpp configuration comparison

| Sessions | Threads | Docs/s | Efficiency |
|----------|---------|--------|------------|
| 1 | 1 | 41.8 | Low |
| 1 | 4 | 72.1 | Medium |
| 4 | 1 | 105.9 | Good |
| **6** | **1** | **125.9** | **Optimal** |
| 8 | 1 | 129.4 | Saturation |
| 6 | 2 | 87.6 | Degraded |

**Conclusion**: Inter-session parallelism (6×1) is optimal on this machine. Beyond 2 sessions, intra-session multithreading degrades performance.

### Model comparison

| Model | Docs/s (8w×1t) | RAM (8 workers) | Deterministic |
|-------|----------------|-----------------|---------------|
| MiniLM-L6-v2-Q (INT8) | 474-696 | 230 MB | ~1.6% variance |
| MiniLM-L6-v2 (FP32) | 305-361 | 740 MB | Yes |
| BGE-small-en (FP32) | 143-150 | 1.1 GB | Yes |

---

## Best Practices

### 1. For large corpora (> 100K texts)

```python
# Stratified sampling for large corpora
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=large_corpus,      # your 2M texts
    autotune_max_samples=100,         # samples 100 representative texts
    offline=True,
)
# → ~56s for autotune (once)
# → 66 docs/s in production
# → ~8h for 2M texts
```

### 2. For semantic search

```python
# Prioritize quality
pool = TextEmbeddingPool(
    "Xenova/bge-small-en-v1.5",      # better quality than MiniLM
    autotune=True,
    autotune_texts=documents,         # your documents
    offline=True,
)
```

### 3. For real-time

```python
# Prioritize speed
pool = TextEmbeddingPool(
    "Xenova/all-MiniLM-L6-v2",        # INT8 quantized version
    autotune=True,
    autotune_texts=queries,           # your short queries
    offline=True,
)
```

### 4. Batch size strategy

| Batch size | Usage |
|------------|-------|
| 8-16 | Long texts (> 100 tokens) |
| 32-64 | General use |
| 128-256 | Short texts (< 20 tokens) |

### 5. General best practices

- **Reuse the pool**: create it once, reuse it for all embeddings
- **Autotune once**: cache prevents re-benchmarking
- **Use your own texts** for autotune (more accurate than synthetic corpus)
- **Close the pool**: `pool.close()` or context manager `with`
- **Offline mode** in production: `offline=True` prevents downloads

### Complete example: RAG Pipeline

```python
from libembedding import TextEmbeddingPool, auto_select_model
import numpy as np

# 1. Automatic model selection (once)
result = auto_select_model("balanced")
print(f"Selected model: {result.model_name}")

# 2. Create pool with optimal config
with TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
    offline=True,
) as pool:

    # 3. Embed documents
    doc_embeddings = pool.embed(documents)

    # 4. Embed queries
    query_embeddings = pool.embed(queries)

    # 5. Nearest neighbor search
    scores = doc_embeddings @ query_embeddings.T
    top_k = np.argsort(scores, axis=0)[-5:]
```
