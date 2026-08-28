---
nav_exclude: true
---

# Performance Tuning

This guide covers libembedding's advanced performance features: session pooling, auto-tuning, and automatic model selection.

## Table of Contents

1. [Session Pool (EmbeddingPool)](#session-pool-embeddingpool)
2. [Auto-Tuning](#auto-tuning)
3. [Automatic Model Selection](#automatic-model-selection)
4. [Benchmarks](#benchmarks)
5. [Best Practices](#best-practices)

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
%LOCALAPPDATA%\libembedding\autotune\8x4_Intel_i7-1065G7_model_ort1.29_v0.2.json
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

## Benchmarks

### Test configuration

- CPU: Intel i7-1065G7 (4c/8t)
- OS: Windows 11
- Model: all-MiniLM-L6-v2 (384-dim)

### Impact of text length

| Tokens/text | Docs/s (8 workers) | ms/text |
|-------------|---------------------|---------|
| 16 | 696 | 6.0 |
| 64 | 150 | 16.9 |
| 128 | 62 | 32.6 |
| 256 | 15 | 68.6 |

> **Note**: Throughput is strongly dependent on text length. Benchmarks with short texts do not predict performance with long texts.

### Configuration comparison

| Configuration | Docs/s | CPU% | Efficiency |
|---------------|--------|------|------------|
| 1×4 (intra-op) | 52 | 99% | Low |
| 4×1 | 102 | 75% | Good |
| **8×1** | **128** | **99%** | **Optimal** |

**Conclusion**: Inter-session parallelism (8×1) is ~2.5x more efficient than intra-session parallelism (1×4) for small Transformers on CPU.

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
