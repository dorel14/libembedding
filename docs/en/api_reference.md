---
nav_exclude: true
---

# Python API Reference

## Main classes

### TextEmbedding

Generates dense vector embeddings from text.

#### Constructor

```python
TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    dim=0,
    pooling="mean",
    num_threads=None,  # deprecated
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"BAAI/bge-small-en-v1.5"` | HuggingFace model name, ONNX repo code, or local directory path containing `model.onnx` + `tokenizer.json` |
| `provider` | `str` | `"cpu"` | Execution provider: `"cpu"`, `"cuda"`, `"coreml"`, `"directml"`, `"tensorrt"`, `"llamacpp"` |
| `device_id` | `int` | `0` | Device index for GPU providers |
| `cache_dir` | `str \| None` | `None` | Model cache directory (`None` = `~/.cache/libembedding`) |
| `max_length` | `int` | `0` | Max token length (`0` = model default) |
| `threads` | `int` | `0` | Number of threads (`0` = auto) |
| `batch_size` | `int` | `256` | Internal batch size for inference |
| `offline` | `bool` | `False` | `True` = use cache only, never download |
| `show_download_progress` | `bool` | `True` | Show download progress bar |
| `dim` | `int` | `0` | Embedding dimension for local models without `config.json` |
| `pooling` | `str` | `"mean"` | Pooling strategy for local models: `"cls"` or `"mean"` |
| `num_threads` | `int \| None` | `None` | **Deprecated** — use `threads` instead |
| `autotune` | `bool` | `False` | `True` = auto-tune `threads` and `batch_size` for best performance. See [performance_tuning.html](performance_tuning.html) |
| `autotune_texts` | `list[str] \| None` | `None` | Custom corpus for autotune (more accurate than synthetic corpus) |
| `autotune_max_samples` | `int` | `100` | Maximum texts to sample from corpus for autotune |
| `auto_workers` | `bool` | `False` | `True` = auto-detect optimal sessions/workers for llama.cpp backend |
| `cache_size` | `int` | `0` | LRU embedding cache size (`0` = disabled) |

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=None)` | `np.ndarray` | Embed texts. Returns array of shape `(n, dim)` with `float32` dtype. L2-normalized. |
| `dim` | `int` (property) | Embedding dimension |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name or local path |
| `info()` | `ModelDesc` | Loaded model descriptor |
| `close()` | `None` | Release underlying C resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List all supported text models |
| `__enter__()` | `self` | Context manager support |
| `__exit__()` | `None` | Calls `close()` automatically |
| `from_gguf(repo, filename=None, provider="llamacpp", ...)` | `TextEmbedding` | Load a GGUF model from HuggingFace (requires llama.cpp backend) |
| `from_mode(mode="balanced", **kwargs)` | `TextEmbedding` | (classmethod) Create TextEmbedding from preset mode: `"fast"`, `"balanced"`, `"quality"` |
| `supports_llamacpp()` | `bool` | (static) Check if llama.cpp backend is compiled in |

#### Example

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)   # (2, 384)
print(embeddings.dtype)   # float32

# Simple semantic search
query = model.embed(["What is AI?"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

---

### SparseTextEmbedding

Generates **sparse** embeddings (sparse vectors with token indices and weights).

#### Constructor

```python
SparseTextEmbedding(
    model_name="prithvida/SPLADE_PP_en_v1",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    num_threads=None,  # deprecated
)
```

**Available models:**

| HuggingFace name | Description |
|------------------|-------------|
| `prithvida/SPLADE_PP_en_v1` | SPLADE++ (default) |
| `BAAI/bge-m3` | Multilingual BGE-M3 |

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=0)` | `list[SparseEmbedding]` | Returns a list of `SparseEmbedding` objects |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name |
| `info()` | `ModelDesc` | Model descriptor |
| `close()` | `None` | Release resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List sparse models |

#### Example

```python
from libembedding import SparseTextEmbedding

sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
    # indices: int32 array of token IDs
    # values: float32 array of weights
```

---

### ImageEmbedding

Generates dense vector embeddings from images.

#### Constructor

```python
ImageEmbedding(
    model_name="Qdrant/clip-ViT-B-32-vision",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    threads=0,
    batch_size=30,
    offline=False,
    show_download_progress=True,
    dim=0,
    num_threads=None,  # deprecated
)
```

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed_files(paths, batch_size=None)` | `np.ndarray` | Embed from file paths. Shape: `(n, dim)` |
| `embed_bytes(images, batch_size=None)` | `np.ndarray` | Embed from raw bytes (JPEG, PNG, etc.) |
| `dim` | `int` (property) | Embedding dimension |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name |
| `info()` | `ModelDesc` | Model descriptor |
| `close()` | `None` | Release resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List image models |

#### Example

```python
from libembedding import ImageEmbedding

model = ImageEmbedding()
embeddings = model.embed_files(["photo.jpg", "diagram.png"])
print(embeddings.shape)  # (2, 512)
```

---

### Reranker

Scores and sorts documents by relevance to a query (cross-encoder).

#### Constructor

```python
Reranker(
    model_name="BAAI/bge-reranker-base",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    num_threads=None,  # deprecated
)
```

**Available models:**

| HuggingFace name | Description |
|------------------|-------------|
| `BAAI/bge-reranker-base` | BGE Reranker base (default) |
| `BAAI/bge-reranker-v2-m3` | Multilingual BGE Reranker v2 |
| `jinaai/jina-reranker-v1-turbo-en` | Jina Reranker v1 turbo English |
 | `jinaai/jina-reranker-v2-base-multilingual` | Jina Reranker v2 multilingual |
| `jinaai/jina-reranker-v1-turbo-en` (quantized) | Jina Reranker v1 turbo English (INT8) |

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `rerank(query, documents, batch_size=0)` | `list[RerankResult]` | Returns results sorted by descending score |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name |
| `info()` | `ModelDesc` | Model descriptor |
| `close()` | `None` | Release resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List reranker models |

#### Example

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)

for r in ranked:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

---

## Data types

### TextEmbeddingPool

Pool of ONNX sessions for inter-session parallelism. See [performance_tuning.html](performance_tuning.html).

```python
class TextEmbeddingPool:
    model_name: str
    workers: int = 0                  # number of sessions (0 = auto-detect)
    threads_per_worker: int = 1       # threads per session
    batch_size: int = 256
    provider: str = "cpu"
    offline: bool = False
    autotune: bool = False            # auto-tune all parameters
    autotune_texts: list[str] = None  # corpus for autotune
    autotune_max_samples: int = 100   # max sample size
```

**Methods:**

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts)` | `np.ndarray` | Embed texts in parallel |
| `num_workers` | `int` (property) | Number of active workers |
| `dim` | `int` (property) | Embedding dimension |
| `close()` | `None` | Release resources |

### TuningResult

Result of autotune for a model.

```python
@dataclass(frozen=True)
class TuningResult:
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### ModelSelectionResult

Result of automatic model selection.

```python
@dataclass(frozen=True)
class ModelSelectionResult:
    model_code: str
    model_name: str
    dim: int
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
    score: float
```

### Global functions

| Function | Description |
|----------|-------------|
| `autotune(model_name, full=False)` | Auto-tune a model. Returns `TuningResult`. |
| `autotune_unified(task, model_name, mode="quick")` | Unified auto-tune for any task type. See [Unified Tuning](#unified-tuning). |
| `sparse_autotune(model_name, mode="quick")` | Auto-tune sparse embedding. Returns `SparseTuningResult`. |
| `image_autotune(model_name, mode="quick")` | Auto-tune image embedding. Returns `ImageTuningResult`. |
| `reranker_autotune(model_name, mode="quick", objective="latency")` | Auto-tune reranker. Returns `RerankerTuningResult`. |
| `reranker_auto_config(model_name, target_latency_ms=500, objective="latency")` | Auto-config reranker for a latency budget. |
| `reranker_auto_config_profile(model_name, profile="balanced")` | Auto-config reranker using a profile. |
| `reranker_autotune_constrained(model_name, mode="quick", objective="latency", min_tokens=32, max_latency_ms=1000)` | Auto-tune with constraints. |
| `clear_reranker_autotune_cache(model_name=None)` | Clear reranker autotune cache. |
| `auto_select_model(use_case="balanced")` | Select best model. Returns `ModelSelectionResult`. |
| `clear_autotune_cache(model_name=None)` | Clear autotune cache. |
| `cache_path()` | Get autotune cache file path. |

---

### Unified Tuning

The unified auto-tuner provides a single API for optimizing all task types (text embedding, sparse, image, reranking).

```python
from libembedding import (
    autotune_unified,
    LEMBED_TASK_EMBEDDING,
    LEMBED_TASK_SPARSE,
    LEMBED_TASK_IMAGE,
    LEMBED_TASK_RERANKING,
    LEMBED_AUTOTUNE_QUICK,
    LEMBED_AUTOTUNE_FULL,
    UnifiedTuningResult,
)

# Tune any task type
result = autotune_unified(
    task=LEMBED_TASK_RERANKING,
    model_name="BAAI/bge-reranker-base",
    mode=LEMBED_AUTOTUNE_QUICK,
)
print(result.threads, result.batch_size, result.max_tokens)
```

#### UnifiedTuningResult

```python
@dataclass(frozen=True)
class UnifiedTuningResult:
    task: str               # "embedding", "reranking", "image", or "sparse"
    threads: int
    batch_size: int
    workers: int            # embedding only
    max_tokens: int         # reranker only
    top_k: int              # sparse only
    min_weight: float       # sparse only
    storage_format: int    # sparse only
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float
```

### SparseTuningResult

```python
@dataclass(frozen=True)
class SparseTuningResult:
    top_k: int
    min_weight: float
    storage_format: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### ImageTuningResult

```python
@dataclass(frozen=True)
class ImageTuningResult:
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### RerankerTuningResult

```python
@dataclass(frozen=True)
class RerankerTuningResult:
    threads: int
    batch_size: int
    max_tokens: int
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float
```

---

### SparseEmbedding

```python
@dataclass(frozen=True)
class SparseEmbedding:
    indices: np.ndarray  # int32 — token IDs
    values:  np.ndarray  # float32 — token weights
```

### RerankResult

```python
@dataclass(frozen=True)
class RerankResult:
    index: int   # index of the document in the input list
    score: float # relevance score (higher = more relevant)
```

### ModelInfo

```python
@dataclass(frozen=True)
class ModelInfo:
    model_name:    str   # e.g. "BAAI/bge-small-en-v1.5"
    model_code:    str   # HF repo code, e.g. "Xenova/bge-small-en-v1.5"
    model_file:    str   # e.g. "onnx/model.onnx"
    description:   str
    dim:           int   # embedding dimension
    max_tokens:    int   # max token length
    pooling:       str   # "cls" or "mean"
    quantization:  str   # "none", "static", "dynamic"
```

### ModelDesc

```python
@dataclass(frozen=True)
class ModelDesc:
    name:        str
    dimension:   int
    max_length:  int
    pooling:     str   # "cls" or "mean"
    num_threads: int
    batch_size:  int
    provider:    str
    device_id:   int
```

---

## Error handling

All exceptions inherit from `LembedError`. The `LlamaError` subclass is raised for llama.cpp/GGUF backend failures:

```python
from libembedding import LembedError
from libembedding.exceptions import (
    InvalidArgumentError,
    OutOfMemoryError,
    OnnxRuntimeError,
    TokenizerError,
    DownloadError,
    IOError,
    ModelNotFoundError,
    UnsupportedError,
    BatchSizeError,
    LlamaError,
)
```

**Hierarchy:**

```
LembedError (Exception)
├── InvalidArgumentError   — NULL pointer or out-of-range enum
├── OutOfMemoryError       — allocation failure
├── OnnxRuntimeError       — ONNX Runtime error
├── TokenizerError         — tokenizer loading/encoding error
├── DownloadError          — download failure
├── IOError                — file I/O error
├── ModelNotFoundError     — unknown model
├── UnsupportedError       — feature disabled at compile time
├── BatchSizeError         — incompatible batch size
└── LlamaError             — llama.cpp/GGUF backend error
```

`LlamaError` provides:
- `status_code` (`int`) — `LEMBED_ERROR_LLAMA`
- `message` (`str`) — `"llama.cpp backend error"`
- `detail` (`str`) — technical detail from llama.cpp

#### Example

```python
from libembedding import TextEmbedding
from libembedding.exceptions import ModelNotFoundError, OnnxRuntimeError

try:
    model = TextEmbedding("nonexistent/model")
except ModelNotFoundError as e:
    print(f"Model not found: {e.detail}")
except OnnxRuntimeError as e:
    print(f"ONNX error: {e.detail}")
except LembedError as e:
    print(f"libembedding error [{e.status_code}]: {e.message}")
```

Each exception provides:
- `status_code` (`int`) — C error code
- `message` (`str`) — generic message (e.g. `"ONNX Runtime error"`)
- `detail` (`str`) — technical detail (e.g. ORT error message)

---

## Utility functions

```python
import libembedding

# List models by category
libembedding.list_text_models()      # list[ModelInfo]
libembedding.list_sparse_models()    # list[ModelInfo]
libembedding.list_image_models()     # list[ModelInfo]
libembedding.list_reranker_models()  # list[ModelInfo]
```
