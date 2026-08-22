# Advanced usage

## Local models (Bring Your Own ONNX)

You can use your own ONNX models without downloading them. Simply place your files in a local directory:

```
my_model/
├── model.onnx
├── tokenizer.json
└── config.json       # optional but recommended
```

Then load it by passing the path:

```python
from libembedding import TextEmbedding

model = TextEmbedding("/path/to/my_model")
embeddings = model.embed(["Hello world"])
```

### Parameters for local models

If your model does not have a `config.json`, you must specify manually:

```python
model = TextEmbedding(
    "/path/to/my_model",
    dim=768,           # embedding dimension
    pooling="mean",    # "cls" or "mean"
    max_length=512,    # max token length
)
```

## Execution providers

The `provider` parameter controls which ONNX Runtime backend is used for inference:

```python
from libembedding import TextEmbedding

# CPU (default, always available)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cpu")

# NVIDIA CUDA
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cuda", device_id=0)

# Apple CoreML (macOS/iOS)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="coreml")

# DirectML (Windows)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="directml")

# NVIDIA TensorRT
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="tensorrt")
```

**Supported providers:**

| Provider | Backend | Platform |
|----------|---------|----------|
| `cpu` | CPU | All |
| `cuda` | CUDA | NVIDIA GPU |
| `coreml` | CoreML | macOS, iOS |
| `directml` | DirectML | Windows |
| `tensorrt` | TensorRT | NVIDIA GPU |

## Cache management

### Default cache directory

Models are downloaded and cached in `~/.cache/libembedding` by default.

### Changing the cache directory

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_dir="/custom/cache/dir"
)
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `LIBEMBEDDING_CACHE_DIR` | Globally override the cache directory |
| `FASTEMBED_CACHE_DIR` | Alternative cache directory (fastembed compatibility) |
| `HF_ENDPOINT` | Custom HuggingFace Hub endpoint URL (e.g. mirror) |

```bash
# Usage examples
export LIBEMBEDDING_CACHE_DIR=/data/models/libembedding
export HF_ENDPOINT=https://hf-mirror.com
```

## Offline mode

Use offline mode to work without network access:

```python
from libembedding import TextEmbedding

# Only attempt to load from local cache
model = TextEmbedding("BAAI/bge-small-en-v1.5", offline=True)
```

Or set the environment variable:

```bash
export LIBEMBEDDING_NO_DOWNLOAD=1
```

## Context managers

All classes support the context manager protocol for automatic C resource management:

```python
from libembedding import TextEmbedding

with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    embeddings = model.embed(["Hello", "World"])
# Resources are released automatically on exit
```

## Manual resource management

If you do not use a context manager, call `close()` explicitly:

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
try:
    embeddings = model.embed(["Hello", "World"])
finally:
    model.close()
```

## Batch processing

Control batch size to balance speed and memory:

```python
from libembedding import TextEmbedding

# Batch size at construction time
model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=64)

# Or at embed() call time
model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=256)
embeddings = model.embed(texts, batch_size=128)  # temporary override
```

## Token limit

Some models have a maximum token length:

```python
from libembedding import TextEmbedding

# Limit to 128 tokens (default is 512 for BGE-small)
model = TextEmbedding("BAAI/bge-small-en-v1.5", max_length=128)
```

## Model lookup by code

libembedding automatically resolves models by HuggingFace name or ONNX repo code:

```python
from libembedding import TextEmbedding

# By full HuggingFace name
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# By ONNX repo code
model = TextEmbedding("Xenova/bge-small-en-v1.5")

# By local path
model = TextEmbedding("/path/to/local/model")
```

## numpy integration

Returned embeddings are L2-normalized `float32` numpy arrays:

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["doc1", "doc2", "doc3"])

# Cosine similarity (dot product because embeddings are L2-normalized)
similarity = embeddings @ embeddings.T

# Nearest neighbor search
query = model.embed(["search query"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

## Full semantic search example

```python
from libembedding import TextEmbedding
import numpy as np

class SemanticSearch:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[str]):
        self.documents.extend(documents)
        new_embeddings = self.model.embed(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
        query_emb = self.model.embed([query])
        scores = (self.embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i]), self.documents[i]) for i in top_indices]

    def close(self):
        self.model.close()
```

## Comparison with fastembed

libembedding is designed as a drop-in replacement for fastembed:

```python
# fastembed
from fastembed import TextEmbedding as FastEmbed
model = FastEmbed("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["Hello"]))

# libembedding — same API
from libembedding import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello"])  # returns np.ndarray directly
```

**Notable differences:**
- `embed()` returns a `np.ndarray` directly instead of a generator
- No separate `passage_embed()` method — everything goes through `embed()`
- Lighter classes, fewer Python dependencies

## Performance tips

For optimal performance:

1. **Reuse the model** — create it once and embed multiple batches
2. **Tune `batch_size`** — increase until memory saturation
3. **Use a GPU provider** — `cuda`, `coreml`, or `directml`
4. **Enable quantization** — choose `_Q` models for faster inference
5. **Disable progress** — `show_download_progress=False` for scripts
