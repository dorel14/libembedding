# libembedding

Fast ONNX-based text, image, and sparse embeddings for Python. **5-8x faster than fastembed** with 3.5x less memory.

Built on a C/C++ backend using ONNX Runtime, exposed to Python via zero-overhead cffi bindings. Supports 44 text embedding models, 5 image models, 2 sparse models, and 4 rerankers with automatic model downloading from HuggingFace Hub.

## Installation

```bash
pip install libembedding
```

**Requirements:** ONNX Runtime must be installed on your system.

```bash
# macOS
brew install onnxruntime

# Ubuntu/Debian
apt install libonnxruntime-dev

# Or set ONNXRUNTIME_ROOT to your installation path
```

## Quick Start

### Text Embeddings

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])

print(embeddings.shape)  # (2, 384)
print(embeddings.dtype)  # float32
```

### Sparse Embeddings

```python
from libembedding import SparseTextEmbedding

model = SparseTextEmbedding()
results = model.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
```

### Image Embeddings

```python
from libembedding import ImageEmbedding

model = ImageEmbedding()
embeddings = model.embed_files(["photo.jpg", "diagram.png"])
```

### Reranking

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
results = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks with many layers",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)
for r in results:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

### Model Discovery

```python
import libembedding

for m in libembedding.list_text_models():
    print(f"{m.model_name:45} dim={m.dim:<5} {m.pooling}")
```

## API Reference

### TextEmbedding

```python
TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # HuggingFace model name, repo code, or local dir path
    provider="cpu",                         # "cpu", "cuda", "coreml", "directml", "tensorrt"
    device_id=0,
    cache_dir=None,                         # None = ~/.cache/libembedding
    max_length=0,                           # 0 = model default
    threads=0,                              # 0 = auto
    batch_size=256,                         # internal batch size for embedding
    offline=False,                          # True = use cache only, never download
    show_download_progress=True,
    dim=0,                                  # embedding dim for local models without config.json
    pooling="mean",                         # "cls" or "mean" for local models
    num_threads=0,                          # deprecated, use threads
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(texts, batch_size=None)` | `np.ndarray (n, dim)` | L2-normalized dense embeddings |
| `dim` | `int` | Embedding dimension |
| `name` | `str` | Model name or local path |
| `info()` | `ModelDesc` | Runtime model descriptor |
| `close()` | `None` | Release resources |

### SparseTextEmbedding

```python
SparseTextEmbedding(model_name="prithvida/SPLADE_PP_en_v1", ...)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(texts, batch_size=0)` | `list[SparseEmbedding]` | Sparse vectors with `.indices` and `.values` |

### ImageEmbedding

```python
ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision", ...)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed_files(paths, batch_size=0)` | `np.ndarray (n, dim)` | Embed from file paths |
| `embed_bytes(images, batch_size=0)` | `np.ndarray (n, dim)` | Embed from raw bytes |

### Reranker

```python
Reranker(model_name="BAAI/bge-reranker-base", ...)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `rerank(query, documents, batch_size=0)` | `list[RerankResult]` | Sorted by score descending |

All classes support context managers (`with TextEmbedding(...) as model:`).

### Local Model Loading

```python
from libembedding import TextEmbedding

# Load from a local directory containing model.onnx + tokenizer.json (+ config.json)
model = TextEmbedding("/path/to/model_dir")
```

## Available Models

**44 text models** including BGE, MiniLM, Nomic, E5, CLIP, Jina, GTE, Snowflake, ModernBERT (with quantized variants).

**5 image models** including CLIP ViT-B/32, ResNet-50, Unicom, Nomic Vision.

**2 sparse models**: SPLADE++, BGE-M3.

**4 reranker models**: BGE Reranker, Jina Reranker.

## Benchmarks

Measured on Apple M-series with `all-MiniLM-L6-v2` (384-dim). Median of 10 runs.

| Metric                   | libembedding | fastembed | Speedup |
|--------------------------|-------------|-----------|---------|
| Single text latency (ms) | **4.4**     | 38.0      | **8.6x**|
| Batch 8 (texts/sec)      | **641**     | 92        | **7.0x**|
| Batch 32 (texts/sec)     | **581**     | 89        | **6.5x**|
| Peak RSS (MB)            | **567**     | 1,981     | **3.5x less**|

## Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `LIBEMBEDDING_CACHE_DIR` | Override model cache directory |
| `FASTEMBED_CACHE_DIR` | Alternative cache dir (fastembed compatibility) |
| `HF_ENDPOINT` | Custom HuggingFace Hub endpoint |

## Building & Publishing

### Prerequisites

```bash
pip install build twine
```

### Build the shared library + wheel

```bash
cd python/

# Step 1: Build the C/C++ shared library and copy it into the package
./setup.sh --build-only

# Step 2: Build sdist and wheel
python -m build
```

This produces files in `dist/`:
```
dist/
  libembedding-0.2.0.tar.gz              # source distribution
  libembedding-0.2.0-py3-none-any.whl    # wheel (includes bundled .dylib/.so)
```

### Upload to PyPI

```bash
# Upload to TestPyPI first to verify
twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ libembedding

# Upload to production PyPI
twine upload dist/*
```

### One-liner (build + upload)

```bash
./setup.sh --build-only && python -m build && twine upload dist/*
```

### Platform-specific wheels

The default wheel is `py3-none-any` and bundles the shared library for the build platform. To build platform-tagged wheels for distribution:

```bash
# macOS (current arch)
./setup.sh --build-only
python -m build

# For other platforms, build on that platform or use cibuildwheel:
pip install cibuildwheel
cibuildwheel --platform linux   # builds manylinux wheels
cibuildwheel --platform macos   # builds macOS wheels
```

## License

MIT
