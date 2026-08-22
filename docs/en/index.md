# libembedding Documentation

Welcome to the **libembedding** documentation, a fast embedding library with both **C/C++** and **Python** APIs powered by ONNX Runtime.

## Documentation structure

| Section | File | Description |
|---------|------|-------------|
| **Getting started** | [getting_started.md](getting_started.md) | Installation, prerequisites and quick start |
| **Python API** | [api_reference.md](api_reference.md) | Complete reference of Python classes |
| **Models** | [models.md](models.md) | Catalog of available models (text, image, sparse, reranker) |
| **Advanced usage** | [advanced_usage.md](advanced_usage.md) | Local models, providers, cache, offline mode, context managers |
| **Error handling** | [api_reference.md#error-handling](api_reference.md#error-handling) | Python exception hierarchy |
| **Français** | [../index.md](../index.md) | Documentation française |

## Overview

```python
from libembedding import TextEmbedding, SparseTextEmbedding, Reranker
import numpy as np

# Dense embeddings
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)  # (2, 384)

# Sparse embeddings
sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning"])
print(results[0].indices.shape, results[0].values.shape)

# Reranking
reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank("What is deep learning?", [
    "Deep learning uses neural networks",
    "The weather is sunny today",
])
print(ranked[0].score, ranked[0].index)
```

**Performance**: 5-8x faster than fastembed, 3.5x less memory.
