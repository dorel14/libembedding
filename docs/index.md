# Documentation libembedding

Bienvenue dans la documentation de **libembedding**, une bibliothèque d'embeddings rapide avec APIs **C/C++** et **Python** basée sur ONNX Runtime.

## Structure de la documentation

| Section | Fichier | Description |
|---------|---------|-------------|
| **Démarrage** | [getting_started.md](getting_started.md) | Installation, prérequis et premiers pas |
| **API Python** | [api_reference.md](api_reference.md) | Référence complète des classes Python |
| **Modèles** | [models.md](models.md) | Catalogue des modèles disponibles (texte, image, sparse, reranker) |
| **Usage avancé** | [advanced_usage.md](advanced_usage.md) | Modèles locaux, providers, cache, mode hors-ligne, context managers |
| **Exceptions** | [api_reference.md#gestion-des-erreurs](api_reference.md#gestion-des-erreurs) | Hiérarchie des exceptions Python |
| **English** | [en/](en/index.md) | English documentation |

## Vue d'ensemble

```python
from libembedding import TextEmbedding, SparseTextEmbedding, Reranker
import numpy as np

# Embeddings denses
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

**Performance** : 5-8x plus rapide que fastembed, 3.5x moins de mémoire.
