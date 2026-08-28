# Performance Tuning

Ce guide couvre les fonctionnalités avancées de performance de libembedding : pool de sessions, auto-tuning, et sélection automatique de modèle.

## Sommaire

1. [Pool de sessions (EmbeddingPool)](#pool-de-sessions-embeddingpool)
2. [Auto-Tuning](#auto-tuning)
3. [Sélection automatique de modèle](#sélection-automatique-de-modèle)
4. [Benchmarks](#benchmarks)
5. [Bonnes pratiques](#bonnes-pratiques)

---

## Pool de sessions (EmbeddingPool)

Pour les petites architectures Transformer (MiniLM, BGE-small, E5-small), le parallélisme **inter-sessions** (plusieurs sessions ONNX indépendantes) est plus efficace que le parallélisme **intra-session** (threads ONNX).

### Quand l'utiliser

| Scénario | Recommandation |
|----------|----------------|
| < 100 embeddings | `TextEmbedding` simple suffit |
| > 100 embeddings | `TextEmbeddingPool` recommandé |
| Production / haut débit | `TextEmbeddingPool` + `autotune=True` |

### Utilisation

```python
from libembedding import TextEmbeddingPool

# Pool avec 8 workers (sessions ONNX indépendantes)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    workers=8,               # nombre de sessions parallèles
    threads_per_worker=1,    # 1 thread par session (évite la contention)
    batch_size=64,
    offline=True,
)

embeddings = pool.embed(texts)
pool.close()
```

### Gain de performance

| Configuration | Docs/s (textes courts) | Speedup |
|---------------|------------------------|---------|
| 1 session × 4 threads | ~100 | 1.0x |
| 4 workers × 1 thread | ~265 | 2.6x |
| **8 workers × 1 thread** | **~360** | **3.6x** |

> **Règle d'or** : `workers × threads ≤ nombre de cœurs CPU`

---

## Auto-Tuning

L'auto-tuning trouve automatiquement la configuration optimale (workers, threads, batch_size) pour votre machine et votre corpus.

### Utilisation simple

```python
from libembedding import TextEmbeddingPool

# Autotune avec corpus synthétique (défaut)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,           # active l'auto-tuning
    offline=True,
)
# → benchmark ~5-15s la première fois, puis cache instantané
```

### Autotune avec votre corpus (recommandé)

Pour des résultats plus précis, fournissez un échantillon de vos vrais textes :

```python
# Utilisez un échantillon représentatif de vos données
sample_texts = votre_csv["text_column"].head(1000).tolist()

pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,    # votre corpus
    autotune_max_samples=100,        # échantillonne 100 textes représentatifs
    offline=True,
)
```

### Cache d'autotuning

Les résultats sont mis en cache par machine + modèle :

```
%LOCALAPPDATA%\libembedding\autotune\8x4_Intel_i7-1065G7_model_ort1.29_v0.2.json
```

| Événement | Comportement |
|-----------|--------------|
| Premier appel | Benchmark + sauvegarde cache |
| Même machine + modèle | Cache hit (< 1ms) |
| Changement de CPU/ORT/modèle | Cache miss → re-benchmark |

```python
from libembedding import clear_autotune_cache

# Effacer le cache d'un modèle
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Effacer tout le cache
clear_autotune_cache()
```

### API complète

```python
# TextEmbedding avec autotune
model = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)

# TextEmbeddingPool avec autotune
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)
```

---

## Sélection automatique de modèle

Pour choisir automatiquement le meilleur modèle selon votre hardware et votre cas d'usage :

```python
from libembedding import auto_select_model, TextEmbeddingPool

# Sélection automatique
result = auto_select_model("balanced")  # "speed", "quality", ou "balanced"

print(f"Modèle: {result.model_name}")
print(f"Dimension: {result.dim}")
print(f"Config: {result.workers} workers × {result.threads} threads")
print(f"Throughput: {result.throughput_docs:.0f} docs/s")

# Utilisation directe
pool = TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
)
```

### Cas d'usage

| Use case | Recommandation |
|----------|----------------|
| Temps réel, latence critique | `"speed"` |
| Recherche sémantique, qualité maximale | `"quality"` |
| Production généraliste | `"balanced"` (défaut) |

---

## Benchmarks

### Configuration testée

- CPU : Intel i7-1065G7 (4c/8t)
- OS : Windows 11
- Modèle : all-MiniLM-L6-v2 (384-dim)

### Impact de la longueur des textes

| Tokens/texte | Docs/s (8 workers) | ms/texte |
|--------------|---------------------|----------|
| 16 | 696 | 6.0 |
| 64 | 150 | 16.9 |
| 128 | 62 | 32.6 |
| 256 | 15 | 68.6 |

> **Note** : Le throughput est fortement dépendant de la longueur des textes. Les benchmarks avec textes courts ne prédisent pas les performances avec textes longs.

### Comparaison des configurations

| Configuration | Docs/s | CPU% | Efficacité |
|---------------|--------|------|------------|
| 1×4 (intra-op) | 52 | 99% | Faible |
| 4×1 | 102 | 75% | Bonne |
| **8×1** | **128** | **99%** | **Optimale** |

**Conclusion** : Le parallélisme inter-sessions (8×1) est ~2.5x plus efficace que le parallélisme intra-session (1×4) pour les petits Transformers sur CPU.

### Comparaison des modèles

| Modèle | Docs/s (8w×1t) | RAM (8 workers) | Déterministe |
|--------|----------------|-----------------|--------------|
| MiniLM-L6-v2-Q (INT8) | 474-696 | 230 MB | ~1.6% variance |
| MiniLM-L6-v2 (FP32) | 305-361 | 740 MB | Oui |
| BGE-small-en (FP32) | 143-150 | 1.1 GB | Oui |

---

## Bonnes pratiques

### 1. Pour les gros corpus (> 100K textes)

```python
# Échantillonnage stratifié automatique
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=large_corpus,      # vos 2M de textes
    autotune_max_samples=100,         # échantillonne 100 textes représentatifs
    offline=True,
)
# → ~56s pour autotuner (une fois)
# → 66 docs/s en production
# → ~8h pour 2M de textes
```

### 2. Pour la recherche sémantique

```python
# Privilégiez la qualité
pool = TextEmbeddingPool(
    "Xenova/bge-small-en-v1.5",      # meilleure qualité que MiniLM
    autotune=True,
    autotune_texts=documents,         # vos documents
    offline=True,
)
```

### 3. Pour le temps réel

```python
# Privilégiez la vitesse
pool = TextEmbeddingPool(
    "Xenova/all-MiniLM-L6-v2",        # version quantifiée INT8
    autotune=True,
    autotune_texts=queries,           # vos requêtes courtes
    offline=True,
)
```

### 4. Stratégie de batch

| Taille de batch | Usage |
|-----------------|-------|
| 8-16 | Textes longs (> 100 tokens) |
| 32-64 | Usage général |
| 128-256 | Textes courts (< 20 tokens) |

### 5. Bonnes pratiques générales

- **Réutilisez le pool** : créez-le une fois, réutilisez-le pour tous les embeddings
- **Autotune une seule fois** : le cache évite de re-benchmarker
- **Utilisez vos propres textes** pour l'autotune (plus précis que le corpus synthétique)
- **Fermez le pool** : `pool.close()` ou context manager `with`
- **Mode offline** en production : `offline=True` évite les téléchargements

### Exemple complet : Pipeline RAG

```python
from libembedding import TextEmbeddingPool, auto_select_model
import numpy as np

# 1. Sélection automatique du modèle (une fois)
result = auto_select_model("balanced")
print(f"Modèle sélectionné: {result.model_name}")

# 2. Création du pool avec config optimale
with TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
    offline=True,
) as pool:

    # 3. Embedding des documents
    doc_embeddings = pool.embed(documents)

    # 4. Embedding des requêtes
    query_embeddings = pool.embed(queries)

    # 5. Recherche du plus proche voisin
    scores = doc_embeddings @ query_embeddings.T
    top_k = np.argsort(scores, axis=0)[-5:]
```
