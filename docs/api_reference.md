---
title: API Python
nav_order: 3
---

# Référence API Python

## Classes principales

### TextEmbedding

Génère des embeddings denses (vecteurs) à partir de textes.

#### Constructeur

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
    num_threads=None,  # déprécié
)
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"BAAI/bge-small-en-v1.5"` | Nom HuggingFace, code de repo, ou chemin local vers un répertoire contenant `model.onnx` + `tokenizer.json` |
| `provider` | `str` | `"cpu"` | Provider d'exécution : `"cpu"`, `"cuda"`, `"coreml"`, `"directml"`, `"tensorrt"` |
| `device_id` | `int` | `0` | Index du device pour les providers GPU |
| `cache_dir` | `str \| None` | `None` | Répertoire de cache des modèles (`None` = `~/.cache/libembedding`) |
| `max_length` | `int` | `0` | Longueur max en tokens (`0` = défaut du modèle) |
| `threads` | `int` | `0` | Nombre de threads (`0` = auto) |
| `batch_size` | `int` | `256` | Taille de batch interne pour l'inférence |
| `offline` | `bool` | `False` | `True` = utilise uniquement le cache, pas de téléchargement |
| `show_download_progress` | `bool` | `True` | Affiche la barre de progression de téléchargement |
| `dim` | `int` | `0` | Dimension de l'embedding pour modèles locaux sans `config.json` |
| `pooling` | `str` | `"mean"` | Stratégie de pooling pour modèles locaux : `"cls"` ou `"mean"` |
| `num_threads` | `int \| None` | `None` | **Déprécié** — utiliser `threads` à la place |
| `autotune` | `bool` | `False` | `True` = auto-tune `threads` et `batch_size` pour meilleures performances. Voir [performance_tuning.html](performance_tuning.html) |
| `autotune_texts` | `list[str] \| None` | `None` | Corpus personnalisé pour l'autotune (plus précis que corpus synthétique) |
| `autotune_max_samples` | `int` | `100` | Nombre max de textes échantillonnés pour l'autotune (si `autotune_texts` fourni) |

#### Méthodes et propriétés

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=None)` | `np.ndarray` | Embed les textes. Retourne un tableau de forme `(n, dim)` en `float32`. L2-normalisé. |
| `dim` | `int` (property) | Dimension de l'embedding |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle ou chemin local |
| `info()` | `ModelDesc` | Descripteur du modèle chargé |
| `close()` | `None` | Libère les ressources C sous-jacentes |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste tous les modèles de texte supportés |
| `__enter__()` | `self` | Support du context manager |
| `__exit__()` | `None` | Appelle `close()` automatiquement |

#### Exemple

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)   # (2, 384)
print(embeddings.dtype)   # float32

# Recherche sémantique simple
query = model.embed(["What is AI?"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

---

### SparseTextEmbedding

Génère des embeddings **sparse** (vecteurs creux avec indices de tokens et poids).

#### Constructeur

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
    num_threads=None,  # déprécié
)
```

**Modèles disponibles :**

| Nom HuggingFace | Description |
|-----------------|-------------|
| `prithvida/SPLADE_PP_en_v1` | SPLADE++ (défaut) |
| `BAAI/bge-m3` | BGE-M3 multilingue |

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=0)` | `list[SparseEmbedding]` | Retourne une liste d'objets `SparseEmbedding` |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle |
| `info()` | `ModelDesc` | Descripteur du modèle |
| `close()` | `None` | Libère les ressources |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste les modèles sparse |

#### Exemple

```python
from libembedding import SparseTextEmbedding

sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
    # indices : tableau int32 des token IDs
    # values : tableau float32 des poids
```

---

### ImageEmbedding

Génère des embeddings denses à partir d'images.

#### Constructeur

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
    num_threads=None,  # déprécié
)
```

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `embed_files(paths, batch_size=None)` | `np.ndarray` | Embed à partir de chemins de fichiers. Forme : `(n, dim)` |
| `embed_bytes(images, batch_size=None)` | `np.ndarray` | Embed à partir de données bytes brutes (JPEG, PNG, etc.) |
| `dim` | `int` (property) | Dimension de l'embedding |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle |
| `info()` | `ModelDesc` | Descripteur du modèle |
| `close()` | `None` | Libère les ressources |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste les modèles image |

#### Exemple

```python
from libembedding import ImageEmbedding

model = ImageEmbedding()
embeddings = model.embed_files(["photo.jpg", "diagram.png"])
print(embeddings.shape)  # (2, 512)
```

---

### Reranker

Score et trie des documents par pertinence par rapport à une requête (cross-encoder).

#### Constructeur

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
    num_threads=None,  # déprécié
)
```

**Modèles disponibles :**

| Nom HuggingFace | Description |
|-----------------|-------------|
| `BAAI/bge-reranker-base` | BGE Reranker base (défaut) |
| `BAAI/bge-reranker-v2-m3` | BGE Reranker v2 multilingue |
| `jinaai/jina-reranker-v1-turbo-en` | Jina Reranker v1 turbo |
| `jinaai/jina-reranker-v2-base-multilingual` | Jina Reranker v2 multilingue |

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `rerank(query, documents, batch_size=0)` | `list[RerankResult]` | Retourne les résultats triés par score décroissant |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle |
| `info()` | `ModelDesc` | Descripteur du modèle |
| `close()` | `None` | Libère les ressources |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste les modèles reranker |

#### Exemple

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

## Types de données

### TextEmbeddingPool

Pool de sessions ONNX pour le parallélisme inter-sessions. Voir [performance_tuning.html](performance_tuning.html).

```python
@dataclass(frozen=True)
class TextEmbeddingPool:
    model_name: str
    workers: int = 0                  # nombre de sessions (0 = auto-detect)
    threads_per_worker: int = 1       # threads par session
    batch_size: int = 256
    provider: str = "cpu"
    offline: bool = False
    autotune: bool = False            # auto-tune tous les paramètres
    autotune_texts: list[str] = None  # corpus pour l'autotune
    autotune_max_samples: int = 100   # taille d'échantillon max
```

**Méthodes :**

| Membre | Type | Description |
|--------|------|-------------|
| `embed(texts)` | `np.ndarray` | Embed les textes en parallèle |
| `num_workers` | `int` (property) | Nombre de workers actifs |
| `dim` | `int` (property) | Dimension de l'embedding |
| `close()` | `None` | Libère les ressources |

### TuningResult

Résultat de l'autotune pour un modèle.

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

Résultat de la sélection automatique de modèle.

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

### Fonctions globales

| Fonction | Description |
|----------|-------------|
| `autotune(model_name, full=False)` | Auto-tune un modèle. Retourne `TuningResult`. |
| `auto_select_model(use_case="balanced")` | Sélectionne le meilleur modèle. Retourne `ModelSelectionResult`. |
| `clear_autotune_cache(model_name=None)` | Efface le cache d'autotune. |

---

### SparseEmbedding

```python
@dataclass(frozen=True)
class SparseEmbedding:
    indices: np.ndarray  # int32 — token IDs
    values:  np.ndarray  # float32 — poids des tokens
```

### RerankResult

```python
@dataclass(frozen=True)
class RerankResult:
    index: int   # index du document dans la liste d'entrée
    score: float # score de pertinence (plus élevé = plus pertinent)
```

### ModelInfo

```python
@dataclass(frozen=True)
class ModelInfo:
    model_name:    str   # ex: "BAAI/bge-small-en-v1.5"
    model_code:    str   # code HF repo, ex: "Xenova/bge-small-en-v1.5"
    model_file:    str   # ex: "onnx/model.onnx"
    description:   str
    dim:           int   # dimension de l'embedding
    max_tokens:    int   # longueur max en tokens
    pooling:       str   # "cls" ou "mean"
    quantization:  str   # "none", "static", "dynamic"
```

### ModelDesc

```python
@dataclass(frozen=True)
class ModelDesc:
    name:        str
    dimension:   int
    max_length:  int
    pooling:     str   # "cls" ou "mean"
    num_threads: int
    batch_size:  int
    provider:    str
    device_id:   int
```

---

## Gestion des erreurs

Toutes les exceptions héritent de `LembedError` :

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
)
```

**Hiérarchie :**

```
LembedError (Exception)
├── InvalidArgumentError   — argument NULL ou hors plage
├── OutOfMemoryError       — échec d'allocation
├── OnnxRuntimeError       — erreur ONNX Runtime
├── TokenizerError         — erreur de chargement/encodage du tokenizer
├── DownloadError          — échec de téléchargement
├── IOError                — erreur fichier
├── ModelNotFoundError     — modèle inconnu
├── UnsupportedError       — fonctionnalité désactivée à la compilation
└── BatchSizeError         — taille de batch incompatible
```

#### Exemple de gestion

```python
from libembedding import TextEmbedding
from libembedding.exceptions import ModelNotFoundError, OnnxRuntimeError

try:
    model = TextEmbedding("modele/inexistant")
except ModelNotFoundError as e:
    print(f"Modèle non trouvé : {e.detail}")
except OnnxRuntimeError as e:
    print(f"Erreur ONNX : {e.detail}")
except LembedError as e:
    print(f"Erreur libembedding [{e.status_code}] : {e.message}")
```

Chaque exception possède :
- `status_code` (`int`) — code d'erreur C
- `message` (`str`) — message générique (ex: `"ONNX Runtime error"`)
- `detail` (`str`) — détail technique (ex: message d'erreur ORT)

---

## Fonctions utilitaires

```python
import libembedding

# Lister les modèles par catégorie
libembedding.list_text_models()      # list[ModelInfo]
libembedding.list_sparse_models()    # list[ModelInfo]
libembedding.list_image_models()     # list[ModelInfo]
libembedding.list_reranker_models()  # list[ModelInfo]
```
