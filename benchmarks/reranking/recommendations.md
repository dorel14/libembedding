# Reranking Benchmarks — Recommendations

## Objectif
Synthétiser les résultats des benchmarks pour orienter la roadmap et l'API Whoosh-NG.

## Avertissement

> **The reranker is approximately 10x slower than embedding generation and should only be applied to a small candidate set.**

## Modèle par défaut recommandé

> **jinaai/jina-reranker-v1-turbo-en-quantized** (INT8) — 5× plus rapide que BGE-base, 1.91× plus rapide que Jina FP32

| Modèle | ms/doc | docs/s | RAM (MB) | Max tokens | Usage |
|--------|--------|--------|----------|------------|-------|
| **jinaai/jina-reranker-v1-turbo-en-quantized** | **20.3** | **49.3** | **~235** | 8192 | **Anglais (défaut)** |
| jinaai/jina-reranker-v1-turbo-en | 38.7 | 25.8 | ~378 | 8192 | Anglais (FP32) |
| jinaai/jina-reranker-v2-base-multilingual | 105.3 | 9.5 | ~500 | 8192 | Multilingue |
| BAAI/bge-reranker-base | 122.0 | 8.2 | ~1138 | 512 | Compatibilité |
| BAAI/bge-reranker-v2-m3 | 393.8 | 2.5 | ~500 | 512 | Multilingue lent |

### Quantification: FP32 vs INT8

| Metric | FP32 | INT8 | Gain |
|--------|------|------|------|
| ms/doc | 38.7 | 20.3 | **1.91×** |
| docs/s | 25.8 | 49.3 | **1.91×** |
| RAM delta | 189 MB | 45 MB | **4.24×** |
| Score correlation (Pearson) | — | 0.874 | Qualité préservée |

## Recommandation d'usage

### Par cas d'usage

| Usage | top_k recommande | Latence (Jina, 128 tokens) |
|-------|-----------------|---------------------------|
| Recherche interactive | 5-10 | 233-454ms |
| Recherche avancee | 20 | ~920ms |
| Batch offline | 50+ | ~2.3s+ |

### Matrice budget reranking (Jina-v1-turbo)

**Configuration**: threads=4, batch=8, iterations=5

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 32 | 78ms | 143ms | 285ms |
| 128 | 233ms | 454ms | 920ms |
| 512 | 1116ms | 2173ms | 4440ms |
| 2048 | 9911ms | 19217ms | 38350ms |

### Budget calculator

| Budget max | Configuration optimale |
|------------|----------------------|
| 200ms | top_k=10, 32 tokens/doc |
| 500ms | top_k=20, 32 tokens/doc |
| 1000ms | top_k=20, 128 tokens/doc |
| 2000ms | top_k=20, 128 tokens/doc |
| 5000ms | top_k=20, 512 tokens/doc |

### Consequence: tronquer les documents longs

Un document de 512 tokens coute **12x plus cher** qu'un document de 32 tokens. **La troncature a 128-256 tokens est recommandee** pour les cas sensibles a la latence.

## Architecture optimale pour Whoosh-NG

```
                 QUERY
                   |
          +--------+--------+
          |                 |
        BM25              Vector
          |                 |
          +--------+--------+
                   |
              Top 50-100
                   |
         Candidate Filter
         (truncate 128 tok)
                   |
              Top 10-20
                   |
               Reranker
                   |
                 Top 5
                   |
                RESULT
```

**Principe**: le retrieval doit maximiser le recall a faible cout ; le reranker doit etre utilise comme une ressource co�teuse et limitee.

## Synthese des resultats

### Configuration optimale trouvee

```python
reranker = Reranker(
    "jinaai/jina-reranker-v1-turbo-en",  # 5x plus rapide que BGE
    threads=4,          # 4 coeurs physiques
    batch_size=8,       # petit batch = meilleur pour cross-encoder
)
```

| Metrique | Valeur (Jina) | Valeur (BGE) |
|----------|---------------|--------------|
| ms/doc (128 tok) | ~46 | ~170 |
| docs/s | ~40 | ~8 |
| RAM | ~1.1 GB | ~1.1 GB |
| Load time | ~10s | ~6s |
| Max tokens | 8192 | 512 |

## Impact Whoosh-NG

### Avant benchmarks
```python
searcher.search(query, limit=100, rerank_top_k=100)
# Cout: ~10 secondes (BGE) -- inacceptable
```

### Apres benchmarks (recommandation)
```python
searcher.search(
    query,
    limit=100,               # BM25 recall
    rerank_top_k=10,         # Reranker precision
    reranker_model="jinaai/jina-reranker-v1-turbo-en-quantized",  # INT8: 1.91x plus rapide
    reranker_threads=4,
    reranker_batch_size=8,
    reranker_max_length=128,  # Troncature
)
# Cout: ~450ms (Jina, 128 tok) -- acceptable
```

### API recommandee
```python
searcher.search(
    query,
    retrieval_top_k=100,         # BM25/Vector recall
    rerank_top_k=10,             # Reranker precision
    reranker_model="jinaai/jina-reranker-v1-turbo-en-quantized",  # INT8: 1.91x plus rapide
    reranker_threads=4,
    reranker_batch_size=8,
    reranker_max_length=128,     # Troncature des documents
    latency_budget_ms=1000,      # Budget automatique
)
```

## Decisions cles

| Decision | Choix | Justification |
|----------|-------|---------------|
| **`reranker_model` par defaut** | **jinaai/jina-reranker-v1-turbo-en-quantized** | **INT8: 1.91× plus rapide, 4.24× moins de RAM** |
| `rerank_top_k` par defaut | **10** | Bon compromis qualite/vitesse |
| `reranker_threads` par defaut | **4** | Optimal sur machine 4c/8t |
| `reranker_batch_size` par defaut | **8** | Optimal pour cross-encoder |
| `reranker_max_length` | **128 tokens** | Troncature pour maitriser la latence |
| Session pool reranker | **Non** | Pas justifie (oversubscription, RAM xN) |
| Chargement modele | **Lazy** | 1.1 GB de RAM |
| **Separation des mondes** | **Oui** | Embedding (throughput, workers, batch 256) != Reranker (latence, 1 session, batch 8) |
| **AutoTuner separe** | **Oui** | `EmbeddingAutoTune()` + `RerankerAutoTune()` |
| **Pipeline hybride** | **BM25 -> Dense -> truncate -> top 10 -> Reranker** | ~1.3s total (Jina) |

## Prochaines etapes

### Phase Investigation COMPLéTÉE

**Resultat: le goulot est le MODèLE, pas le code.**

1. ✅ Benchmarks complétés sur BGE-base
2. ✅ Benchmark top_k (le plus utile pour Whoosh-NG)
3. ✅ **Model comparison**: Jina-v1-turbo **5x plus rapide** que BGE-base
4. ✅ **Document length**: scaling linéaire avec tokens
5. ✅ **Profiling estimation**: 97% en inference ONNX
6. ✅ **ORT baseline**: 87.7 ms/doc avec settings actuels
7. ✅ **Matrice Jina × longueur × top_k**: budget reranking complet

### Phase Optimisation (restante)

8. ⬜ Tester quantification (QInt8/UInt8 vs FP32)
9. ⬜ Exposer les settings ORT (10-30% potentiel)
10. ⬜ Intégrer `rerank_top_k=10` par défaut dans Whoosh-NG
11. ⬜ Implémenter lazy loading du reranker
12. ⬜ Implémenter `EmbeddingAutoTune()` et `RerankerAutoTune()` séparés
13. ⬜ Implémenter `EmbeddingPool` (workers pour embedding seulement)
14. ⬜ Documenter les recommendations dans README

### Ce qu'on NE fera PAS

- ❌ Optimiser le tokenizer (estimation: 0.3% du temps)
- ❌ Réécrire le post-processing (estimation: 0.01% du temps)
- ❌ Optimiser les copies mémoire (estimation: 1% du temps)
- ❌ Session pool pour reranker (oversubscription prouvée)

## Fichiers de benchmark

| Fichier | Contenu | Statut |
|---------|---------|--------|
| `bench_latency.py` | Cout absolu par taille de corpus | Complete |
| `bench_throughput.py` | Throughput par threads | Complete |
| `bench_batching.py` | Sensibilite au batch_size | Complete |
| `bench_memory.py` | RAM par modele | Complete |
| `bench_workers.py** | Sensibilite au parallelisme | Complete |
| `bench_topk.py** | Cout vs top_k (UX) | Complete |
| `bench_models.py** | Comparaison MiniLM/BGE/Jina | Complete |
| `bench_length.py** | Sensibilite longueur documents | Complete |
| `bench_jina_matrix.py** | Matrice Jina x longueur x top_k | Complete |
| `bench_common.py` | Utilitaires partages | Cree |
| `investigation.md` | Plan et resultats | Mis a jour |
