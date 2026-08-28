# Reranking Benchmarks — Recommendations

## Objectif
Synthétiser les résultats des benchmarks pour orienter la roadmap et l'API Whoosh-NG.

## Questions à répondre

### 1. Quel est le coût réel du reranking ?

| Taille corpus | Embedding (ms) | Reranking (ms) | Total (ms) |
|---------------|----------------|----------------|------------|
| 10 docs       | ?              | ?              | ?          |
| 20 docs       | ?              | ?              | ?          |
| 50 docs       | ?              | ?              | ?          |
| 100 docs      | ?              | ?              | ?          |
| 200 docs      | ?              | ?              | ?          |

### 2. Quel modèle choisir ?

| Critère       | MiniLM-L6-v2 | BGE-base | BGE-large |
|---------------|--------------|----------|-----------|
| ms/doc        | ?            | ?        | ?         |
| RAM           | ?            | ?        | ?         |
| Qualité       | ?            | ?        | ?         |
| **Recommandation** | ?        | ?        | ?         |

### 3. Quel batch_size pour le reranking ?

| batch_size | docs/sec | ms/doc | Recommandation |
|------------|----------|--------|----------------|
| 1          | ?        | ?      | ?              |
| 8          | ?        | ?      | ?              |
| 16         | ?        | ?      | ?              |
| 32         | ?        | ?      | ?              |

### 4. Workers : 1 session × N threads ou N sessions × 1 thread ?

| Config | Docs/s | Recommandation |
|--------|--------|----------------|
| 1x4    | ?      | ?              |
| 2x2    | ?      | ?              |
| 4x1    | ?      | ?              |
| 8x1    | ?      | ?              |

### 5. Quelle est la meilleure stratégie de pipeline ?

Comparer :
```
BM25 → top 100 → Dense → top 20 → Reranker
```
vs
```
BM25 → top 100 → Reranker
```

**Hypothèse**: le reranker n'a besoin de voir que top 10-30 documents pour atteindre presque la même qualité.

## Impact Whoosh-NG

### Avant benchmarks
```python
searcher.search(query, limit=100, rerank_top_k=100)
```

### Après benchmarks (si coût explose)
```python
searcher.search(query, limit=100, rerank_top_k=20)
```

### API recommandée
```python
searcher.search(
    query,
    limit=100,           # BM25 recall
    rerank_top_k=20,     # Reranker precision (à ajuster selon benchmarks)
    reranker_model="BAAI/bge-reranker-base",
    reranker_batch_size=8,
)
```

## Prochaines étapes

1. Exécuter tous les benchmarks sur i7-1065G7
2. Comparer résultats avec fastembed / sentence-transformers
3. Ajuster `rerank_top_k` par défaut dans Whoosh-NG selon les résultats
4. Si workers améliorent significativement : implémenter session pool pour reranker
5. Documenter les recommendations dans README
