# Reranking Benchmarks — Realistic Documents with P50/P95/P99

## Objectif
Mesurer la latence réelle du reranking avec des documents de longueures réalistes.
**Le P95 est la métrique critique pour les SLA de production.**

## Méthodologie

- Modèle: `jinaai/jina-reranker-v1-turbo-en`
- Threads: 4, Batch: 8
- Warmup: 2 itérations
- Itérations: 30
- Métriques: P50 (médiane), P95, P99, Moyenne
- Documents: longueurs réalistes (20-500 tokens)

## Résultats

### P50 (médiane)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 20 | 46ms | 94ms | 189ms |
| 50 | 83ms | 172ms | 337ms |
| 100 | 155ms | 302ms | 587ms |
| 200 | 299ms | 594ms | 1193ms |
| 500 | 876ms | 1828ms | 3526ms |

### P95 (SLA production)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 20 | 56ms | 108ms | 225ms |
| 50 | 104ms | 203ms | 414ms |
| 100 | 179ms | 381ms | 710ms |
| 200 | 363ms | 711ms | 1342ms |
| 500 | 933ms | 2221ms | 3914ms |

### P99 (pire cas)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 20 | 57ms | 112ms | 240ms |
| 50 | 106ms | 241ms | 522ms |
| 100 | 185ms | 407ms | 864ms |
| 200 | 831ms | 780ms | 1351ms |
| 500 | 936ms | 2284ms | 4151ms |

## Analyse

### 1. Queue lourde significative

| tokens/doc | top_k=5 | P50→P99 ratio |
|------------|---------|---------------|
| 20 | 46ms → 57ms | 1.2x |
| 100 | 155ms → 185ms | 1.2x |
| 200 | 299ms → 831ms | **2.8x** |
| 500 | 876ms → 936ms | 1.1x |

**Le P99 peut être jusqu'à 2.8x le P50** pour les documents longs. C'est critique pour les SLA.

### 2. Recommandations SLA

| Budget P95 | Configuration |
|------------|--------------|
| 100ms | top_k=5, 50 tokens |
| 200ms | top_k=10, 50 tokens |
| 500ms | top_k=10, 100 tokens |
| 1000ms | top_k=5, 500 tokens |

### 3. Implication pour Whoosh-NG

**Troncature à 100 tokens recommandée**:
- P95 top_k=10: 381ms (acceptable)
- P99 top_k=10: 407ms (queue lourde modérée)

**Documents 200+ tokens**:
- P95 top_k=5: 363ms (OK)
- P99 top_k=5: 831ms (queue lourde!)

## Commandes

```bash
cd benchmarks/reranking
python bench_realistic.py --lengths 20,50,100,200,500 --top-k 5,10,20
```
