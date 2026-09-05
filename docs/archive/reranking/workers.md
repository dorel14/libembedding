# Reranking Benchmarks — Workers

## Objectif
Tester l'analogie avec l'embedding : est-ce que le Cross Encoder réagit de la même façon au parallélisme ?

Pour l'embedding, on a montré que N sessions × 1 thread était souvent plus efficace qu'une session × N threads.

## Méthodologie

- Modèle: `BAAI/bge-reranker-base`
- Corpus: 50 documents
- Configurations:

| Config | Workers | Threads/worker | Total threads |
|--------|---------|----------------|---------------|
| 1x1    | 1       | 1              | 1             |
| 1x2    | 1       | 2              | 2             |
| 1x4    | 1       | 4              | 4             |
| 1x8    | 1       | 8              | 8             |

- batch_size: 8 (optimal trouvé dans batching.md)
- Warmup: 1 itération
- Itérations: 5
- Métriques: docs/sec, ms/doc, RAM

## Résultats (batch_size=8, 50 docs)

| Config | Docs/s | ms/doc | RAM (MB) |
|--------|--------|--------|----------|
| 1x1    | 9.6    | 104.4  | 1138     |
| 1x2    | 10.1   | 99.1   | 1138     |
| 1x4    | 10.1   | 99.1   | 1138     |
| 1x8    | 4.8    | 209.9  | 1138     |

## Résultats (batch_size=256, 20 docs)

| Config | Docs/s | ms/doc | RAM (MB) |
|--------|--------|--------|----------|
| 1x1    | 3.9    | 256.9  | -        |
| 1x2    | 6.0    | 166.4  | -        |
| 1x4    | 10.2   | 98.3   | -        |
| 1x8    | 6.1    | 162.6  | -        |

## Hypothèse vs Réalité

| Hypothèse | Réalité |
|-----------|---------|
| Le Cross Encoder réagit différemment | **Oui, mais pas comme attendu** |
| Plus de sessions = plus de RAM | **Non testé (pas de session pool)** |
| Le batch_size joue un rôle plus important | **Oui, batch_size=8 est optimal** |

## Analyse

### 1. 4 threads = sweet spot
- **99 ms/doc** — optimal
- Au-delà, l'oversubscription CPU dégrade les perfs
- Le cross-encoder est plus sensible à la contention que l'embedding

### 2. 8 threads = contre-productif
- **210 ms/doc** — pire que 1 thread!
- Cause: la machine a probablement 4 cœurs physiques / 8 logiques
- L'hyperthreading ne double pas la puissance de calcul
- ONNX Runtime crée plus de threads que nécessaire

### 3. Comparaison embedding vs reranking

| Métrique | Embedding (MiniLM) | Reranking (BGE-base) |
|----------|-------------------|---------------------|
| ms/doc optimal | ~10 | ~99 |
| Threads optimal | 4 | 4 |
| Oversubscription à | 8+ | 8+ |
| batch_size optimal | 256 | 8 |

**Même pattern de scaling**, mais le reranking est ~10x plus lent.

### 4. Session pool (non implémenté)

Le session pool pour rerankers n'existe pas encore. Les prédictions:

| Config | Docs/s estimé | RAM estimé |
|--------|--------------|------------|
| 1x4    | 10.1         | 1138 MB    |
| 2x2    | ~15-18       | ~2276 MB   |
| 4x1    | ~25-30       | ~4552 MB   |

**Estimation**: 4 sessions × 1 thread pourrait donner ~30 docs/s, mais au prix de 4.5 GB de RAM.

## Recommandation

**Pour l'instant: 1 session × 4 threads × batch_size=8.**

Ne pas implémenter de session pool tant que:
1. Le benchmark ne montre pas un gain ≥ 10% (cf. roadmap Task 18)
2. La RAM disponible le permet
3. Le cas d'usage justifie la complexité

## Commandes

```bash
cd benchmarks/reranking
python bench_workers.py --model BAAI/bge-reranker-base --docs 50 --total-threads 1,2,4,8 --batch-size 8
```

## Dépendances

- Nécessite `EmbeddingPool` / session pool (Task 18, conditionnel)
- Si session pool non implémenté, test avec `threads` uniquement
