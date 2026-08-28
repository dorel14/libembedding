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
| 1x4    | 1       | 4              | 4             |
| 2x2    | 2       | 2              | 4             |
| 4x1    | 4       | 1              | 4             |
| 8x1    | 8       | 1              | 8             |

- batch_size: optimal trouvé dans `batching.md`
- Warmup: 1 itération
- Itérations: 10
- Métriques: docs/sec, ms/doc, CPU%, RAM

## Résultats attendus

| Config | Docs/s | ms/doc | CPU% | RAM (MB) |
|--------|--------|--------|------|----------|
| 1x4    | ?      | ?      | ?    | ?        |
| 2x2    | ?      | ?      | ?    | ?        |
| 4x1    | ?      | ?      | ?    | ?        |
| 8x1    | ?      | ?      | ?    | ?        |

## Hypothèse

Le Cross Encoder pourrait réagir différemment que l'embedding :
- Le modèle est plus petit (plus de sessions = plus de mémoire)
- Le batch_size joue peut-être un rôle plus important
- Le overhead de plusieurs sessions pourrait être plus pénalisant

## Commandes

```bash
cd benchmarks/reranking
python bench_workers.py --model BAAI/bge-reranker-base --docs 50 --total-threads 4,8
```

## Dépendances

- Nécessite `EmbeddingPool` / session pool (Task 12, conditionnel)
- Si session pool non implémenté, test avec `threads` uniquement
