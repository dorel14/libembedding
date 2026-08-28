# Reranking Benchmarks — Throughput

## Objectif
Mesurer le throughput en docs/seconde pour différentes tailles de corpus.

## Méthodologie

- Modèles: `cross-encoder/ms-marco-MiniLM-L6-v2`, `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`
- Threads: 1, 2, 4, 8
- Warmup: 1 itération
- Itérations: 10
- Métriques: docs/sec, ms/doc

## Résultats attendus

| Modèle | ms/doc | RAM (MB) |
|--------|--------|----------|
| MiniLM-L6-v2 | ? | ? |
| BGE-base | ? | ? |
| BGE-large | ? | ? |

## Commandes

```bash
cd benchmarks/reranking
python bench_throughput.py --models MiniLM-L6-v2,BGE-base,BGE-large
```
