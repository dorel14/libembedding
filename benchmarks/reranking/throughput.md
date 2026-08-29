# Reranking Benchmarks — Throughput

## Objectif
Mesurer le throughput en docs/seconde pour différentes configurations de threads.

## Méthodologie

- Modèle: `BAAI/bge-reranker-base`
- Corpus: 50 documents
- Batch size: 8 (optimal)
- Threads: 1, 2, 4, 8
- Warmup: 1 itération
- Itérations: 5
- Métriques: docs/sec, ms/doc

## Résultats (batch_size=8, 50 docs)

| Threads | ms/doc | docs/sec |
|---------|--------|----------|
| 1       | 213.1  | 4.7      |
| 2       | 119.9  | 8.3      |
| 4       | 93.1   | 10.7     |
| 8       | 209.9  | 4.8      |

## Résultats (batch_size=256, 50 docs)

| Threads | ms/doc | docs/sec |
|---------|--------|----------|
| 1       | 261.9  | 3.8      |
| 2       | 166.4  | 6.0      |
| 4       | 98.3   | 10.2     |
| 8       | 162.6  | 6.1      |

## Analyse

### 1. Le gain 1→4 threads est significatif
- 1→2 threads: **47% de gain** (213→120 ms/doc)
- 2→4 threads: **22% de gain** (120→93 ms/doc)
- Scaling quasi-linéaire jusqu'à 4 threads

### 2. 8 threads = oversubscription
- **Dégradation de 54%** vs 4 threads (93→210 ms/doc)
- La machine (probablement 4c/8t) souffre de contention CPU
- Le scheduling ONNX Runtime n'est pas optimal au-delà du nombre de cœurs physiques

### 3. batch_size=8 vs 256
- Légère supériorité de batch_size=8 (93 vs 98 ms/doc à 4 threads)
- Différence modeste mais cohérente avec batching.md

## Recommandation

**Configuration optimale: 4 threads, batch_size=8** → ~93 ms/doc, ~10.7 docs/sec

```bash
reranker = Reranker("BAAI/bge-reranker-base", threads=4, batch_size=8)
```

## Commandes

```bash
cd benchmarks/reranking
python bench_throughput.py --models BAAI/bge-reranker-base --docs 50 --threads 1,2,4,8 --batch-size 8
```
