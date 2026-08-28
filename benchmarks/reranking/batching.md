# Reranking Benchmarks — Batching

## Objectif
Mesurer la sensibilité du reranking au batch_size.

Le reranking est souvent très sensible au batching. On risque d'avoir de grosses surprises.

## Méthodologie

- Modèle: `BAAI/bge-reranker-base`
- Corpus: 50 documents fixes
- batch_size: 1, 8, 16, 32
- Threads: auto (0)
- Warmup: 1 itération
- Itérations: 10
- Métriques: temps total, ms/doc, docs/sec

## Résultats attendus

| batch_size | Temps total (ms) | ms/doc | docs/sec |
|------------|-----------------|--------|----------|
| 1          | ?               | ?      | ?        |
| 8          | ?               | ?      | ?        |
| 16         | ?               | ?      | ?        |
| 32         | ?               | ?      | ?        |

## Hypothèse

Le batch_size optimal pour le reranking peut être très différent de l'embedding.

Embedding: batch_size=256 optimal
Reranking: batch_size=8 ou 16 peut être optimal

## Commandes

```bash
cd benchmarks/reranking
python bench_batching.py --model BAAI/bge-reranker-base --docs 50 --batch-sizes 1,8,16,32
```
