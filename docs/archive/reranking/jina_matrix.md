# Reranking Benchmarks — Jina-v1-turbo Matrix

## Objectif
Matrice complète: longueur des documents × top_k pour Jina-v1-turbo-en.

**C'est le benchmark le plus utile pour construire un "budget reranking" dans Whoosh-NG.**

## Méthodologie

- Modèle: `jinaai/jina-reranker-v1-turbo-en`
- Threads: 4
- Batch size: 8
- Warmup: 1 itération
- Itérations: 5
- Métrique: Médiane

## Résultats

### Matrice temps total (ms)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 32 | 78 | 143 | 285 |
| 128 | 233 | 454 | 920 |
| 512 | 1116 | 2173 | 4440 |
| 2048 | 9911 | 19217 | 38350 |

### Matrice ms/doc

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 32 | 15.5 | 14.3 | 14.3 |
| 128 | 46.6 | 45.4 | 46.0 |
| 512 | 223.1 | 217.3 | 222.0 |
| 2048 | 1982.1 | 1921.7 | 1917.5 |

### Matrice UX

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 32 | Excellent | Excellent | Good |
| 128 | Good | Good | OK |
| 512 | Slow | Slow | Too slow |
| 2048 | Too slow | Too slow | Too slow |

## Budget Calculator

| Budget max | Configuration optimale |
|------------|----------------------|
| 200ms | top_k=10, 32 tokens/doc |
| 500ms | top_k=20, 32 tokens/doc |
| 1000ms | top_k=20, 128 tokens/doc |
| 2000ms | top_k=20, 128 tokens/doc |
| 5000ms | top_k=20, 512 tokens/doc |

## Recommandation

**Troncature à 128 tokens**:
- top_k=10: 454ms (Good UX)
- top_k=20: 920ms (OK UX)

**Sans troncature (512 tokens)**:
- top_k=5: 1116ms (deja lent)

## Commandes

```bash
cd benchmarks/reranking
python bench_jina_matrix.py --lengths 32,128,512,2048 --top-k 5,10,20
```
