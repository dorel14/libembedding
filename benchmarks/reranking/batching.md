# Reranking Benchmarks — Batching

## Objectif
Mesurer la sensibilité du reranking au batch_size.

Le reranking est souvent très sensible au batching. On risque d'avoir de grosses surprises.

## Méthodologie

- Modèle: `BAAI/bge-reranker-base`
- Corpus: 50 documents fixes
- batch_size: 1, 8, 16, 32
- Threads: 4 (optimal)
- Warmup: 1 itération
- Itérations: 5
- Métriques: temps total, ms/doc, docs/sec

## Résultats (4 threads, 50 docs)

| batch_size | Temps total (ms) | ms/doc | docs/sec |
|------------|-----------------|--------|----------|
| 1          | 5221            | 104.4  | 9.6      |
| 8          | 4954            | 99.1   | 10.1     |
| 16         | 6604            | 132.1  | 7.6      |
| 32         | 5120            | 102.4  | 9.8      |

## Hypothèse vs Réalité

| Métrique | Hypothèse embedding | Réalité reranking |
|----------|--------------------|--------------------|
| batch_size optimal | 256 | **8** |
| Comportement | Plus grand = plus rapide | **Plus grand = plus lent** |

## Analyse

### 1. batch_size=8 est optimal
- **99.1 ms/doc** — le meilleur compromis throughput/latency
- 5% plus rapide que batch_size=1
- 25% plus rapide que batch_size=16

### 2. batch_size=16 est le PIRE
- **132.1 ms/doc** — dégradation de 33% vs batch_size=8
- Cause probable: les paires query+doc dépassent la taille du cache L2/L3
- Chaque paire fait jusqu'à 512 tokens → 16 paires = 8192 tokens en parallèle

### 3. Comparaison avec l'embedding

| Type | batch_size optimal | ms/doc |
|------|-------------------|--------|
| Embedding (MiniLM) | 256 | ~10 |
| Reranking (BGE-base) | 8 | ~99 |

**Le reranking a un batch_size optimal 32x plus petit que l'embedding.**

C'est contre-intuitif mais s'explique:
- L'embedding traite des textes courts indépendants
- Le reranking traite des paires query+doc (2x plus longs)
- Chaque forward pass cross-encoder est plus coûteux
- La pression mémoire augmente quadratiquement avec le batch

## Recommandation

**batch_size=8 par défaut pour le reranking.**

```python
reranker = Reranker("BAAI/bge-reranker-base", batch_size=8)
```

## Commandes

```bash
cd benchmarks/reranking
python bench_batching.py --model BAAI/bge-reranker-base --docs 50 --batch-sizes 1,8,16,32 --threads 4
```
