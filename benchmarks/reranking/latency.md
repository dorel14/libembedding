# Reranking Benchmarks — Latency

## Objectif
Mesurer le coût absolu du reranking pour différentes tailles de corpus.

C'est le premier chiffre que les utilisateurs veulent connaître.

## Méthodologie

- Modèle par défaut: `BAAI/bge-reranker-base`
- Requête fixe: `"What is deep learning?"`
- Documents: corpus synthétique de tailles variables
- Threads: 4 (optimal trouvé dans workers.md)
- Batch size: 8 (optimal trouvé dans batching.md)
- Warmup: 1 itération
- Itérations: 5
- Médiane + p95

## Corpus de test

| Taille | Description |
|--------|-------------|
| 10 docs | Petit contexte |
| 20 docs | Petit contexte |
| 50 docs | Contexte moyen |
| 100 docs | Contexte standard |
| 200 docs | Gros contexte |

## Résultats (4 threads, batch_size=8)

| Docs | Temps (ms) | ms/doc | docs/sec |
|------|-----------|--------|----------|
| 10   | 1100      | 110.0  | 9.1      |
| 20   | 2549      | 127.5  | 7.8      |
| 50   | 5693      | 113.9  | 8.8      |
| 100  | 12176     | 121.8  | 8.2      |
| 200  | 20940     | 104.7  | 9.6      |

## Résultats (1 thread, batch_size=256)

| Docs | Temps (ms) | ms/doc | docs/sec |
|------|-----------|--------|----------|
| 10   | 2336      | 233.6  | 4.3      |
| 20   | 5109      | 255.5  | 3.9      |
| 50   | 13095     | 261.9  | 3.8      |

## Hypothèse vs Réalité

| Métrique | Hypothèse | Réalité (4t) | Réalité (1t) |
|----------|-----------|-------------|-------------|
| ms/doc   | 20-50     | 110-127     | 233-262     |

**Le reranker est ~10x plus cher que l'embedding** (embedding ≈ 10 ms/doc, reranking ≈ 110 ms/doc en config optimale).

L'hypothèse 20-50 ms/doc était optimiste — elle supposait probablement un GPU ou un modèle plus léger.

## Commandes

```bash
cd benchmarks/reranking
python bench_latency.py --model BAAI/bge-reranker-base --docs 10,20,50,100,200 --threads 4 --batch-size 8
```

## Analyse

- **Scalabilité linéaire**: le temps total grossit proportionnellement au nombre de docs
- **ms/doc stable**: ~110-127 ms/doc quelle que soit la taille (pas d'effet de batch significatif)
- **Variance**: le ms/doc à 20 docs (127.5) est légèrement plus élevé qu'à 200 docs (104.7) — l'overhead de session est mieux amorti sur de gros corpus
