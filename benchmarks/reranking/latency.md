# Reranking Benchmarks — Latency

## Objectif
Mesurer le coût absolu du reranking pour différentes tailles de corpus.

C'est le premier chiffre que les utilisateurs veulent connaître.

## Méthodologie

- Modèle par défaut: `BAAI/bge-reranker-base`
- Requête fixe: `"What is deep learning?"`
- Documents: corpus synthétique de tailles variables
- Threads: 1 (mesure pure du modèle)
- Warmup: 1 itération
- Itérations: 10
- Médiane + p95

## Corpus de test

| Taille | Description |
|--------|-------------|
| 10 docs | Petit contexte |
| 20 docs | Petit contexte |
| 50 docs | Contexte moyen |
| 100 docs | Contexte standard |
| 200 docs | Gros contexte |

## Résultats attendus

| Docs | Temps (ms) | ms/doc |
|------|-----------|--------|
| 10   | ?         | ?      |
| 20   | ?         | ?      |
| 50   | ?         | ?      |
| 100  | ?         | ?      |
| 200  | ?         | ?      |

## Hypothèse

Le reranker coûte ≈ 20-50 ms/document (beaucoup plus cher que l'embedding ≈ 4-10 ms/doc).

## Commandes

```bash
cd benchmarks/reranking
python bench_latency.py --model BAAI/bge-reranker-base --docs 10,20,50,100,200
```
