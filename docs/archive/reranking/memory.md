# Reranking Benchmarks — Memory

## Objectif
Mesurer la consommation mémoire pour le modèle de reranker.

## Méthodologie

- Modèle: `BAAI/bge-reranker-base`
- Mesurer peak RSS après chargement modèle
- Mesurer peak RSS pendant inference (50 docs, batch_size=8)
- Threads: 4
- Platform: Windows (WorkingSetSize via wmic)

## Résultats

| Modèle | RAM après load (MB) | RAM pendant inference (MB) | Delta (MB) |
|--------|---------------------|---------------------------|------------|
| BGE-base | 1138 | 1283 | 145 |

## Analyse

### 1. Empreinte modèle (~1.1 GB)
- Le fichier ONNX `onnx/model.onnx` fait ~50 MB
- L'runtime ONNX charge le graphe + poids en mémoire
- Le tokenizer HuggingFace (vocab + merges) ajoute ~100 MB
- **Total après load: ~1.1 GB** — cohérent avec un modèle base (~100M params)

### 2. Delta inference (+145 MB)
- Allocation des tenseurs d'entrée (batch × seq_len × sizeof(int64))
- Allocation des tenseurs de sortie (logits)
- Buffers internes ONNX Runtime
- **Pic à 1.28 GB** pour 50 docs en batch_size=8

### 3. Comparaison avec l'embedding

| Type | RAM load (MB) | RAM/doc (KB) |
|------|--------------|--------------|
| Embedding (MiniLM) | ~250 | ~5 |
| Reranking (BGE-base) | ~1138 | ~2900 |

**Le reranking consomme ~4.5x plus de RAM que l'embedding.**

C'est critique pour les déploiements multi-modèles ou les environnements contraints (serverless, edge).

## Impact pour Whoosh-NG

### Scénario 1: Embedding + Reranking simultanés
```
Embedding model:  ~250 MB
Reranker model:   ~1138 MB
Total:            ~1388 MB (+ overhead applicatif)
```

### Scénario 2: Reranking seul
```
Reranker model:   ~1138 MB
Total:            ~1200 MB (+ overhead)
```

### Scénario 3: Plusieurs workers (futur session pool)
```
1 worker:   ~1138 MB
2 workers:  ~2276 MB
4 workers:  ~4552 MB
```

**Attention**: multiplier les workers multiplie la RAM linéairement.

## Commandes

```bash
cd benchmarks/reranking
python bench_memory.py --models BAAI/bge-reranker-base --docs 50 --threads 4
```

## Recommandations

1. **Charger le reranker à la demande** — ne pas le garder en mémoire si pas utilisé
2. **Libérer explicitement** — `reranker.close()` pour libérer la RAM
3. **Session pool conditionnel** — ne pas implémenter tant que la RAM le permet
4. **Modèles légers** — envisager des modèles plus petits (distillation, quantization)
