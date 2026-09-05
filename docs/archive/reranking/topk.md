# Reranking Benchmarks — top_k

## Objectif
Mesurer le coût réel du reranking selon le nombre de candidats (top_k).

**C'est le benchmark le plus utile pour les utilisateurs de Whoosh-NG.**

Il répond directement à la question: *"Combien de documents puis-je reranker avant que l'expérience utilisateur ne devienne trop lente ?"*

## Méthodologie

- Modèle: `BAAI/bge-reranker-base`
- Requête: `"What is deep learning?"`
- top_k: 5, 10, 20, 50, 100
- Threads: 4 (optimal)
- Batch size: 8 (optimal)
- Warmup: 1 itération
- Itérations: 10
- Médiane

## Résultats

| top_k | temps (ms) | ms/doc | docs/sec | UX |
|-------|-----------|--------|----------|-----|
| 5     | 423       | 84.6   | 11.8     | **Bon** |
| 10    | 802       | 80.2   | 12.5     | **OK** |
| 20    | 1746      | 87.3   | 11.5     | Lent |
| 50    | 4352      | 87.0   | 11.5     | Trop lent |
| 100   | 10205     | 102.1  | 9.8      | Trop lent |

## Analyse

### 1. Sweet spot UX: top_k=5-10
- **5 docs: 423ms** — impression de instantané
- **10 docs: 802ms** — acceptable, l'utilisateur attend sans frustration
- **20 docs: 1.75s** — l'utilisateur remarque le délai
- **50+ docs: 4.3s+** — inacceptable pour du temps réel

### 2. ms/doc stable
- ~80-102 ms/doc quelle que soit la taille
- Pas d'effet de batch significatif (le bottleneck est le forward pass cross-encoder)

### 3. Seuils UX reconnus

| Seuil | Perception | top_k max |
|-------|------------|-----------|
| < 500ms | Instantané | ~5 |
| 500ms-1s | Rapide | ~10 |
| 1s-2s | Perceptible | ~20 |
| 2s-5s | Lent | ~50 |
| > 5s | Inacceptable | > 80 |

## Recommandation Whoosh-NG

### Valeur par défaut
```python
rerank_top_k=10  # 802ms — bon compromis qualité/vitesse
```

### Selon le contexte
```python
# Recherche interactive (temps réel)
rerank_top_k=5   # 423ms — instantané

# Recherche standard
rerank_top_k=10  # 802ms — rapide

# Recherche approfondie (batch, offline)
rerank_top_k=20  # 1.75s — acceptable si pas interactif
```

### Ce qu'il NE faut PAS faire
```python
# ERREUR: rerank_top_k=100 coûte 10.2 secondes!
rerank_top_k=100  # INACCEPTABLE
```

## Pipeline Hybride Recommandé

```
BM25 → top 100 → Dense Embedding → top 10 → Reranker
```

Coût total estimé:
- BM25: ~5ms
- Dense × 100: ~1000ms
- Rerank × 10: ~802ms
- **Total: ~1.8s**

Alternative plus rapide:
```
BM25 → top 50 → Dense Embedding → top 5 → Reranker
```
- BM25: ~5ms
- Dense × 50: ~500ms
- Rerank × 5: ~423ms
- **Total: ~928ms**

## Commandes

```bash
cd benchmarks/reranking
python bench_topk.py --model BAAI/bge-reranker-base --top-k 5,10,20,50,100 --threads 4 --batch-size 8
```
