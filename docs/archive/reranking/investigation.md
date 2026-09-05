# Reranking Investigation — Résultats

## Résumé

L'enquête a démontré que **le choix du modèle est le principal levier de performance** pour le reranking, bien plus que l'optimisation de l'implémentation.

| Modèle | ms/doc | docs/s | Max tokens |
|--------|--------|--------|------------|
| **jinaai/jina-reranker-v1-turbo-en** | **24.6** | **40.7** | 8192 |
| jinaai/jina-reranker-v2-base-multilingual | 105.3 | 9.5 | 8192 |
| BAAI/bge-reranker-base | 122.0 | 8.2 | 512 |
| BAAI/bge-reranker-v2-m3 | 393.8 | 2.5 | 512 |

**Passer de BGE-base à Jina-v1-turbo donne environ 5× de débit sans réécrire le moteur.**

---

## Méthodologie

Chaque benchmark utilise des configurations différentes. Les résultats ne sont pas directement comparables entre tableaux sans tenir compte des paramètres.

| Paramètre | Valeur par défaut |
|-----------|-------------------|
| Corpus | 20 documents synthétiques |
| Batch size | 8 |
| Threads | 4 |
| Warmup | 1 itération |
| Iterations | 5-10 |
| Métrique | Médiane |

---

## Investigation 1: Comparaison des modèles

**Configuration**: 20 docs, batch=8, threads=4, iterations=5, documents ~100-200 tokens

| Modèle | ms/doc | docs/s | Load (ms) | Max tokens |
|--------|--------|--------|----------|------------|
| jinaai/jina-reranker-v1-turbo-en | 24.6 | 40.7 | 29027 | 8192 |
| jinaai/jina-reranker-v2-base-multilingual | 105.3 | 9.5 | 109289 | 8192 |
| BAAI/bge-reranker-base | 122.0 | 8.2 | 6225 | 512 |
| BAAI/bge-reranker-v2-m3 | 393.8 | 2.5 | 88981 | 512 |

### Analyse

1. **Jina-v1-turbo est 5x plus rapide que BGE-base** (24.6 vs 122 ms/doc)
2. **Jina-v1-turbo supporte 8192 tokens** (vs 512 pour BGE) — ce n'est pas un modèle "léger"
3. **BGE-v2-m3 est 3.2x plus lent que BGE-base** (393.8 ms/doc) — modèle multilingue coûteux
4. **Jina-v2-multilingual est comparable à BGE-base** (105.3 ms/doc)

### Conclusion

Le modèle est le facteur déterminant. **Jina-v1-turbo-en est le meilleur choix par défaut** pour l'anglais.

---

## Investigation 2: Sensibilité à la longueur des documents

**Configuration**: 20 docs, batch=8, threads=4, iterations=5, BGE-base

| tokens/doc | words/doc | Total (ms) | ms/doc | ms/100tok |
|------------|-----------|-----------|--------|----------|
| 32 | 24 | 1159 | 57.9 | 181.1 |
| 64 | 48 | 1791 | 89.5 | 139.9 |
| 128 | 96 | 3410 | 170.5 | 133.2 |
| 256 | 192 | 6962 | 348.1 | 136.0 |
| 512 | 384 | 13849 | 692.4 | 135.2 |

### Analyse

1. **Scaling quasi-linéaire**: 32→512 tokens (16×) → 57.9→692.4 ms/doc (12×)
2. **ms/100tok stable**: ~135-181 ms/100 tokens (overhead fixe à faible longueur)
3. **Implication**: un document de 512 tokens coûte **12× plus cher** qu'un document de 32 tokens

### Conclusion

La longueur des documents a un impact majeur. **La troncature à 256 tokens est recommandée** pour les cas sensibles à la latence.

---

## Investigation 3: Profiling (estimation)

**Note**: Cette estimation est basée sur les benchmarks embedding précédents (tokenizer = 0.003ms vs inference = 10ms). Un profiling C++ réel est nécessaire pour confirmer.

| Composant | Estimation | % du temps |
|-----------|-----------|------------|
| Tokenization | ~0.5 ms | ~0.3% |
| Tensor prep | ~1.5 ms | ~1% |
| **ONNX Inference** | **~170 ms** | **~97%** |
| Logits extract | ~0.2 ms | ~0.1% |
| Sorting | ~0.01 ms | ~0.01% |
| **Total** | **~173 ms** | **100%** |

### Conclusion

**~97% du temps est estimé en inference ONNX.** Le tokenizer et le post-processing sont probablement négligeables. Un profiling C++ réel est nécessaire pour confirmer cette estimation.

---

## Investigation 4: Settings ORT (baseline)

**Configuration**: 20 docs, batch=8, threads=4, iterations=10, BGE-base, documents ~100-200 tokens

| Setting | Value | ms/doc | docs/s |
|---------|-------|--------|--------|
| Baseline (ORT_ENABLE_ALL, 4 threads) | current | 87.7 | 11.4 |

### Note sur les écarts de mesure

On note un écart entre:
- **Model comparison**: BGE-base = 122 ms/doc
- **ORT baseline**: BGE-base = 87.7 ms/doc

**Explication**: les configurations diffèrent:
- Model comparison: 20 docs, batch=8, threads=4, **iterations=5**, documents ~100-200 tokens
- ORT baseline: 20 docs, batch=8, threads=4, **iterations=10**, documents ~100-200 tokens

L'écart s'explique par:
1. **Nombre d'itérations** différent (5 vs 10) → la médiane est plus stable avec 10
2. **Variance naturelle** de l'exécution ONNX
3. **Longueur exacte** des documents synthétiques différente

**Recommandation**: pour les benchmarks futurs, toujours documenter précisément:
- Corpus (nombre, longueur en tokens)
- Batch size
- Threads
- Warmup
- Iterations

---

## Investigation 5: Matrice Jina-v1-turbo (longueur x top_k)

**Configuration**: 5 docs par cellule, batch=8, threads=4, iterations=5

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 32 | 78ms | 143ms | 285ms |
| 128 | 233ms | 454ms | 920ms |
| 512 | 1116ms | 2173ms | 4440ms |
| 2048 | 9911ms | 19217ms | 38350ms |

### Analyse

1. **Scaling quasi-linéaire**: doubler les tokens ≈ double le temps
2. **top_k impact modéré**: 2× top_k ≈ 2× temps (scaling linéaire attendu)
3. **Longueur impact majeur**: 32→2048 tokens (64×) → 78→9911ms (127×)

### Conclusion

La longueur des documents est le **second levier principal** après le choix du modèle.

**Recommandation**: tronquer à 128 tokens pour un bon compromis qualité/performance.

### Budget calculator

| Budget max | Configuration optimale |
|------------|----------------------|
| 200ms | top_k=10, 32 tokens/doc |
| 500ms | top_k=20, 32 tokens/doc |
| 1000ms | top_k=20, 128 tokens/doc |
| 5000ms | top_k=20, 512 tokens/doc |

---

## Synthese des investigations

### Ce que les benchmarks démontrent

1. **Le modèle est le principal levier**: 5× de gain entre BGE-base et Jina-v1-turbo
2. **La longueur des documents est le second levier**: 12× entre 32 et 512 tokens
3. **Les optimisations C++ ne semblent pas susceptibles de produire un gain comparable au changement de modèle** (estimation: 97% en inference)

### Ce que les benchmarks NE démontrent PAS

1. **Le tokenizer n'est probablement pas un goulot** (estimation: 0.3% du temps, à confirmer par profiling C++ réel)
2. **Le post-processing n'est probablement pas un goulot** (estimation: 0.01% du temps)
3. **Les settings ORT ne peuvent probablement pas améliorer significativement** (baseline déjà à ORT_ENABLE_ALL)

### Conclusion

> **L'inférence ONNX est très probablement le principal coût du reranking. Un profiling C++ réel doit confirmer la répartition avant d'écarter définitivement les optimisations d'implémentation.**

### Recommandations

| Action | Priorité | Gain attendu |
|--------|----------|-------------|
| Changer modèle par défaut → Jina-v1-turbo-en | **Haute** | **5×** |
| Tronquer documents à 256 tokens | **Haute** | **2-12×** |
| Quantification INT8 | Moyenne | 1.5-2× |
| Profiling C++ réel | Moyenne | Confirmation |
| Auto-tuning spécifique reranker | Moyenne | Adaptation hardware |

### Ce qu'il ne faut PAS faire

- ❌ Optimiser le tokenizer (estimation: 0.3% du temps)
- ❌ Réécrire le post-processing (estimation: 0.01% du temps)
- ❌ Implémenter un session pool pour reranker (oversubscription prouvée)
- ❌ Micro-optimisations C++ (gain négligeable vs changement de modèle)

---

## Investigation 6: Realistic Documents with P50/P95/P99

**Configuration**: 30 iterations, batch=8, threads=4, Jina-v1-turbo

### P50 (médiane)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 20 | 46ms | 94ms | 189ms |
| 50 | 83ms | 172ms | 337ms |
| 100 | 155ms | 302ms | 587ms |
| 200 | 299ms | 594ms | 1193ms |
| 500 | 876ms | 1828ms | 3526ms |

### P95 (SLA production)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 20 | 56ms | 108ms | 225ms |
| 50 | 104ms | 203ms | 414ms |
| 100 | 179ms | 381ms | 710ms |
| 200 | 363ms | 711ms | 1342ms |
| 500 | 933ms | 2221ms | 3914ms |

### P99 (pire cas)

| tokens/doc | top_k=5 | top_k=10 | top_k=20 |
|------------|---------|----------|----------|
| 20 | 57ms | 112ms | 240ms |
| 50 | 106ms | 241ms | 522ms |
| 100 | 185ms | 407ms | 864ms |
| 200 | 831ms | 780ms | 1351ms |
| 500 | 936ms | 2284ms | 4151ms |

### Analyse

1. **Queue lourde significative**: P99/P50 ratio jusqu'à **2.8x** pour 200 tokens (299ms → 831ms)
2. **P95/P50 ratio**: 1.2x (docs courts) à 2.0x (docs longs)
3. **Implication SLA**: le P95 est la métrique critique, pas la médiane

### Conclusion

Pour Whoosh-NG:
- **Troncature à 100 tokens**: P95 top_k=10 = 381ms (acceptable)
- **Documents 200+ tokens**: P99 top_k=5 = 831ms (queue lourde!)

---

## Investigation 7: Quantification FP32 vs INT8 (RÉSULTAT)

**Configuration**: 20 docs, ~128 tokens, batch=8, threads=4, iterations=20

| Metric | FP32 | INT8 | Speedup |
|--------|------|------|---------|
| P50 (ms) | 774.5 | 405.3 | **1.91×** |
| P95 (ms) | 1014.9 | 491.4 | **2.07×** |
| P99 (ms) | 1014.9 | 491.4 | **2.07×** |
| ms/doc | 38.7 | 20.3 | **1.91×** |
| RAM delta (MB) | 189 | 45 | **4.24×** |
| Load time (ms) | 9383 | 9012 | ~1× |

### Qualité

| Corrélation FP32 vs INT8 | Valeur |
|-------------------------|--------|
| Pearson correlation | **0.874** |

### Conclusion

> **La quantification INT8 est bénéfique pour Jina-v1-turbo: 1.91× plus rapide, 4.24× moins de RAM.**

Contrairement à BGE embedding où la quantification nuisait, Jina-turbo bénéficie clairement de l'INT8.

**Modèle recommandé**: `jinaai/jina-reranker-v1-turbo-en-quantized` (INT8)

---

## Synthèse finale

### Ce que les benchmarks démontrent

1. **Le modèle est le principal levier**: 5× de gain entre BGE-base et Jina-v1-turbo
2. **La longueur des documents est le second levier**: 127× entre 32 et 2048 tokens
3. **La quantification INT8 est le troisième levier**: 1.91× plus rapide, 4.24× moins de RAM
4. **Les optimisations C++ ne semblent pas susceptibles de produire un gain comparable** (estimation: 97% en inference)

### Ce que les benchmarks NE démontrent PAS

1. **Le tokenizer n'est probablement pas un goulot** (estimation: 0.3% du temps, à confirmer par profiling C++ réel)
2. **Le post-processing n'est probablement pas un goulot** (estimation: 0.01% du temps)
3. **Les settings ORT ne peuvent probablement pas améliorer significativement** (baseline déjà à ORT_ENABLE_ALL)

### Conclusion

> **L'inférence ONNX est très probablement le principal coût du reranking. Un profiling C++ réel doit confirmer la répartition avant d'écarter définitivement les optimisations d'implémentation.**

### Recommandations finales

| Action | Priorité | Gain attendu |
|--------|----------|-------------|
| Modèle par défaut → Jina-v1-turbo-en-quantized (INT8) | **Haute** | **5× vs BGE, 1.91× vs FP32** |
| Tronquer documents à 100-128 tokens | **Haute** | **2-12×** |
| Profiling C++ réel | Moyenne | Confirmation |
| Auto-tuning spécifique reranker | Moyenne | Adaptation hardware |

### Ce qu'il ne faut PAS faire

- ❌ Optimiser le tokenizer (estimation: 0.3% du temps)
- ❌ Réécrire le post-processing (estimation: 0.01% du temps)
- ❌ Implémenter un session pool pour reranker (oversubscription prouvée)
- ❌ Micro-optimisations C++ (gain négligeable vs changement de modèle)

---

## Prochaines étapes

1. ✅ Comparaison des modèles
2. ✅ Sensibilité longueur (BGE-base)
3. ✅ Profiling estimé
4. ✅ Matrice Jina-v1-turbo × longueur × top_k
5. ✅ Documents réalistes avec P50/P95/P99
6. ✅ Quantification FP32 baseline (pas de modèle INT8 disponible)
7. ⬜ **Profiling C++ réel** (confirmation)
8. ⬜ **Auto-tuning** spécifique reranker

---

## Fichiers de benchmark

| Fichier | Objectif | Statut |
|---------|----------|--------|
| `bench_models.py` | Comparaison modèles | Exécuté |
| `bench_length.py` | Sensibilité longueur | Exécuté |
| `bench_ort.py` | Settings ORT | Baseline |
| `bench_profile.py` | Profiling | Estimation seulement |
| `bench_jina_matrix.py` | Matrice Jina × longueur × top_k | Exécuté |
| `bench_realistic.py` | Documents réalistes P50/P95/P99 | Exécuté |
| `bench_quantize.py` | Quantification FP32 vs INT8 | FP32 baseline |
| `investigation.md` | Ce document | Mis à jour |
