# Reranking Auto-Tuner — Résultats

## Implémentation

### C++ API (`autotuner.h`)

```c
// Reranker tuning result
typedef struct {
    int threads;
    int batch_size;
    int max_tokens;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
    double p95_latency_ms;
} lembed_reranker_tuning_result_t;

// Auto-tune reranker (threads × batch × max_tokens)
lembed_status_t lembed_reranker_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,  // QUICK (5-15s) or FULL (30-120s)
    lembed_reranker_tuning_result_t* result);

// Auto-config for target latency budget
lembed_status_t lembed_reranker_auto_config(
    const char* model_name,
    double target_latency_ms,
    lembed_reranker_tuning_result_t* result);
```

### Python API

```python
from libembedding import reranker_autotune, reranker_auto_config

# Find optimal configuration
result = reranker_autotune("jinaai/jina-reranker-v1-turbo-en-quantized")
print(f"Optimal: {result.threads} threads, batch={result.batch_size}, tokens={result.max_tokens}")
print(f"P50={result.latency_ms:.1f}ms, P95={result.p95_latency_ms:.1f}ms")

# Find config for latency budget
result = reranker_auto_config(target_latency_ms=200)
print(f"Budget 200ms: {result.threads} threads, {result.max_tokens} tokens")
```

## Résultats

### FP32 vs INT8 (auto-tuner QUICK mode)

| Modèle | threads | batch | tokens | P50 (ms) | P95 (ms) | docs/s |
|--------|---------|-------|--------|----------|----------|--------|
| **FP32** | 4 | 16 | 64 | 277.0 | 283.6 | 72.2 |
| **INT8** | 4 | 16 | 64 | 131.5 | 151.1 | 152.1 |

> **INT8 est 2.1× plus rapide que FP32** (277/131.5 = 2.11×)

### Configuration optimale trouvée

- **threads = 4** (sur machine 4c/8t)
- **batch_size = 16**
- **max_tokens = 64** (compromis throughput/latence)

### Auto-config (latency budget)

| Budget | Configuration | P50 réel |
|--------|--------------|----------|
| 500ms | t=4, b=16, tok=64 | ~131ms |
| 300ms | t=4, b=16, tok=64 | ~131ms |
| 200ms | t=4, b=4, tok=64 | ~150ms |
| 100ms | t=4, b=4, tok=32 | ~90ms |

## Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `include/libembedding/autotuner.h` | Ajout API reranker auto-tuner |
| `include/libembedding/detail/autotuner_impl.hpp` | Implémentation reranker auto-tuner |
| `include/libembedding/types.h` | Ajout modèle quantifié + default |
| `include/libembedding/model_registry.h` | Entrée modèle quantifié |
| `include/libembedding/model_registry.h` | Ajout `lembed_find_reranker_model_by_code` |
| `python/src/libembedding/_cdefs.h` | Déclarations C pour Python |
| `python/src/libembedding/reranker.py` | Wrappers Python |
| `python/src/libembedding/types.py` | Type `RerankerTuningResult` |
| `python/src/libembedding/__init__.py` | Exports |

## Cache

Les résultats sont cachés par modèle et CPU :
- Location: `%LOCALAPPDATA%\libembedding\autotune\reranker\`
- Format: JSON avec configuration optimale
- Invalidation: changement de modèle ou CPU

## Commandes

```bash
# Auto-tuner reranker
python -c "from libembedding import reranker_autotune; print(reranker_autotune())"

# Auto-config avec budget
python -c "from libembedding import reranker_auto_config; print(reranker_auto_config(target_latency_ms=200))"
```
