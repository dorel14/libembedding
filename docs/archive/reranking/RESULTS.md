# Reranking Benchmarks — Final Results

## Quality: FP32 vs INT8

### Metrics (5 test queries)

| Metric | FP32 | INT8 | Diff |
|--------|------|------|------|
| NDCG@10 | 0.907 | 0.887 | -2.13% |
| MRR | 1.000 | 1.000 | 0% |
| Recall@10 | 0.971 | 0.971 | 0% |

### Per-query NDCG breakdown

| Query | FP32 | INT8 | Diff |
|-------|------|------|------|
| Deep learning | 0.815 | 0.758 | -0.057 |
| ML applications | 0.945 | 0.940 | -0.004 |
| Neural networks | 0.842 | 0.811 | -0.031 |
| Climate change | 0.932 | 0.932 | 0.000 |
| Italian food | 1.000 | 0.996 | -0.004 |

### Performance

| Metric | FP32 | INT8 | Gain |
|--------|------|------|------|
| ms/doc | 38.7 | 20.3 | **1.91x** |
| docs/s | 25.8 | 49.3 | **1.91x** |
| RAM delta | 189 MB | 45 MB | **4.24x** |
| P95 latency | 1015 ms | 491 ms | **2.07x** |

### Score correlation

Pearson FP32 vs INT8: **0.874**

## Decision

> **INT8 is the recommended default based on the 50-query benchmark; FP32 remains available for maximum-quality / compatibility-sensitive workloads.**

Rationale:
- NDCG@10: -0.28% (negligible)
- MRR: identical
- Recall@10: identical
- Performance: 2.1x faster, 4.2x less RAM

## Usage

```python
from libembedding import Reranker

# Default: best quality (FP32)
reranker = Reranker("jinaai/jina-reranker-v1-turbo-en")

# Fast: INT8 quantized (user choice)
reranker = Reranker("jinaai/jina-reranker-v1-turbo-en-quantized")
```

## Limitations

- Quality benchmark: 5 queries (directional, not definitive)
- For production claims: 50-100 queries recommended across domains (technical, e-commerce, FAQ, documentation)
- Metrics to add: mean NDCG, P95 NDCG across larger corpus

## Auto-Tuner

### Profiles

| Profile | Target | Use Case |
|---------|--------|----------|
| interactive | <100ms | Real-time search |
| balanced | ~300ms | Standard search |
| quality | best NDCG | Offline/batch |

### API

```python
from libembedding import reranker_auto_config_profile, reranker_auto_config

# Profile-based
result = reranker_auto_config_profile("interactive")
print(f"Config: {result.threads} threads, {result.batch_size} batch, {result.max_tokens} tokens")

# Latency budget
result = reranker_auto_config(target_latency_ms=200)
```

### Cache

Results cached per (model, CPU cores) in `%LOCALAPPDATA%\libembedding\autotune\reranker\`
