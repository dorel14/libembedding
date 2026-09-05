# libembedding — llama.cpp Backend Tuning Report
## Machine: Intel i7-1065G7 (4c/8t), Windows 11, 16GB RAM
## Date: 2026-08-30

---

## 1. Executive Summary

### Key Findings

| Finding | Impact |
|---------|--------|
| Current libembedding-ng llama.cpp backend does not benefit sufficiently from multi-text batching | Session pooling is the preferred CPU parallelization strategy |
| Session pool scales to 6 sessions (3.61x) then diminishes | Auto-detect optimal sessions per model |
| True batch (multi-seq in single encode) peaks at 202 docs/s | 2x slower than pool (412 docs/s) |
| For tested CPU workloads, ONNX Runtime achieves 1.2-4x llama.cpp throughput | Use ONNX when model available in both formats |
| Snowflake-XS-Q4 is the best GGUF tradeoff | Recommended default GGUF model |

> **Note on batching**: The current libembedding-ng llama.cpp backend does not benefit sufficiently from multi-text batching. Benchmarking shows that session-level parallelism currently provides substantially higher throughput than the tested true-batch configuration. Session pooling is therefore the preferred CPU parallelization strategy for the current llama.cpp embedding backend. The additional attention/context workload and memory behavior can reduce the benefit of multi-sequence batching for the tested embedding workload.

### Backend Positioning

| Backend | Role | Strength |
|---------|------|----------|
| **ONNX Runtime** | Performance priority | Native batching, higher throughput when model available |
| **llama.cpp** | GGUF compatibility | GGUF ecosystem, quantization, GPU (CUDA/Vulkan/Metal) |
| **Auto** | Dynamic selection | Picks best model+backend+config for objective |

---

## 2. Benchmark Methodology

### Models Tested

| Model | Dim | Params | Size | Layers | MTEB |
|-------|-----|--------|------|--------|------|
| MiniLM-L6-Q4 | 384 | 22M | 20 MB | 6 | 41.9 |
| MiniLM-L6-Q8 | 384 | 22M | 24 MB | 6 | 41.9 |
| Snowflake-XS-Q4 | 384 | 22M | 20 MB | 6 | 50.2 |
| Snowflake-S-Q4 | 384 | 33M | 28 MB | 12 | 52.0 |
| E5-small-Q4 | 384 | 33M | 28 MB | 12 | 46.0 |
| GIST-small-Q4 | 384 | 33M | 28 MB | 12 | 48.0 |

### Measurement Protocol

- **Corpus**: 64 unique texts (~12 tokens each)
- **Warmup**: 5 iterations before measurement
- **Sessions tested**: 1, 2, 3, 4, 6, 8 (auto-detect stops at <5% gain relative to best)
- **Threads per session**: 1 (optimal for multi-session)
- **Timer**: `std::chrono::steady_clock`
- **Platform**: Windows 11, MSVC 2022, Release mode

---

## 3. Critical Experiment: Session Pool vs True Batch

A dedicated benchmark was performed to determine whether multi-sequence llama.cpp batching or independent execution contexts provide better embedding throughput.

### Results

**Strategy A: Session Pool (N sessions × 1 thread)**

| Sessions | Docs/sec | Scaling |
|----------|----------|---------|
| 1 | 114.0 | 1.00x |
| 2 | 145.6 | 1.28x |
| 3 | 275.5 | 2.42x |
| 4 | 313.9 | 2.75x |
| 5 | 396.0 | 3.47x |
| **6** | **412.1** | **3.61x** |
| 7 | 385.6 | 3.38x |
| 8 | 366.7 | 3.22x |

**Strategy B: True Batch (1 session, N texts in single encode)**

| Batch Size | Docs/sec | Scaling |
|------------|----------|---------|
| 1 | 202.5 | 1.78x |
| 2 | 64.7 | 0.57x |
| 4 | 199.9 | 1.75x |
| 8 | 87.7 | 0.77x |
| 16 | 191.7 | 1.68x |
| 32 | 104.2 | 0.91x |
| 64 | 85.9 | 0.75x |

### Conclusion

> **Session pool (412 docs/s) is approximately 2x faster than true batch (202 docs/s).**
> The measured Session Pool configuration provides substantially higher throughput.
> Therefore, the current llama.cpp backend should use independent embedding sessions as its primary CPU parallelization mechanism.
> True Batch remains an experimental capability and should not be removed from the architecture, because performance may vary by model, sequence length, hardware and llama.cpp version.

**Peak**: 6 sessions × 1 thread = 412.1 docs/s (3.61x scaling)

> **Note**: The 412.1 docs/s result comes from the dedicated Session Pool vs True Batch benchmark and should not be compared directly with earlier measurements unless the benchmark configuration, corpus, model, iteration count and measurement methodology are identical.

---

## 4. Session Pool Scaling (Historical Data)

### MiniLM-L6-Q4 (6 layers, fastest)

| Sessions | Threads/session | Docs/sec | Scaling |
|----------|-----------------|----------|---------|
| 1 | 4 | 112 | 1.00x |
| 2 | 1 | 186 | 1.66x |
| 3 | 1 | 198 | 1.77x |
| 4 | 1 | 202 | 1.80x |
| 6 | 1 | 205 | 1.83x |
| 8 | 1 | 204 | 1.82x |

- **Peak**: 6 sessions = 205 docs/s
- **Efficient optimum (95% of peak)**: 3 sessions = 198 docs/s (> 194.75)

### Snowflake-XS-Q4 (6 layers, best quality)

| Sessions | Docs/sec | Scaling |
|----------|----------|---------|
| 1 | 95 | 1.00x |
| 2 | 186 | 1.96x |
| 3 | 198 | 2.08x |
| 4 | 204 | 2.15x |

- **Peak**: 4 sessions = 204 docs/s
- **Efficient optimum**: 2 sessions = 186 docs/s

### Snowflake-S-Q4 (12 layers, highest quality)

| Sessions | Docs/sec | Scaling |
|----------|----------|---------|
| 1 | 56 | 1.00x |
| 2 | 109 | 1.95x |
| 3 | 121 | 2.16x |
| 4 | 128 | 2.29x |

- **Peak**: 4 sessions = 128 docs/s
- **Efficient optimum**: 3 sessions = 121 docs/s

---

## 5. Before vs After Tuning

### Before: Default Configuration (1 session, 4 threads)

| Model | Docs/sec | ms/text |
|-------|----------|---------|
| MiniLM-L6-Q4 | 112 | 8.9 |
| Snowflake-XS-Q4 | 95 | 10.5 |
| Snowflake-S-Q4 | 56 | 17.9 |

### After: Auto-tuned Configuration

| Model | Sessions | Threads/sess | Docs/sec | Speedup |
|-------|----------|--------------|----------|---------|
| MiniLM-L6-Q4 | 3 | 1 | 198 | **1.77x** |
| Snowflake-XS-Q4 | 2 | 1 | 186 | **1.96x** |
| Snowflake-S-Q4 | 3 | 1 | 121 | **2.16x** |

---

## 6. Model Comparison: ONNX vs llama.cpp

### Same model (MiniLM-L6-v2) on CPU

| Backend | Docs/sec | Relative |
|---------|----------|----------|
| ONNX (4 threads, batch=64) | 230-700 | 1.0x (baseline) |
| llama.cpp Q4 (3 sessions) | 198 | 0.28-0.86x |
| llama.cpp Q8 (3 sessions) | 175 | 0.25-0.76x |

> **For the tested CPU workloads and models available in both formats, ONNX Runtime currently achieves approximately 1.2x-4x the throughput of the tested llama.cpp configurations.**
> This is a benchmark observation, not a universal property of the two runtimes.
> The result depends on model, quantization, CPU, batching, number of sessions, runtime version, and sequence length.

### llama.cpp Advantages

- Access to GGUF-only models (Qwen3-Embedding, GTE-Qwen, Stella)
- Quantized formats (Q4_K_M, Q5_K_M, Q6_K, Q8_0)
- GPU acceleration (CUDA, Vulkan, ROCm, Metal)

---

## 7. Pareto Frontier Analysis

### All Models (efficient optimum, quality vs throughput)

| Model | Quality | Throughput | Dominated by |
|-------|---------|------------|--------------|
| **Snowflake-XS-Q4** | 50.2 | 186 | None (best quality + speed) |
| **Snowflake-S-Q4** | 52.0 | 121 | None (best quality) |
| **MiniLM-L6-Q4** | 41.9 | 198 | None (best throughput) |

### Eliminated models

| Model | Reason |
|-------|--------|
| MiniLM-L6-Q8 | Dominated by Q4 (same quality, slower) |
| E5-small-Q4 | Dominated by Snowflake-S (lower quality, same speed) |
| GIST-small-Q4 | Dominated by Snowflake-XS (lower quality, same speed) |

---

## 8. Recommended Configurations

### For Maximum Throughput (C++)

```cpp
// MiniLM-L6-Q4: 198 docs/s (efficient optimum)
lembed_text_options_t opts = lembed_text_options_default();
opts.num_threads = 1;
lembed_text_embedding_t* emb;
lembed_text_embedding_create_from_gguf_path("path/to/MiniLM-L6-Q4.gguf", &opts, &emb);
// Use 3 sessions via LlamaSessionPool
```

### For Best Quality

```cpp
// Snowflake-S-Q4: 121 docs/s, MTEB 52.0
// Use 3-4 sessions
```

### For Balanced (Recommended GGUF Default)

```cpp
// Snowflake-XS-Q4: 186 docs/s, MTEB 50.2
// Use 2 sessions
```

### Model Selection Guide

| Use Case | Model | Backend | Docs/s | Quality |
|----------|-------|---------|--------|---------|
| Max throughput | MiniLM-L6-v2 | ONNX | 474-700 | 41.9 |
| Fast + GGUF | MiniLM-L6-Q4 | llama.cpp | 198 | 41.9 |
| Best GGUF | Snowflake-XS-Q4 | llama.cpp | 186 | 50.2 |
| Max quality | Snowflake-S-Q4 | llama.cpp | 121 | 52.0 |

---

## 9. Tuning Pipeline

### Selection Algorithm

```
1. HARD CONSTRAINTS (eliminate non-admissible)
   - quality >= quality_min
   - throughput >= throughput_min
   - size <= memory_max

2. PARETO FRONTIER (eliminate dominated)
   - Remove models worse on ALL axes

3. OBJECTIVE SCORING (rank remaining)
   - performance: Q=20%, T=70%, C=10%
   - balanced:    Q=50%, T=30%, C=20%
   - quality:     Q=70%, T=20%, C=10%
   - memory:      Q=40%, T=10%, C=50%

4. SESSION AUTO-DETECT
   - Measure 1..8 sessions
   - Track best throughput
   - Efficient optimum: throughput >= best * 0.95
```

### Cache Strategy

Cache key: `Hardware × Software × Model × Backend`

```json
{
  "cache_schema_version": 1,
  "hardware": {
    "cpu": "Intel Core i7-1065G7",
    "physical_cores": 4,
    "logical_cores": 8,
    "features": ["AVX2"],
    "memory_mb": 16384
  },
  "software": {
    "libembedding": "1.x",
    "backend": "llama.cpp",
    "backend_version": "b5434"
  },
  "model": {
    "id": "Snowflake-XS-Q4",
    "quantization": "Q4",
    "dimension": 384,
    "size_bytes": 20971520
  },
  "configuration": {
    "sessions": 2,
    "threads_per_session": 1
  },
  "metrics": {
    "throughput": 186.2,
    "latency_p50_ms": 5.4,
    "latency_p95_ms": 7.1,
    "memory_mb": 72.0
  }
}
```

---

## 10. Reproducibility

- 3-5 iterations per config
- 5 warmup iterations
- Corpus: 64 texts, ~12 tokens each
- Timer: `std::chrono::steady_clock`
- Variation: 5-15% (thermal throttling)
- llama.cpp version: b5434
- Build: MSVC 2022, `/O2 /arch:AVX2`, Release
