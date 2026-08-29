# Sparse Embedding Status

## Models Available

| Model | Files | Inference | Issue |
|-------|-------|-----------|-------|
| SPLADE++ | All present (model.onnx, tokenizer.json, config.json) | Failing | ONNX Runtime "Invalid argument" |
| BGE-M3 Sparse | Missing onnx_data | Not tested | Need to download external data |

## Benchmark Infrastructure

Script: `benchmarks/sparse/bench_sparse.py`

Measures:
- Load time
- RAM delta
- P50/P95 latency
- ms/doc
- docs/sec
- Average terms per document

## Next Steps

1. Debug SPLADE++ ONNX model compatibility (may need model-specific fixes)
2. Download BGE-M3 external data files
3. Run benchmarks once inference works
4. Compare sparse vs dense retrieval quality

## Files Created

| File | Content |
|------|---------|
| `benchmarks/sparse/bench_sparse.py` | Benchmark script |
| `docs/sparse.md` | This document |
