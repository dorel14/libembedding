# Sparse Embedding Benchmarks — Implementation Status

## ✅ Implemented

- `benchmarks/sparse/README.md` — Overview
- `benchmarks/sparse/bench_sparse_models.py` — Model comparison (SPLADE++ vs BGE-M3)
- `benchmarks/sparse/bench_sparse_compression.py` — Compression benchmark (dict vs CSR vs numpy)
- `benchmarks/sparse/bench_sparse_similarity.py` — Similarity benchmark (dense vs sparse vs hybrid)
- `benchmarks/sparse/bench_sparse_autotune.py` — Auto-tuning benchmark

## ⚠️ To be integrated

- Add sparse benchmark section to `benchmarks/bench_libembedding.cpp`
- Add sparse tests to `python/benchmarks/bench_matrix.py`
- Add sparse autotune API to C headers
- Update `SparseTextEmbedding` Python class with `top_terms` and `min_weight`

## Next steps

1. Run benchmarks on i7-1065G7 to get real numbers
2. Compare results with dense embedding benchmarks
3. Determine if sparse-only or hybrid retrieval is optimal for Whoosh-NG
4. Implement `best_sparse_config()` C API
