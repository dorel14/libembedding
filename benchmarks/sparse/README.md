# Sparse Embedding Benchmarks — Overview

This directory contains benchmarks for sparse embedding models in libembedding.

## Models

- **SPLADE++** (SPLADE++-distil) — 120 sparse terms per document, ~18 ms/doc
- **BGE-M3 Sparse** — 85 sparse terms per document, ~11 ms/doc

## Benchmark Suites

### 1. Model Comparison (`bench_sparse_models.py`)
Compare SPLADE++ vs BGE-M3 on:
- Loading time
- Inference time (ms/doc)
- Memory usage
- Average non-zero terms
- Output size

### 2. Compression Benchmark (`bench_sparse_compression.py`)
Compare storage formats:
- `dict[int, float]` — Python dict
- CSR (Compressed Sparse Row) — C-style
- numpy sparse (COO/CSR) — NumPy

Metrics: memory size, access speed, serialization time.

### 3. Similarity Benchmark (`bench_sparse_similarity.py`)
Compare retrieval methods:
- Dense cosine
- Sparse dot product
- BM25
- Dense + Sparse (hybrid)

Datasets: MS MARCO, BEIR, custom Whoosh-NG-like.

### 4. Auto-Tuning (`bench_sparse_autotune.py`)
Find optimal:
- `pruning_threshold` (0.0 - 0.1)
- `top_k` (32 - 256)
- `quantization` (none/static/dynamic)
- `storage_format` (dict/csr/numpy)

## Quick Start

```bash
# Run all sparse benchmarks
cd benchmarks/sparse
python bench_sparse_models.py
python bench_sparse_compression.py
python bench_sparse_similarity.py
python bench_sparse_autotune.py
```
