# Image Embedding - Quantization

## Available Quantized Models (Xenova/clip-vit-base-patch32)

| Model | Size | Type |
|-------|------|------|
| vision_model_int8.onnx | 88.6 MB | INT8 |
| vision_model_uint8.onnx | 88.6 MB | UINT8 |
| vision_model_fp16.onnx | 176 MB | FP16 |
| vision_model_bnb4.onnx | 58.3 MB | 4-bit |
| vision_model_q4.onnx | 63.6 MB | 4-bit |

Source: https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx

## Quantization Benchmark Methodology

### Performance metrics
For each quantized variant, measure:
- Load time
- RAM delta (load + inference)
- P50/P99 latency
- ms/image
- images/second

### Quality metrics
Compare FP32 vs INT8:
- Image->Image retrieval (Recall@K, NDCG@K)
- Score correlation (Pearson)

### Expected behavior (based on reranking experience)
- Jina INT8: 1.91x faster, NDCG -0.28% (beneficial)
- BGE INT8: may degrade (must measure)

## Storage Cost Analysis

### Per-vector size
| Model | Dimensions | FP32 | INT8 |
|-------|-----------|------|------|
| CLIP B/32 | 512 | 2 KB | 0.5 KB |
| ResNet-50 | 2048 | 8 KB | 2 KB |
| UNICOM B/16 | 768 | 3 KB | 0.75 KB |

### At scale (1 million images)
| Model | FP32 | INT8 |
|-------|------|------|
| CLIP B/32 | 2.05 GB | 0.51 GB |
| ResNet-50 | 8.19 GB | 2.05 GB |

## Integration Steps

1. Download quantized model from HuggingFace
2. Add to model_registry.h with quantization flag
3. Run performance benchmark
4. Run quality benchmark
5. Compare and decide default

## Files

| File | Content |
|------|---------|
| `benchmarks/image/bench_quantize.py` | Quantization benchmark script |
| `docs/image/quantization.md` | This document |
