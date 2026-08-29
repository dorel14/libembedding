# Image Embedding Status

## Models Available (5)

| Model | Dim | Architecture | ms/image | img/s |
|-------|-----|-------------|----------|-------|
| openai/clip-vit-base-patch32 | 512 | CLIP ViT-B/32 | 72.6 | 13.8 |
| microsoft/resnet-50 | 2048 | ResNet-50 | 52.2 | 19.2 |
| open-metric-learning/unicom-vit-b-16 | 768 | UNICOM ViT-B/16 | 330.8 | 3.0 |
| open-metric-learning/unicom-vit-b-32 | 512 | UNICOM ViT-B/32 | 69.2 | 14.5 |
| nomic-ai/nomic-embed-vision-v1.5 | 768 | Nomic Vision | 327.7 | 3.1 |

## Profiling Results (preprocessing vs inference)

| Model | Pipeline ms/image | Preprocess est. | Inference est. | Ratio |
|-------|------------------|-----------------|----------------|-------|
| CLIP B/32 | 72.6 | 16.0 | 274.4 | 17x |
| ResNet-50 | 52.2 | 44.2 | 164.5 | 3.7x |
| UNICOM B/32 | 69.2 | 0.5 | 276.3 | 527x |

**Conclusion**: Inference ONNX dominates for all models. Preprocessing optimization has limited impact.

## Batching Impact

| Model | batch=1 ms/img | batch=4 ms/img | Gain |
|-------|---------------|----------------|------|
| CLIP B/32 | 83.9 | 64.2 | 23% |
| ResNet-50 | 49.6 | - | ~0% |
| UNICOM B/32 | 89.6 | 65.5 | 27% |

**Conclusion**: Batching gives 22-27% improvement for transformer models, ~0% for ResNet.

## Batch Benchmark Results

| Model | Batch=1 | Batch=4 | Batch=8 | Batch=16 | Batch=32 | Optimal |
|-------|---------|---------|---------|----------|----------|---------|
| CLIP B/32 | 97.6 ms | 82.9 ms | 128.8 ms* | 70.0 ms | 69.4 ms | 32 (1.41x) |
| ResNet-50 | 52.5 ms | 53.3 ms | 52.5 ms | 50.8 ms | 55.2 ms | 16 (1.03x) |
| UNICOM B/32 | 77.1 ms | 69.2 ms | 73.1 ms | 72.6 ms | 79.3 ms | 4 (1.11x) |

*CLIP batch=8 anomalie: plus lent que batch=4 et 16 (effet seuil memoire ORT)

## Roadmap

- [x] Benchmark models
- [x] Profile preprocessing vs inference
- [x] Light pipeline optimization (zero-copy, CHW)
- [x] Benchmark batch 1->32
- [ ] Quantization benchmark
- [ ] Quality benchmark (image->image, text->image)
- [ ] Storage/index cost measurement
- [ ] Auto-tuner

## Batch Results Summary

- **CLIP B/32**: batch 32 optimal, 1.41x speedup (69.4 ms/img)
- **ResNet-50**: batch 16 optimal, 1.03x speedup (50.8 ms/img) — minimal gain
- **UNICOM B/32**: batch 4 optimal, 1.11x speedup (69.2 ms/img)

1. Model choice (biggest lever)
2. Batching (22-27% gain)
3. Quantization (1.5-2x potential)
4. Threads / ORT settings
5. Preprocessing (limited impact)

## Key Insight for Whoosh-NG

- ResNet-50: fastest (19.2 img/s) but 2048-dim = 8KB/vector
- CLIP B/32: 13.8 img/s but 512-dim = 2KB/vector + shared image/text space
- For multimodal search, CLIP may be better despite lower throughput
