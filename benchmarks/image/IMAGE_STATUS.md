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

## Roadmap

- [x] Benchmark models
- [x] Profile preprocessing vs inference
- [x] Light pipeline optimization (zero-copy, CHW)
- [ ] Benchmark batch 1→32
- [ ] Quantization benchmark
- [ ] Quality benchmark (image→image, text→image)
- [ ] Storage/index cost measurement
- [ ] Auto-tuner

## Priority Order

1. Model choice (biggest lever)
2. Batching (22-27% gain)
3. Quantization (1.5-2x potential)
4. Threads / ORT settings
5. Preprocessing (limited impact)

## Key Insight for Whoosh-NG

- ResNet-50: fastest (19.2 img/s) but 2048-dim = 8KB/vector
- CLIP B/32: 13.8 img/s but 512-dim = 2KB/vector + shared image/text space
- For multimodal search, CLIP may be better despite lower throughput
