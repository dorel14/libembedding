# Image Embedding - Storage Cost Analysis

## Per-Vector Size

| Model | Dimensions | FP32 | FP16 | INT8 |
|-------|-----------|------|------|------|
| CLIP B/32 | 512 | 2 KB | 1 KB | 0.5 KB |
| CLIP B/16 | 768 | 3 KB | 1.5 KB | 0.75 KB |
| UNICOM B/32 | 512 | 2 KB | 1 KB | 0.5 KB |
| UNICOM B/16 | 768 | 3 KB | 1.5 KB | 0.75 KB |
| ResNet-50 | 2048 | 8 KB | 4 KB | 2 KB |
| Nomic Vision | 768 | 3 KB | 1.5 KB | 0.75 KB |

## Total Storage at Scale

### 1 Million Images

| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| CLIP B/32 | 2.05 GB | 1.02 GB | 0.51 GB |
| ResNet-50 | 8.19 GB | 4.10 GB | 2.05 GB |
| UNICOM B/16 | 3.07 GB | 1.54 GB | 0.77 GB |

### 10 Million Images

| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| CLIP B/32 | 20.5 GB | 10.2 GB | 5.1 GB |
| ResNet-50 | 81.9 GB | 41.0 GB | 20.5 GB |
| UNICOM B/16 | 30.7 GB | 15.4 GB | 7.7 GB |

## Index Overhead

### HNSW Index (approximate)
- Vector data: 100% of base size
- Graph connections: ~10-20% of base size
- Total: ~110-120% of base size

### Flat Index (brute force)
- Vector data: 100% of base size
- No overhead

### IVF Index
- Vector data: 100% of base size
- Centroids: negligible
- Total: ~100% of base size

## Whoosh-NG Integration Considerations

### Hybrid Search Architecture
```
Document
 ├── text
 │    ├── BM25 (sparse, inverted index)
 │    ├── dense vector (384-1024 dims)
 │    └── sparse vector (SPLADE++)
 │
 └── images
      └── image vector (512-2048 dims)
```

### Storage per Document (1 image + text)
| Component | Dense 512 | Dense 2048 |
|-----------|-----------|------------|
| Text dense vector | 2 KB | 8 KB |
| Text sparse vector | ~0.5 KB | ~0.5 KB |
| Image vector (FP32) | 2 KB | 8 KB |
| Image vector (INT8) | 0.5 KB | 2 KB |
| **Total (FP32 image)** | **4.5 KB** | **16.5 KB** |
| **Total (INT8 image)** | **3 KB** | **10.5 KB** |

### Total Storage (1M documents, 1 image each)
| Configuration | Vectors Only | + HNSW Overhead |
|---------------|-------------|-----------------|
| Text 512 + Image 512 FP32 | 4.0 GB | 4.5 GB |
| Text 512 + Image 512 INT8 | 2.5 GB | 2.9 GB |
| Text 512 + Image 2048 FP32 | 10.0 GB | 11.5 GB |
| Text 512 + Image 2048 INT8 | 4.5 GB | 5.2 GB |

## Recommendation

### For Whoosh-NG Image Search
1. **CLIP B/32 INT8**: Best balance of quality, speed, and storage
   - 0.5 KB/vector
   - 1.21x faster than FP32
   - 3.95x less RAM
   - Shared text/image space (multimodal)

2. **ResNet-50 FP32**: Only if pure image→image retrieval is needed
   - 8 KB/vector (16x more than CLIP INT8)
   - No multimodal capability
   - Faster inference (19.2 vs 14.7 img/s)

### Multimodal Use Cases
If Whoosh-NG needs:
- "Find images similar to this image" → ResNet-50 or CLIP
- "Find images matching this text" → CLIP only
- "Find documents with similar images" → CLIP only

CLIP is the only model that supports all three use cases.

## Files

| File | Content |
|------|---------|
| `benchmarks/image/STORAGE.md` | This document |
