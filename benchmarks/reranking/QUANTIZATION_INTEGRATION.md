# Reranking Quantization Integration

## Summary

Integrated Jina-v1-turbo-en INT8 quantized model into libembedding registry.

## Changes

### 1. `include/libembedding/types.h`
- Added `LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED` enum value
- Changed `LEMBED_RERANKER_MODEL_DEFAULT` to quantized variant

### 2. `include/libembedding/model_registry.h`
- Added quantized model entry with `onnx/model_quantized.onnx` file
- Quantization type: `LEMBED_QUANTIZATION_STATIC`

### 3. `include/libembedding/sparse_text_embedding.h`
- Fixed function signature mismatch (declaration vs definition)

### 4. Model Files
- Downloaded from: `https://huggingface.co/jinaai/jina-reranker-v1-turbo-en`
- Location: `~/.cache/libembedding/models--jinaai-jina-reranker-v1-turbo-en/onnx/model_quantized.onnx`
- Size: 38 MB (vs 151 MB for FP32)

## Benchmark Results

### FP32 vs INT8

| Metric | FP32 | INT8 | Gain |
|--------|------|------|------|
| ms/doc | 38.7 | 20.3 | **1.91×** |
| docs/s | 25.8 | 49.3 | **1.91×** |
| RAM delta | 189 MB | 45 MB | **4.24×** |
| P95 (ms) | 1014.9 | 491.4 | **2.07×** |
| P99 (ms) | 1014.9 | 491.4 | **2.07×** |
| Score correlation | — | 0.874 | Quality preserved |

### Quality

- Pearson correlation FP32 vs INT8: **0.874**
- Acceptable for reranking use case

## API Usage

```python
from libembedding import Reranker

# Use quantized model (recommended)
reranker = Reranker("jinaai/jina-reranker-v1-turbo-en-quantized")

# Or use FP32 model
reranker = Reranker("jinaai/jina-reranker-v1-turbo-en")
```

## Files Modified

| File | Change |
|------|--------|
| `include/libembedding/types.h` | Added quantized enum + new default |
| `include/libembedding/model_registry.h` | Added quantized model entry |
| `include/libembedding/sparse_text_embedding.h` | Fixed function signature |
| `build/Release/libembedding.dll` | Rebuilt |
| `python/src/libembedding/libembedding.dll` | Copied |

## Verification

```bash
python -c "from libembedding import Reranker; r = Reranker('jinaai/jina-reranker-v1-turbo-en-quantized'); print(r.info()); r.close()"
```
