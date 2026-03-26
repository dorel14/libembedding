# libembedding

A C/C++ header-only library for generating text embeddings, sparse embeddings, image embeddings, and document reranking using ONNX Runtime. Models are automatically downloaded from HuggingFace Hub on first use.

Inspired by [fastembed](https://github.com/qdrant/fastembed) (Python) and [fastembed-rs](https://github.com/qdrant/fastembed-rs) (Rust).

## Features

- **44 text embedding models** (BGE, MiniLM, Nomic, E5, CLIP, Jina, GTE, Snowflake, etc.)
- **2 sparse embedding models** (SPLADE++, BGE-M3)
- **5 image embedding models** (CLIP ViT-B-32, ResNet-50, Unicom, Nomic Vision)
- **4 reranker models** (BGE Reranker, Jina Reranker)
- Automatic model downloading and caching from HuggingFace Hub
- Pure C API (`extern "C"`) for maximum FFI compatibility
- Header-only (STB-style `#define LIBEMBEDDING_IMPLEMENTATION`)
- CLS and Mean pooling with L2 normalization
- Batch processing with configurable batch sizes
- CPU, CUDA, CoreML, DirectML execution providers
- Custom/user-defined model support (bring your own ONNX)

## Requirements

| Dependency | Required | Notes |
|---|---|---|
| **ONNX Runtime** >= 1.16 | Yes | `brew install onnxruntime` or set `ONNXRUNTIME_ROOT` |
| **libcurl** >= 7.0 | Optional | For model downloading. Disable with `-DLIBEMBEDDING_NO_DOWNLOAD=ON` |
| **cJSON** | Bundled | Included in `third_party/` |
| **stb_image** | Bundled | Included in `third_party/`. Disable with `-DLIBEMBEDDING_NO_IMAGE=ON` |
| **CMake** >= 3.18 | Build only | |
| **C++17 compiler** | Build only | GCC 7+, Clang 5+, MSVC 2017+ |

No Rust toolchain required. The tokenizer is implemented natively in C++ (supports WordPiece and BPE models via `tokenizer.json`).

## Quick Start

### Build

```bash
./build.sh           # Release build
./build.sh Debug     # Debug build
```

Or manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### Run Tests

```bash
./run_tests.sh                # Unit tests only (no network)
./run_tests.sh --integration  # Include integration tests (downloads models)
```

### Run Examples

```bash
./run_examples.sh basic_embedding
./run_examples.sh batch_embedding
```

## Integration Guide

### Step 1: Add to Your Project

libembedding is header-only. Copy the `include/libembedding/` directory into your project, or use CMake's `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(libembedding
    GIT_REPOSITORY https://github.com/yourorg/libembedding
    GIT_TAG main
)
FetchContent_MakeAvailable(libembedding)

target_link_libraries(your_app PRIVATE libembedding::libembedding)
```

Or with an existing checkout:

```cmake
add_subdirectory(path/to/libembedding)
target_link_libraries(your_app PRIVATE libembedding::libembedding)
```

### Step 2: Implementation File

Create **exactly one** `.cpp` file in your project that defines the implementation:

```cpp
// embedding_impl.cpp
#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
```

This compiles all the library implementation code into a single translation unit. The implementation file **must** be C++ (`.cpp`) because the internal detail headers use C++17. All other files in your project include headers normally and can use the C API from either C or C++.

### Step 3: Use the API

```cpp
// any_file.cpp (or .c if only using declarations, not LIBEMBEDDING_IMPLEMENTATION)
#include <libembedding/text_embedding.h>

void my_function(void) {
    lembed_text_options_t opts = lembed_text_options_default();
    lembed_text_embedding_t* embedder = NULL;

    lembed_text_embedding_create(&opts, &embedder);
    // ... use embedder ...
    lembed_text_embedding_free(embedder);
}
```

> **Note:** Files that `#define LIBEMBEDDING_IMPLEMENTATION` must be compiled as C++ (`.cpp`). Files that only call the C API functions (without the implementation define) can be plain C.

## C API Reference

### Error Handling

Every function returns `lembed_status_t`. On failure, call `lembed_last_error()` for a detailed message (thread-local).

```c
lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);
if (s != LEMBED_OK) {
    fprintf(stderr, "Error [%s]: %s\n",
            lembed_status_message(s),  // e.g. "ONNX Runtime error"
            lembed_last_error());      // e.g. "Failed to load model: ..."
}
```

Status codes:

| Code | Meaning |
|---|---|
| `LEMBED_OK` | Success |
| `LEMBED_ERROR_INVALID_ARGUMENT` | NULL pointer or out-of-range enum |
| `LEMBED_ERROR_OUT_OF_MEMORY` | malloc failed |
| `LEMBED_ERROR_ONNX_RUNTIME` | ONNX Runtime session or inference error |
| `LEMBED_ERROR_TOKENIZER` | Tokenizer loading or encoding error |
| `LEMBED_ERROR_DOWNLOAD` | Model download failed |
| `LEMBED_ERROR_IO` | File I/O error |
| `LEMBED_ERROR_MODEL_NOT_FOUND` | Unknown model enum |
| `LEMBED_ERROR_UNSUPPORTED` | Feature disabled at compile time |
| `LEMBED_ERROR_BATCH_SIZE` | Batch size incompatible with dynamic quantization |

---

### Text Embedding

Generate dense vector representations of text. The default model is `BAAI/bge-small-en-v1.5` (384 dimensions).

```c
#include <libembedding/text_embedding.h>

// 1. Configure options
lembed_text_options_t opts = lembed_text_options_default();
opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;  // or any LEMBED_TEXT_* enum
opts.num_threads = 4;                        // 0 = auto

// 2. Create embedder (downloads model on first use)
lembed_text_embedding_t* embedder = NULL;
lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

// 3. Embed texts
const char* texts[] = { "Hello world", "How are you?" };
lembed_embeddings_t result = {0};
s = lembed_text_embedding_embed(embedder, texts, 2, 0 /* batch_size */, &result);

// 4. Access results
//    result.data  — flat float array [num_embeddings * dim]
//    result.dim   — embedding dimension (e.g. 384)
//    result.num_embeddings — number of embeddings
for (int i = 0; i < result.num_embeddings; i++) {
    float* vec = result.data + i * result.dim;
    // vec[0..dim-1] is the L2-normalized embedding for texts[i]
}

// 5. Cleanup
lembed_embeddings_free(&result);
lembed_text_embedding_free(embedder);
```

**Available text models** (44 total):

| Enum | Model | Dim | Pooling |
|---|---|---|---|
| `LEMBED_TEXT_BGE_SMALL_EN_V15` | BAAI/bge-small-en-v1.5 (default) | 384 | CLS |
| `LEMBED_TEXT_ALL_MINILM_L6_V2` | sentence-transformers/all-MiniLM-L6-v2 | 384 | Mean |
| `LEMBED_TEXT_BGE_BASE_EN_V15` | BAAI/bge-base-en-v1.5 | 768 | CLS |
| `LEMBED_TEXT_BGE_LARGE_EN_V15` | BAAI/bge-large-en-v1.5 | 1024 | CLS |
| `LEMBED_TEXT_NOMIC_EMBED_TEXT_V15` | nomic-ai/nomic-embed-text-v1.5 | 768 | Mean |
| `LEMBED_TEXT_BGE_M3` | BAAI/bge-m3 (multilingual) | 1024 | CLS |
| `LEMBED_TEXT_MULTILINGUAL_E5_LARGE` | intfloat/multilingual-e5-large | 1024 | Mean |
| `LEMBED_TEXT_GTE_LARGE_EN_V15` | Alibaba-NLP/gte-large-en-v1.5 | 1024 | CLS |
| `LEMBED_TEXT_CLIP_VIT_B32` | openai/clip-vit-base-patch32 (text) | 512 | Mean |
| ... | *Use `lembed_list_text_models()` for the full list* | | |

Quantized variants (suffix `_Q`) are available for most models.

---

### Sparse Text Embedding

Generate sparse vectors (term weights) using SPLADE++ or BGE-M3.

```c
#include <libembedding/sparse_text_embedding.h>

lembed_sparse_options_t opts = lembed_sparse_options_default();
// opts.model = LEMBED_SPARSE_SPLADE_PP_V1;  // default
// opts.model = LEMBED_SPARSE_BGE_M3;

lembed_sparse_embedding_ctx_t* embedder = NULL;
lembed_sparse_text_embedding_create(&opts, &embedder);

const char* texts[] = { "machine learning algorithms" };
lembed_sparse_embeddings_t result = {0};
lembed_sparse_text_embedding_embed(embedder, texts, 1, 0, &result);

// Each result has (indices, values, length)
for (int i = 0; i < result.items[0].length; i++) {
    int32_t token_id = result.items[0].indices[i];
    float weight     = result.items[0].values[i];
    // Use for sparse retrieval (e.g. inverted index lookup)
}

lembed_sparse_embeddings_free(&result);
lembed_sparse_text_embedding_free(embedder);
```

---

### Image Embedding

Generate dense vectors from images using CLIP or ResNet models.

```c
#include <libembedding/image_embedding.h>

lembed_image_options_t opts = lembed_image_options_default();
// opts.model = LEMBED_IMAGE_CLIP_VIT_B32;  // default, 512-dim

lembed_image_embedding_t* embedder = NULL;
lembed_image_embedding_create(&opts, &embedder);

// From file paths
const char* files[] = { "photo.jpg", "diagram.png" };
lembed_embeddings_t result = {0};
lembed_image_embedding_embed_files(embedder, files, 2, 0, &result);

// Or from memory buffers
// lembed_image_embedding_embed_bytes(embedder, buffers, sizes, 2, 0, &result);

lembed_embeddings_free(&result);
lembed_image_embedding_free(embedder);
```

| Enum | Model | Dim |
|---|---|---|
| `LEMBED_IMAGE_CLIP_VIT_B32` | CLIP ViT-B/32 (default) | 512 |
| `LEMBED_IMAGE_RESNET50` | ResNet-50 | 2048 |
| `LEMBED_IMAGE_UNICOM_VIT_B16` | Unicom ViT-B/16 | 768 |
| `LEMBED_IMAGE_UNICOM_VIT_B32` | Unicom ViT-B/32 | 512 |
| `LEMBED_IMAGE_NOMIC_EMBED_VISION_V15` | Nomic embed vision v1.5 | 768 |

---

### Reranker

Score and sort documents by relevance to a query using cross-encoder models.

```c
#include <libembedding/reranker.h>

lembed_reranker_options_t opts = lembed_reranker_options_default();
// opts.model = LEMBED_RERANKER_BGE_BASE;  // default

lembed_reranker_t* reranker = NULL;
lembed_reranker_create(&opts, &reranker);

const char* query = "What is deep learning?";
const char* docs[] = {
    "Deep learning uses neural networks with many layers",
    "The weather is sunny today",
    "Neural networks are inspired by biological brains"
};

lembed_rerank_results_t result = {0};
lembed_reranker_rerank(reranker, query, docs, 3, 0, &result);

// Results are pre-sorted by score (descending)
for (int i = 0; i < result.count; i++) {
    printf("#%d score=%.4f doc[%d]=\"%s\"\n",
           i + 1, result.items[i].score,
           result.items[i].index,
           docs[result.items[i].index]);
}

lembed_rerank_results_free(&result);
lembed_reranker_free(reranker);
```

| Enum | Model |
|---|---|
| `LEMBED_RERANKER_BGE_BASE` | BAAI/bge-reranker-base (default) |
| `LEMBED_RERANKER_BGE_V2_M3` | BAAI/bge-reranker-v2-m3 (multilingual) |
| `LEMBED_RERANKER_JINA_V1_TURBO_EN` | jinaai/jina-reranker-v1-turbo-en |
| `LEMBED_RERANKER_JINA_V2_BASE_MULTILINGUAL` | jinaai/jina-reranker-v2-base-multilingual |

---

### Model Registry

Query available models at runtime without creating an embedder.

```c
#include <libembedding/model_registry.h>

// List all text models
const lembed_model_info_t* models;
int count;
lembed_list_text_models(&models, &count);

for (int i = 0; i < count; i++) {
    printf("%-40s dim=%-4d %s pooling  %s\n",
           models[i].model_name,
           models[i].dim,
           models[i].pooling == LEMBED_POOLING_CLS ? "CLS" : "Mean",
           models[i].description);
}

// Get info for a specific model
lembed_model_info_t info;
lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15, &info);
printf("Code: %s, File: %s\n", info.model_code, info.model_file);

// Find model by HuggingFace repo code
int idx = lembed_find_text_model_by_code("Xenova/bge-small-en-v1.5");
// idx == LEMBED_TEXT_BGE_SMALL_EN_V15
```

The `lembed_model_info_t` struct:

```c
typedef struct {
    const char* model_name;    // e.g. "BAAI/bge-small-en-v1.5"
    const char* model_code;    // HF repo, e.g. "Xenova/bge-small-en-v1.5"
    const char* model_file;    // e.g. "onnx/model.onnx"
    const char* description;
    int         dim;           // embedding dimension
    int         max_tokens;    // max input token length
    int         pooling;       // LEMBED_POOLING_CLS or LEMBED_POOLING_MEAN
    int         quantization;  // LEMBED_QUANTIZATION_NONE/STATIC/DYNAMIC
} lembed_model_info_t;
```

---

### Custom Models (Bring Your Own ONNX)

Use any ONNX model not in the built-in registry:

```c
#include <libembedding/text_embedding.h>

// Load your ONNX model and tokenizer.json into memory
unsigned char* onnx_bytes = load_file("my_model.onnx", &onnx_size);
unsigned char* tok_bytes  = load_file("tokenizer.json", &tok_size);

lembed_user_defined_model_t model = {
    .onnx_data          = onnx_bytes,
    .onnx_data_size     = onnx_size,
    .tokenizer_json     = tok_bytes,
    .tokenizer_json_size = tok_size,
    .pooling            = LEMBED_POOLING_MEAN,
    .dim                = 768,
    .max_length         = 512,
};

lembed_text_embedding_t* embedder = NULL;
lembed_text_embedding_create_custom(&model, LEMBED_PROVIDER_CPU, 0, &embedder);

// Use exactly like built-in models
lembed_embeddings_t result = {0};
lembed_text_embedding_embed(embedder, texts, n, 0, &result);

lembed_embeddings_free(&result);
lembed_text_embedding_free(embedder);
```

---

### Options Reference

All option structs have a `_default()` constructor that returns safe defaults:

```c
typedef struct {
    lembed_text_model_t         model;         // default: LEMBED_TEXT_BGE_SMALL_EN_V15
    lembed_execution_provider_t provider;      // default: LEMBED_PROVIDER_CPU
    int                         device_id;     // default: 0
    const char*                 cache_dir;     // default: NULL (~/.cache/libembedding)
    int                         max_length;    // default: 0 (model's default)
    int                         num_threads;   // default: 0 (auto)
    int                         show_download_progress; // default: 1
} lembed_text_options_t;
```

**Execution providers:**

| Enum | Backend |
|---|---|
| `LEMBED_PROVIDER_CPU` | CPU (default, always available) |
| `LEMBED_PROVIDER_CUDA` | NVIDIA CUDA (requires CUDA toolkit + ORT CUDA provider) |
| `LEMBED_PROVIDER_COREML` | Apple CoreML (macOS/iOS) |
| `LEMBED_PROVIDER_DIRECTML` | DirectML (Windows) |
| `LEMBED_PROVIDER_TENSORRT` | NVIDIA TensorRT |

---

### Memory Management

The library allocates output buffers internally. Always free them with the matching free function:

| Allocator | Free function |
|---|---|
| `lembed_text_embedding_embed()` | `lembed_embeddings_free(&result)` |
| `lembed_sparse_text_embedding_embed()` | `lembed_sparse_embeddings_free(&result)` |
| `lembed_image_embedding_embed_*()` | `lembed_embeddings_free(&result)` |
| `lembed_reranker_rerank()` | `lembed_rerank_results_free(&result)` |
| `lembed_text_embedding_create()` | `lembed_text_embedding_free(ctx)` |
| `lembed_ensure_*_model()` | `lembed_free_string(path)` |

---

### Environment Variables

| Variable | Purpose |
|---|---|
| `LIBEMBEDDING_CACHE_DIR` | Override default model cache directory |
| `FASTEMBED_CACHE_DIR` | Alternative cache dir (compatibility with fastembed) |
| `HF_ENDPOINT` | Custom HuggingFace Hub endpoint URL |
| `ONNXRUNTIME_ROOT` | Path to ONNX Runtime installation (for CMake) |

---

## Compile-Time Configuration

Define these **before** including any libembedding header:

```c
#define LIBEMBEDDING_NO_DOWNLOAD  // Disable model downloading (offline only)
#define LIBEMBEDDING_NO_IMAGE     // Disable image embedding (removes stb dependency)
```

Or via CMake:

```bash
cmake .. -DLIBEMBEDDING_NO_DOWNLOAD=ON -DLIBEMBEDDING_NO_IMAGE=ON
```

## Architecture

```
                    +-----------------------+
                    |   libembedding.h      |  <-- single include
                    |   (umbrella header)   |
                    +-----------+-----------+
                                |
     +--------+--------+---+---+---+--------+--------+
     |        |        |       |   |        |        |
  types.h  error.h  model   text  image  sparse  reranker.h
                   registry emb.  emb.   emb.
                      .h    .h    .h     .h
                             |       |        |
                   +---------+-----------+-----------+
                   |       detail/ (C++ internals)   |
                   |                                 |
                   |  onnx_session_impl.hpp          |
                   |  tokenizer_impl.hpp (built-in)  |
                   |  pooling.hpp / normalize.hpp    |
                   |  downloader_impl.hpp            |
                   |  sparse_postprocess.hpp         |
                   |  image_preprocess.hpp           |
                   +---------------------------------+
                               |
                   +-----------+-----------+
                   |    External deps      |
                   |  ONNX Runtime (C API) |
                   |  cJSON (bundled)      |
                   |  stb_image (bundled)  |
                   |  libcurl (optional)   |
                   +-----------------------+
```

## License

MIT
