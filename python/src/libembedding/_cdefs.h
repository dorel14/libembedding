/* Flattened C declarations for cffi — derived from libembedding public headers.
 * No preprocessor directives, no C++ constructs. */

/* ── Error handling ─────────────────────────────────────────────── */

typedef enum {
    LEMBED_OK = 0,
    LEMBED_ERROR_INVALID_ARGUMENT,
    LEMBED_ERROR_OUT_OF_MEMORY,
    LEMBED_ERROR_ONNX_RUNTIME,
    LEMBED_ERROR_TOKENIZER,
    LEMBED_ERROR_DOWNLOAD,
    LEMBED_ERROR_IO,
    LEMBED_ERROR_MODEL_NOT_FOUND,
    LEMBED_ERROR_UNSUPPORTED,
    LEMBED_ERROR_BATCH_SIZE,
} lembed_status_t;

const char* lembed_status_message(lembed_status_t status);
const char* lembed_last_error(void);
const char* lembed_version(void);

/* ── Enums ──────────────────────────────────────────────────────── */

typedef enum {
    LEMBED_PROVIDER_CPU = 0,
    LEMBED_PROVIDER_CUDA,
    LEMBED_PROVIDER_COREML,
    LEMBED_PROVIDER_DIRECTML,
    LEMBED_PROVIDER_TENSORRT,
} lembed_execution_provider_t;

typedef enum {
    LEMBED_POOLING_CLS = 0,
    LEMBED_POOLING_MEAN,
} lembed_pooling_t;

typedef enum {
    LEMBED_QUANTIZATION_NONE = 0,
    LEMBED_QUANTIZATION_STATIC,
    LEMBED_QUANTIZATION_DYNAMIC,
} lembed_quantization_t;

typedef enum {
    LEMBED_TEXT_ALL_MINILM_L6_V2 = 0,
    LEMBED_TEXT_ALL_MINILM_L6_V2_Q,
    LEMBED_TEXT_ALL_MINILM_L12_V2,
    LEMBED_TEXT_ALL_MINILM_L12_V2_Q,
    LEMBED_TEXT_ALL_MPNET_BASE_V2,
    LEMBED_TEXT_BGE_BASE_EN_V15,
    LEMBED_TEXT_BGE_BASE_EN_V15_Q,
    LEMBED_TEXT_BGE_LARGE_EN_V15,
    LEMBED_TEXT_BGE_LARGE_EN_V15_Q,
    LEMBED_TEXT_BGE_SMALL_EN_V15,
    LEMBED_TEXT_BGE_SMALL_EN_V15_Q,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V1,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V15,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V15_Q,
    LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2,
    LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q,
    LEMBED_TEXT_PARAPHRASE_ML_MPNET_BASE_V2,
    LEMBED_TEXT_BGE_SMALL_ZH_V15,
    LEMBED_TEXT_BGE_LARGE_ZH_V15,
    LEMBED_TEXT_BGE_M3,
    LEMBED_TEXT_MODERNBERT_EMBED_LARGE,
    LEMBED_TEXT_MULTILINGUAL_E5_SMALL,
    LEMBED_TEXT_MULTILINGUAL_E5_BASE,
    LEMBED_TEXT_MULTILINGUAL_E5_LARGE,
    LEMBED_TEXT_MXBAI_EMBED_LARGE_V1,
    LEMBED_TEXT_MXBAI_EMBED_LARGE_V1_Q,
    LEMBED_TEXT_GTE_BASE_EN_V15,
    LEMBED_TEXT_GTE_BASE_EN_V15_Q,
    LEMBED_TEXT_GTE_LARGE_EN_V15,
    LEMBED_TEXT_GTE_LARGE_EN_V15_Q,
    LEMBED_TEXT_CLIP_VIT_B32,
    LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_CODE,
    LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_EN,
    LEMBED_TEXT_EMBEDDING_GEMMA_300M,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L_Q,
    LEMBED_TEXT_MODEL_COUNT,
} lembed_text_model_t;

typedef enum {
    LEMBED_SPARSE_SPLADE_PP_V1 = 0,
    LEMBED_SPARSE_BGE_M3,
    LEMBED_SPARSE_MODEL_COUNT,
} lembed_sparse_model_t;

typedef enum {
    LEMBED_IMAGE_CLIP_VIT_B32 = 0,
    LEMBED_IMAGE_RESNET50,
    LEMBED_IMAGE_UNICOM_VIT_B16,
    LEMBED_IMAGE_UNICOM_VIT_B32,
    LEMBED_IMAGE_NOMIC_EMBED_VISION_V15,
    LEMBED_IMAGE_MODEL_COUNT,
} lembed_image_model_t;

typedef enum {
    LEMBED_RERANKER_BGE_BASE = 0,
    LEMBED_RERANKER_BGE_V2_M3,
    LEMBED_RERANKER_JINA_V1_TURBO_EN,
    LEMBED_RERANKER_JINA_V2_BASE_MULTILINGUAL,
    LEMBED_RERANKER_MODEL_COUNT,
} lembed_reranker_model_t;

/* ── Structs ────────────────────────────────────────────────────── */

typedef struct {
    const char* model_name;
    const char* model_code;
    const char* model_file;
    const char* description;
    int         dim;
    int         max_tokens;
    int         pooling;
    int         quantization;
} lembed_model_info_t;

typedef struct {
    const char*                name;
    int                        dimension;
    int                        max_length;
    int                        pooling;
    int                        num_threads;
    int                        batch_size;
    lembed_execution_provider_t provider;
    int                        device_id;
} lembed_model_desc_t;

typedef struct {
    unsigned long long texts_embedded;
    unsigned long long batches_run;
    double avg_latency_ms;
} lembed_stats_t;

typedef struct lembed_text_embedding    lembed_text_embedding_t;
typedef struct lembed_sparse_embedding  lembed_sparse_embedding_ctx_t;
typedef struct lembed_image_embedding   lembed_image_embedding_t;
typedef struct lembed_reranker          lembed_reranker_t;

typedef struct {
    float*  data;
    int     num_embeddings;
    int     dim;
} lembed_embeddings_t;

typedef struct {
    int32_t* indices;
    float*   values;
    int      length;
} lembed_sparse_embedding_t;

typedef struct {
    lembed_sparse_embedding_t* items;
    int count;
} lembed_sparse_embeddings_t;

typedef struct {
    int   index;
    float score;
} lembed_rerank_result_t;

typedef struct {
    lembed_rerank_result_t* items;
    int count;
} lembed_rerank_results_t;

typedef struct {
    lembed_text_model_t         model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         pooling;
    int                         dim;
} lembed_text_options_t;

typedef struct {
    lembed_sparse_model_t       model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
} lembed_sparse_options_t;

typedef struct {
    lembed_image_model_t        model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         dim;
} lembed_image_options_t;

typedef struct {
    lembed_reranker_model_t     model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
} lembed_reranker_options_t;

typedef struct {
    const unsigned char* onnx_data;
    size_t               onnx_data_size;
    const unsigned char* tokenizer_json;
    size_t               tokenizer_json_size;
    const unsigned char* config_json;
    size_t               config_json_size;
    lembed_pooling_t     pooling;
    int                  dim;
    int                  max_length;
} lembed_user_defined_model_t;

/* ── Functions ──────────────────────────────────────────────────── */

/* Options defaults */
lembed_text_options_t    lembed_text_options_default(void);
lembed_sparse_options_t  lembed_sparse_options_default(void);
lembed_image_options_t   lembed_image_options_default(void);
lembed_reranker_options_t lembed_reranker_options_default(void);

/* Text embedding */
lembed_status_t lembed_text_embedding_create(const lembed_text_options_t* options, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_create_custom(const lembed_user_defined_model_t* model, lembed_execution_provider_t provider, int num_threads, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_create_from_path(const char* dir_path, const lembed_text_options_t* options, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_embed(lembed_text_embedding_t* ctx, const char* const* texts, int num_texts, int batch_size, lembed_embeddings_t* result);
lembed_status_t lembed_text_embedding_embed_stream(lembed_text_embedding_t* ctx, const char* const* texts, int num_texts, int batch_size, void (*callback)(const float* embedding, int dim, void* userdata), void* userdata);
int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx);
const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx);
const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx);
int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx);
void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out);
void lembed_text_embedding_free(lembed_text_embedding_t* ctx);

/* Sparse text embedding */
lembed_status_t lembed_sparse_text_embedding_create(const lembed_sparse_options_t* options, lembed_sparse_embedding_ctx_t** out);
lembed_status_t lembed_sparse_text_embedding_create_from_path(const char* dir_path, const lembed_sparse_options_t* options, lembed_sparse_embedding_ctx_t** out);
lembed_status_t lembed_sparse_text_embedding_embed(lembed_sparse_embedding_ctx_t* ctx, const char* const* texts, int num_texts, int batch_size, lembed_sparse_embeddings_t* result);
const lembed_model_desc_t* lembed_sparse_text_embedding_desc(const lembed_sparse_embedding_ctx_t* ctx);
const char* lembed_sparse_text_embedding_model_name(const lembed_sparse_embedding_ctx_t* ctx);
int lembed_sparse_text_embedding_max_length(const lembed_sparse_embedding_ctx_t* ctx);
void lembed_sparse_text_embedding_stats(const lembed_sparse_embedding_ctx_t* ctx, lembed_stats_t* out);
void lembed_sparse_text_embedding_free(lembed_sparse_embedding_ctx_t* ctx);

/* Image embedding */
lembed_status_t lembed_image_embedding_create(const lembed_image_options_t* options, lembed_image_embedding_t** out);
lembed_status_t lembed_image_embedding_create_from_path(const char* dir_path, const lembed_image_options_t* options, lembed_image_embedding_t** out);
lembed_status_t lembed_image_embedding_embed_files(lembed_image_embedding_t* ctx, const char* const* file_paths, int num_images, int batch_size, lembed_embeddings_t* result);
lembed_status_t lembed_image_embedding_embed_bytes(lembed_image_embedding_t* ctx, const unsigned char* const* image_data, const int* image_sizes, int num_images, int batch_size, lembed_embeddings_t* result);
int lembed_image_embedding_dim(const lembed_image_embedding_t* ctx);
const lembed_model_desc_t* lembed_image_embedding_desc(const lembed_image_embedding_t* ctx);
const char* lembed_image_embedding_model_name(const lembed_image_embedding_t* ctx);
int lembed_image_embedding_max_length(const lembed_image_embedding_t* ctx);
void lembed_image_embedding_stats(const lembed_image_embedding_t* ctx, lembed_stats_t* out);
void lembed_image_embedding_free(lembed_image_embedding_t* ctx);

/* Reranker */
lembed_status_t lembed_reranker_create(const lembed_reranker_options_t* options, lembed_reranker_t** out);
lembed_status_t lembed_reranker_create_from_path(const char* dir_path, const lembed_reranker_options_t* options, lembed_reranker_t** out);
lembed_status_t lembed_reranker_rerank(lembed_reranker_t* ctx, const char* query, const char* const* documents, int num_documents, int batch_size, lembed_rerank_results_t* result);
const lembed_model_desc_t* lembed_reranker_desc(const lembed_reranker_t* ctx);
const char* lembed_reranker_model_name(const lembed_reranker_t* ctx);
int lembed_reranker_max_length(const lembed_reranker_t* ctx);
void lembed_reranker_stats(const lembed_reranker_t* ctx, lembed_stats_t* out);
void lembed_reranker_free(lembed_reranker_t* ctx);

/* Model registry */
lembed_status_t lembed_get_text_model_info(lembed_text_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_sparse_model_info(lembed_sparse_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_image_model_info(lembed_image_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_reranker_model_info(lembed_reranker_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_list_text_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_sparse_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_image_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_reranker_models(const lembed_model_info_t** out, int* count);
int lembed_find_text_model_by_code(const char* model_code);
int lembed_find_sparse_model_by_code(const char* model_code);

/* Downloader */
lembed_status_t lembed_ensure_text_model(lembed_text_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_sparse_model(lembed_sparse_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_image_model(lembed_image_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_reranker_model(lembed_reranker_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
void lembed_free_string(char* s);

/* Similarity */
float lembed_cosine_similarity(const float* a, const float* b, int dim);
float lembed_dot_product(const float* a, const float* b, int dim);
float lembed_euclidean_distance(const float* a, const float* b, int dim);

/* Memory free */
void lembed_embeddings_free(lembed_embeddings_t* result);
void lembed_sparse_embeddings_free(lembed_sparse_embeddings_t* result);
void lembed_rerank_results_free(lembed_rerank_results_t* result);

/* Autotuner */
typedef struct {
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
} lembed_tuning_result_t;

typedef enum {
    LEMBED_AUTOTUNE_QUICK = 0,
    LEMBED_AUTOTUNE_FULL = 1
} lembed_autotune_mode_t;

lembed_status_t lembed_autotune(const char* model_name, lembed_autotune_mode_t mode, lembed_tuning_result_t* result);
lembed_status_t lembed_autotune_custom(const char* model_name, const char* const* texts, int n_texts, lembed_autotune_mode_t mode, lembed_tuning_result_t* result);
void lembed_autotune_clear_cache(const char* model_name);

/* Auto model selection */
typedef struct {
    const char* model_code;
    const char* model_name;
    int dim;
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
    double score;
} lembed_model_selection_t;

lembed_status_t lembed_auto_select_model(const char* use_case, lembed_model_selection_t* result);
