/*
 * libembedding - text_embedding.h
 * Dense text embedding C API
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TEXT_EMBEDDING_H
#define LIBEMBEDDING_TEXT_EMBEDDING_H

#include "types.h"
#include "model_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default-initialize text options */
lembed_text_options_t lembed_text_options_default(void);

/* Create a text embedding context (downloads model if needed) */
lembed_status_t lembed_text_embedding_create(
    const lembed_text_options_t* options,
    lembed_text_embedding_t** out);

/* Create from user-defined model (bring your own ONNX) */
lembed_status_t lembed_text_embedding_create_custom(
    const lembed_user_defined_model_t* model,
    lembed_execution_provider_t provider,
    int num_threads,
    lembed_text_embedding_t** out);

/* Embed texts. Result is a flat float array [num_texts * dim].
 * batch_size=0 means use the context's configured batch_size. */
lembed_status_t lembed_text_embedding_embed(
    lembed_text_embedding_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    lembed_embeddings_t* result);

/* Embed texts as a stream. Calls `callback` once per text with a pointer to
 * a dim-sized float array. batch_size=0 means use the context's configured
 * batch_size. Allows processing large numbers of texts without allocating
 * a single result buffer. */
lembed_status_t lembed_text_embedding_embed_stream(
    lembed_text_embedding_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    void (*callback)(const float* embedding, int dim, void* userdata),
    void* userdata);

/* Get embedding dimension */
int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx);

/* Introspection: runtime model descriptor */
const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx);
const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx);
int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx);

/* Runtime statistics */
void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out);

/* Destroy context */
void lembed_text_embedding_free(lembed_text_embedding_t* ctx);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_TEXT_EMBEDDING_IMPL
#define LIBEMBEDDING_TEXT_EMBEDDING_IMPL

#include "model_registry.h"
#include "downloader.h"
#include "detail/onnx_session_impl.hpp"
#include "detail/tokenizer_impl.hpp"
#include "detail/pooling.hpp"
#include "detail/normalize.hpp"
#include "detail/batch.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

/* Internal struct definition */
struct lembed_text_embedding {
    lembed::detail::OnnxSession session;
    lembed::detail::TokenizerWrapper tokenizer;
    lembed_pooling_t pooling;
    lembed_quantization_t quantization;
    int dim;
    int max_length;
    std::string output_key; /* empty = use precedence */

    /* Runtime metadata for introspection */
    std::string model_name_str;
    int num_threads;
    int batch_size;
    lembed_execution_provider_t provider;
    int device_id;
    lembed_model_desc_t desc; /* desc.name points to model_name_str.c_str() */

    /* Stats counters */
    uint64_t texts_embedded = 0;
    uint64_t batches_run = 0;
    double   total_latency_ms = 0.0;
    int      stats_calls = 0;

    /* Reusable per-batch buffers (avoids repeated allocation) */
    std::vector<int64_t> ids_buf_;
    std::vector<int64_t> mask_buf_;
    std::vector<int64_t> type_buf_;
    std::vector<float> pooled_buf_;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Helper: populate the cached desc struct on a context */
static void lembed__text_update_desc(lembed_text_embedding_t* ctx) {
    if (!ctx) return;
    ctx->desc.name = ctx->model_name_str.c_str();
    ctx->desc.dimension = ctx->dim;
    ctx->desc.max_length = ctx->max_length;
    ctx->desc.pooling = ctx->pooling;
    ctx->desc.num_threads = ctx->num_threads;
    ctx->desc.batch_size = ctx->batch_size;
    ctx->desc.provider = ctx->provider;
    ctx->desc.device_id = ctx->device_id;
}

lembed_status_t lembed_text_embedding_create(
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        /* Get model info */
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_text_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        /* Ensure model files are downloaded (offline flag skips download) */
        char* model_dir_cstr = nullptr;
        s = lembed_ensure_text_model(options->model, options->cache_dir,
                                     options->show_download_progress,
                                     options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        /* Allocate context */
        auto* ctx = new lembed_text_embedding();
        ctx->pooling = (lembed_pooling_t)info.pooling;
        ctx->quantization = (lembed_quantization_t)info.quantization;
        ctx->dim = info.dim;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;
        ctx->model_name_str = info.model_name;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        /* Special output key for EmbeddingGemma */
        if (options->model == LEMBED_TEXT_EMBEDDING_GEMMA_300M) {
            ctx->output_key = "sentence_embedding";
        }

        /* Load ONNX session */
        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->session.load_from_file(onnx_path.c_str(),
                                    options->num_threads,
                                    (int)options->provider);

        /* Load tokenizer */
        std::string tok_path = model_dir + "/tokenizer.json";
        ctx->tokenizer.load_from_file(tok_path, ctx->max_length);

        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_text_embedding_create_custom(
        const lembed_user_defined_model_t* model,
        lembed_execution_provider_t provider,
        int num_threads,
        lembed_text_embedding_t** out) {
    if (!model || !out || !model->onnx_data || !model->tokenizer_json)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        auto* ctx = new lembed_text_embedding();
        ctx->pooling = model->pooling;
        ctx->quantization = LEMBED_QUANTIZATION_NONE;
        ctx->dim = model->dim;
        ctx->max_length = (model->max_length > 0) ? model->max_length : LEMBED_DEFAULT_MAX_LENGTH;
        ctx->model_name_str = "custom-model";
        ctx->num_threads = num_threads;
        ctx->batch_size = LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = provider;
        ctx->device_id = 0;

        /* Load ONNX from memory */
        ctx->session.load_from_memory(model->onnx_data, model->onnx_data_size,
                                      num_threads, (int)provider);

        /* Load tokenizer from memory */
        std::string tok_blob((const char*)model->tokenizer_json, model->tokenizer_json_size);
        ctx->tokenizer.load_from_blob(tok_blob, ctx->max_length);

        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_text_embedding_create_from_path(
        const char* dir_path,
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!dir_path || !dir_path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        /* Read model.onnx */
        std::string onnx_path = std::string(dir_path) + "/model.onnx";
        std::string onnx_data = lembed::detail::read_file_to_string(onnx_path);

        /* Read tokenizer.json */
        std::string tok_path = std::string(dir_path) + "/tokenizer.json";
        std::string tok_data = lembed::detail::read_file_to_string(tok_path);

        /* Read config.json (optional) */
        std::string config_path = std::string(dir_path) + "/config.json";
        std::string config_data;
        bool has_config = lembed::detail::file_exists(config_path);
        if (has_config) {
            config_data = lembed::detail::read_file_to_string(config_path);
        }

        /* Resolve dim, max_length, pooling */
        lembed_user_defined_model_t udm;
        memset(&udm, 0, sizeof(udm));
        udm.onnx_data = (const unsigned char*)onnx_data.data();
        udm.onnx_data_size = onnx_data.size();
        udm.tokenizer_json = (const unsigned char*)tok_data.data();
        udm.tokenizer_json_size = tok_data.size();

        int dim = 0;
        int max_length = 0;
        lembed_pooling_t pooling = LEMBED_POOLING_MEAN;

        if (has_config) {
            lembed::detail::parse_config_json(config_data, &dim, &max_length);
            pooling = lembed::detail::infer_pooling_from_path(dir_path);
        }

        /* Fallback to options */
        if (dim == 0) dim = options->dim;
        if (max_length == 0) max_length = options->max_length;
        if (options->pooling == LEMBED_POOLING_CLS) pooling = LEMBED_POOLING_CLS;

        if (dim == 0) {
            lembed::detail::set_error(
                "Cannot determine model dimension: no config.json found and "
                "options.dim not set");
            return LEMBED_ERROR_INVALID_ARGUMENT;
        }

        udm.dim = dim;
        udm.pooling = pooling;
        udm.max_length = max_length;

        /* Create via custom (loads from memory) */
        lembed_text_embedding_t* ctx = nullptr;
        lembed_status_t s = lembed_text_embedding_create_custom(
            &udm, options->provider, options->num_threads, &ctx);
        if (s != LEMBED_OK) return s;

        /* Override name and batch_size for local path */
        ctx->model_name_str = dir_path;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_IO;
    }
}

lembed_status_t lembed_text_embedding_embed(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        lembed_embeddings_t* result) {
    if (!ctx || !texts || num_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Dynamic quantization: batch_size must be >= num_texts */
    if (ctx->quantization == LEMBED_QUANTIZATION_DYNAMIC) {
        if (batch_size > 0 && batch_size < num_texts) {
            lembed::detail::set_error(
                "Dynamic quantization cannot be used with batching smaller than total texts");
            return LEMBED_ERROR_BATCH_SIZE;
        }
        batch_size = num_texts;
    }

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_texts;
        result->data = (float*)malloc((size_t)num_texts * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

        auto t_start = std::chrono::high_resolution_clock::now();

        int out_offset = 0;

        int num_batches = lembed::detail::batch_count(num_texts, batch_size);
        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_texts, batch_size);
            int bsz = range.end - range.start;

            /* Tokenize batch */
            std::vector<std::string> batch_texts;
            batch_texts.reserve(bsz);
            for (int i = range.start; i < range.end; i++) {
                batch_texts.push_back(texts[i]);
            }

            auto enc = ctx->tokenizer.encode_batch(batch_texts);

            /* Flatten to contiguous arrays (reuse pre-allocated buffers) */
            int seq_len = enc.seq_length;
            size_t flat_size = (size_t)bsz * seq_len;
            ctx->ids_buf_.resize(flat_size);
            ctx->mask_buf_.resize(flat_size);
            ctx->type_buf_.assign(flat_size, 0);

            for (int i = 0; i < bsz; i++) {
                for (int j = 0; j < seq_len; j++) {
                    ctx->ids_buf_[i * seq_len + j] = enc.input_ids[i][j];
                    ctx->mask_buf_[i * seq_len + j] = enc.attention_mask[i][j];
                    if ((int)enc.token_type_ids[i].size() > j)
                        ctx->type_buf_[i * seq_len + j] = enc.token_type_ids[i][j];
                }
            }

            /* Select output tensor index before run */
            int oi = ctx->output_key.empty()
                ? ctx->session.select_output()
                : ctx->session.select_output(ctx->output_key.c_str());

            /* Run ONNX (request only the needed output) */
            auto outputs = ctx->session.run(
                ctx->ids_buf_.data(), ctx->mask_buf_.data(), ctx->type_buf_.data(),
                bsz, seq_len, oi);
            auto& output = outputs[0];

            /* Determine tensor dimensions */
            int ndim = (int)output.shape.size();
            int out_batch = (int)output.shape[0];
            int out_seq = (ndim >= 3) ? (int)output.shape[1] : 0;
            int out_dim = (ndim >= 3) ? (int)output.shape[2] :
                        (ndim == 2) ? (int)output.shape[1] : dim;

            /* Pooling (reuse pre-allocated buffer) */
            ctx->pooled_buf_.resize((size_t)bsz * out_dim);
            if (ctx->pooling == LEMBED_POOLING_CLS) {
                lembed::detail::pool_cls(output.data.data(),
                    out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
            } else {
                lembed::detail::pool_mean(output.data.data(), ctx->mask_buf_.data(),
                    out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
            }

            /* L2 normalize */
            lembed::detail::l2_normalize(ctx->pooled_buf_.data(), bsz, out_dim);

            /* Copy to output */
            std::memcpy(result->data + out_offset * dim,
                       ctx->pooled_buf_.data(), (size_t)bsz * dim * sizeof(float));
            out_offset += bsz;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ctx->texts_embedded += num_texts;
        ctx->batches_run += num_batches;
        ctx->total_latency_ms += elapsed_ms;
        ctx->stats_calls++;

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (result->data) { free(result->data); result->data = NULL; }
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx) {
    return ctx ? ctx->dim : 0;
}

lembed_status_t lembed_text_embedding_embed_stream(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        void (*callback)(const float* embedding, int dim, void* userdata),
        void* userdata) {
    if (!ctx || !texts || num_texts <= 0 || !callback)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        int dim = ctx->dim;
        int num_batches = lembed::detail::batch_count(num_texts, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_texts, batch_size);
            int bsz = range.end - range.start;

            std::vector<std::string> batch_texts;
            batch_texts.reserve(bsz);
            for (int i = range.start; i < range.end; i++) {
                batch_texts.push_back(texts[i]);
            }

            auto enc = ctx->tokenizer.encode_batch(batch_texts);
            int seq_len = enc.seq_length;
            size_t flat_size = (size_t)bsz * seq_len;
            ctx->ids_buf_.resize(flat_size);
            ctx->mask_buf_.resize(flat_size);
            ctx->type_buf_.assign(flat_size, 0);

            for (int i = 0; i < bsz; i++) {
                for (int j = 0; j < seq_len; j++) {
                    ctx->ids_buf_[i * seq_len + j] = enc.input_ids[i][j];
                    ctx->mask_buf_[i * seq_len + j] = enc.attention_mask[i][j];
                    if ((int)enc.token_type_ids[i].size() > j)
                        ctx->type_buf_[i * seq_len + j] = enc.token_type_ids[i][j];
                }
            }

            int oi = ctx->output_key.empty()
                ? ctx->session.select_output()
                : ctx->session.select_output(ctx->output_key.c_str());

            auto outputs = ctx->session.run(
                ctx->ids_buf_.data(), ctx->mask_buf_.data(), ctx->type_buf_.data(),
                bsz, seq_len, oi);

            auto& output = outputs[0];
            int ndim = (int)output.shape.size();
            int out_batch = (int)output.shape[0];
            int out_seq = (ndim >= 3) ? (int)output.shape[1] : 0;
            int out_dim = (ndim >= 3) ? (int)output.shape[2] :
                        (ndim == 2) ? (int)output.shape[1] : dim;

            ctx->pooled_buf_.resize((size_t)bsz * out_dim);
            if (ctx->pooling == LEMBED_POOLING_CLS) {
                lembed::detail::pool_cls(output.data.data(),
                    out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
            } else {
                lembed::detail::pool_mean(output.data.data(), ctx->mask_buf_.data(),
                    out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
            }

            lembed::detail::l2_normalize(ctx->pooled_buf_.data(), bsz, out_dim);

            /* Call callback for each embedding in the batch */
            for (int i = 0; i < bsz; i++) {
                callback(ctx->pooled_buf_.data() + (size_t)i * out_dim, dim, userdata);
            }
        }

        ctx->texts_embedded += num_texts;
        ctx->batches_run += num_batches;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx) {
    return ctx ? &ctx->desc : nullptr;
}

const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx) {
    return ctx ? ctx->model_name_str.c_str() : nullptr;
}

int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx) {
    return ctx ? ctx->max_length : 0;
}

void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out) {
    if (!out) return;
    if (!ctx) {
        memset(out, 0, sizeof(*out));
        return;
    }
    out->texts_embedded = ctx->texts_embedded;
    out->batches_run = ctx->batches_run;
    out->avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
}

void lembed_text_embedding_free(lembed_text_embedding_t* ctx) {
    delete ctx;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_H */
