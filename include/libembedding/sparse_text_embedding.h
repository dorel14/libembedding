/*
 * libembedding - sparse_text_embedding.h
 * Sparse text embedding C API (SPLADE, BGE-M3)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_H
#define LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_H

#include "types.h"
#include "model_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

lembed_sparse_options_t lembed_sparse_options_default(void);

lembed_status_t lembed_sparse_text_embedding_create(
    const lembed_sparse_options_t* options,
    lembed_sparse_embedding_ctx_t** out);

lembed_status_t lembed_sparse_text_embedding_embed(
    lembed_sparse_embedding_ctx_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    lembed_sparse_embeddings_t* result);

/* Introspection */
const lembed_model_desc_t* lembed_sparse_text_embedding_desc(const lembed_sparse_embedding_ctx_t* ctx);
const char* lembed_sparse_text_embedding_model_name(const lembed_sparse_embedding_ctx_t* ctx);
int lembed_sparse_text_embedding_max_length(const lembed_sparse_embedding_ctx_t* ctx);

void lembed_sparse_text_embedding_free(lembed_sparse_embedding_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_IMPL
#define LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_IMPL

#include "model_registry.h"
#include "downloader.h"
#include "detail/onnx_session_impl.hpp"
#include "detail/tokenizer_impl.hpp"
#include "detail/sparse_postprocess.hpp"
#include "detail/batch.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct lembed_sparse_embedding {
    lembed::detail::OnnxSession session;
    lembed::detail::TokenizerWrapper tokenizer;
    lembed_sparse_model_t model;
    int max_length;

    /* Runtime metadata for introspection */
    std::string model_name_str;
    int num_threads;
    int batch_size;
    lembed_execution_provider_t provider;
    int device_id;
    lembed_model_desc_t desc;
};

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_sparse_text_embedding_create(
        const lembed_sparse_options_t* options,
        lembed_sparse_embedding_ctx_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_sparse_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        char* model_dir_cstr = nullptr;
        s = lembed_ensure_sparse_model(options->model, options->cache_dir,
                                       options->show_download_progress,
                                       options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        auto* ctx = new lembed_sparse_embedding();
        ctx->model = options->model;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;
        ctx->model_name_str = info.model_name;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->session.load_from_file(onnx_path.c_str(),
                                    options->num_threads,
                                    (int)options->provider);

        std::string tok_path = model_dir + "/tokenizer.json";
        ctx->tokenizer.load_from_file(tok_path, ctx->max_length);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = 0;
        ctx->desc.max_length = ctx->max_length;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_sparse_text_embedding_create_from_path(
        const char* dir_path,
        const lembed_sparse_options_t* options,
        lembed_sparse_embedding_ctx_t** out) {
    if (!dir_path || !dir_path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        /* Read files from directory */
        std::string onnx_data = lembed::detail::read_file_to_string(
            std::string(dir_path) + "/model.onnx");
        std::string tok_data = lembed::detail::read_file_to_string(
            std::string(dir_path) + "/tokenizer.json");

        /* Read config.json (optional) */
        std::string config_path = std::string(dir_path) + "/config.json";
        bool has_config = lembed::detail::file_exists(config_path);
        int dim = 0, max_len = 0;
        if (has_config) {
            std::string config_data = lembed::detail::read_file_to_string(config_path);
            lembed::detail::parse_config_json(config_data, &dim, &max_len);
        }

        int max_length = (max_len > 0) ? max_len :
                         (options->max_length > 0) ? options->max_length
                                                     : LEMBED_DEFAULT_MAX_LENGTH;

        auto* ctx = new lembed_sparse_embedding();
        ctx->model = LEMBED_SPARSE_MODEL_COUNT; /* sentinel: local model */
        ctx->max_length = max_length;
        ctx->model_name_str = dir_path;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        /* Load ONNX from memory */
        ctx->session.load_from_memory(
            (const void*)onnx_data.data(), onnx_data.size(),
            options->num_threads, (int)options->provider);

        /* Load tokenizer from memory */
        ctx->tokenizer.load_from_blob(tok_data, ctx->max_length);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = 0;
        ctx->desc.max_length = ctx->max_length;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_IO;
    }
}

lembed_status_t lembed_sparse_text_embedding_embed(
        lembed_sparse_embedding_ctx_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        lembed_sparse_embeddings_t* result) {
    if (!ctx || !texts || num_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        result->count = num_texts;
        result->items = (lembed_sparse_embedding_t*)calloc(
            num_texts, sizeof(lembed_sparse_embedding_t));
        if (!result->items) return LEMBED_ERROR_OUT_OF_MEMORY;

        int out_offset = 0;
        int num_batches = lembed::detail::batch_count(num_texts, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_texts, batch_size);
            int bsz = range.end - range.start;

            std::vector<std::string> batch_texts;
            for (int i = range.start; i < range.end; i++)
                batch_texts.push_back(texts[i]);

            auto enc = ctx->tokenizer.encode_batch(batch_texts);
            int seq_len = enc.seq_length;

            std::vector<int64_t> ids_flat(bsz * seq_len);
            std::vector<int64_t> mask_flat(bsz * seq_len);
            std::vector<int64_t> type_flat(bsz * seq_len, 0);

            for (int i = 0; i < bsz; i++) {
                for (int j = 0; j < seq_len; j++) {
                    ids_flat[i * seq_len + j] = enc.input_ids[i][j];
                    mask_flat[i * seq_len + j] = enc.attention_mask[i][j];
                }
            }

            auto outputs = ctx->session.run(
                ids_flat.data(), mask_flat.data(), type_flat.data(),
                bsz, seq_len);

            /* SPLADE: output is [batch, seq_len, vocab_size] */
            auto& output = outputs[0];
            int vocab_size = (output.shape.size() >= 3)
                ? (int)output.shape[2]
                : (int)output.shape[1];

            auto sparse_results = lembed::detail::splade_postprocess(
                output.data.data(), mask_flat.data(),
                bsz, seq_len, vocab_size);

            /* Copy to C output */
            for (int i = 0; i < bsz; i++) {
                auto& sr = sparse_results[i];
                int idx = out_offset + i;
                result->items[idx].length = (int)sr.indices.size();
                result->items[idx].indices = (int32_t*)malloc(
                    sr.indices.size() * sizeof(int32_t));
                result->items[idx].values = (float*)malloc(
                    sr.values.size() * sizeof(float));
                if (result->items[idx].indices && result->items[idx].values) {
                    std::memcpy(result->items[idx].indices, sr.indices.data(),
                               sr.indices.size() * sizeof(int32_t));
                    std::memcpy(result->items[idx].values, sr.values.data(),
                               sr.values.size() * sizeof(float));
                }
            }
            out_offset += bsz;
        }

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        lembed_sparse_embeddings_free(result);
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

const lembed_model_desc_t* lembed_sparse_text_embedding_desc(const lembed_sparse_embedding_ctx_t* ctx) {
    return ctx ? &ctx->desc : nullptr;
}

const char* lembed_sparse_text_embedding_model_name(const lembed_sparse_embedding_ctx_t* ctx) {
    return ctx ? ctx->model_name_str.c_str() : nullptr;
}

int lembed_sparse_text_embedding_max_length(const lembed_sparse_embedding_ctx_t* ctx) {
    return ctx ? ctx->max_length : 0;
}

void lembed_sparse_text_embedding_free(lembed_sparse_embedding_ctx_t* ctx) {
    delete ctx;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_H */
