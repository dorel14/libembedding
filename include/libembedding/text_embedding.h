/*
 * libembedding - text_embedding.h
 * Dense text embedding C API
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TEXT_EMBEDDING_H
#define LIBEMBEDDING_TEXT_EMBEDDING_H

#include "types.h"

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
 * batch_size=0 means use default (256). */
lembed_status_t lembed_text_embedding_embed(
    lembed_text_embedding_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    lembed_embeddings_t* result);

/* Get embedding dimension */
int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx);

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

/* Internal struct definition */
struct lembed_text_embedding {
    lembed::detail::OnnxSession session;
    lembed::detail::TokenizerWrapper tokenizer;
    lembed_pooling_t pooling;
    lembed_quantization_t quantization;
    int dim;
    int max_length;
    std::string output_key; /* empty = use precedence */
};

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_text_embedding_create(
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        /* Get model info */
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_text_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        /* Ensure model files are downloaded */
        char* model_dir_cstr = nullptr;
        s = lembed_ensure_text_model(options->model, options->cache_dir,
                                     options->show_download_progress, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        /* Allocate context */
        auto* ctx = new lembed_text_embedding();
        ctx->pooling = (lembed_pooling_t)info.pooling;
        ctx->quantization = (lembed_quantization_t)info.quantization;
        ctx->dim = info.dim;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;

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

        /* Load ONNX from memory */
        ctx->session.load_from_memory(model->onnx_data, model->onnx_data_size,
                                      num_threads, (int)provider);

        /* Load tokenizer from memory */
        std::string tok_blob((const char*)model->tokenizer_json, model->tokenizer_json_size);
        ctx->tokenizer.load_from_blob(tok_blob, ctx->max_length);

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
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

    if (batch_size <= 0) batch_size = LEMBED_DEFAULT_BATCH_SIZE;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_texts;
        result->data = (float*)malloc((size_t)num_texts * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

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

            /* Flatten to contiguous arrays */
            int seq_len = enc.seq_length;
            std::vector<int64_t> ids_flat(bsz * seq_len);
            std::vector<int64_t> mask_flat(bsz * seq_len);
            std::vector<int64_t> type_flat(bsz * seq_len, 0);

            for (int i = 0; i < bsz; i++) {
                for (int j = 0; j < seq_len; j++) {
                    ids_flat[i * seq_len + j] = enc.input_ids[i][j];
                    mask_flat[i * seq_len + j] = enc.attention_mask[i][j];
                    if ((int)enc.token_type_ids[i].size() > j)
                        type_flat[i * seq_len + j] = enc.token_type_ids[i][j];
                }
            }

            /* Run ONNX */
            auto outputs = ctx->session.run(
                ids_flat.data(), mask_flat.data(), type_flat.data(),
                bsz, seq_len);

            /* Select output tensor */
            int oi = ctx->output_key.empty()
                ? ctx->session.select_output()
                : ctx->session.select_output(ctx->output_key.c_str());
            auto& output = outputs[oi];

            /* Determine tensor dimensions */
            int ndim = (int)output.shape.size();
            int out_batch = (int)output.shape[0];
            int out_seq = (ndim >= 3) ? (int)output.shape[1] : 0;
            int out_dim = (ndim >= 3) ? (int)output.shape[2] :
                          (ndim == 2) ? (int)output.shape[1] : dim;

            /* Pooling */
            std::vector<float> pooled(bsz * out_dim);
            if (ctx->pooling == LEMBED_POOLING_CLS) {
                lembed::detail::pool_cls(output.data.data(),
                    out_batch, out_seq, out_dim, ndim, pooled.data());
            } else {
                lembed::detail::pool_mean(output.data.data(), mask_flat.data(),
                    out_batch, out_seq, out_dim, ndim, pooled.data());
            }

            /* L2 normalize */
            lembed::detail::l2_normalize(pooled.data(), bsz, out_dim);

            /* Copy to output */
            std::memcpy(result->data + out_offset * dim,
                       pooled.data(), (size_t)bsz * dim * sizeof(float));
            out_offset += bsz;
        }

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

void lembed_text_embedding_free(lembed_text_embedding_t* ctx) {
    delete ctx;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_H */
