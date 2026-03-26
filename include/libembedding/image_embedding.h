/*
 * libembedding - image_embedding.h
 * Image embedding C API
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_IMAGE_EMBEDDING_H
#define LIBEMBEDDING_IMAGE_EMBEDDING_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

lembed_image_options_t lembed_image_options_default(void);

lembed_status_t lembed_image_embedding_create(
    const lembed_image_options_t* options,
    lembed_image_embedding_t** out);

/* Embed images from file paths */
lembed_status_t lembed_image_embedding_embed_files(
    lembed_image_embedding_t* ctx,
    const char* const* file_paths,
    int num_images,
    int batch_size,
    lembed_embeddings_t* result);

/* Embed images from memory buffers */
lembed_status_t lembed_image_embedding_embed_bytes(
    lembed_image_embedding_t* ctx,
    const unsigned char* const* image_data,
    const int* image_sizes,
    int num_images,
    int batch_size,
    lembed_embeddings_t* result);

int lembed_image_embedding_dim(const lembed_image_embedding_t* ctx);

void lembed_image_embedding_free(lembed_image_embedding_t* ctx);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_IMAGE_EMBEDDING_IMPL
#define LIBEMBEDDING_IMAGE_EMBEDDING_IMPL

#ifndef LIBEMBEDDING_NO_IMAGE

#include "model_registry.h"
#include "downloader.h"
#include "detail/onnx_session_impl.hpp"
#include "detail/image_preprocess.hpp"
#include "detail/normalize.hpp"
#include "detail/batch.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct lembed_image_embedding {
    lembed::detail::OnnxSession session;
    int dim;
    int target_size; /* image preprocessing target size */
};

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_image_embedding_create(
        const lembed_image_options_t* options,
        lembed_image_embedding_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_image_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        char* model_dir_cstr = nullptr;
        s = lembed_ensure_image_model(options->model, options->cache_dir,
                                      options->show_download_progress, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        auto* ctx = new lembed_image_embedding();
        ctx->dim = info.dim;
        ctx->target_size = 224; /* Standard for CLIP, ResNet, etc. */

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->session.load_from_file(onnx_path.c_str(),
                                    options->num_threads,
                                    (int)options->provider);

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_image_embedding_embed_files(
        lembed_image_embedding_t* ctx,
        const char* const* file_paths,
        int num_images,
        int batch_size,
        lembed_embeddings_t* result) {
    if (!ctx || !file_paths || num_images <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = 32;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_images;
        result->data = (float*)malloc((size_t)num_images * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

        int out_offset = 0;
        int num_batches = lembed::detail::batch_count(num_images, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_images, batch_size);
            int bsz = range.end - range.start;
            int ts = ctx->target_size;

            /* Preprocess images into [N, 3, H, W] tensor */
            std::vector<float> batch_data(bsz * 3 * ts * ts);
            for (int i = 0; i < bsz; i++) {
                auto img = lembed::detail::load_and_preprocess_image(
                    file_paths[range.start + i], ts);
                std::memcpy(batch_data.data() + i * 3 * ts * ts,
                           img.data.data(), 3 * ts * ts * sizeof(float));
            }

            /* Run ONNX with float input */
            int64_t shape[4] = { bsz, 3, ts, ts };

            /* Find the pixel_values input name (varies by model) */
            const char* input_name = "pixel_values";
            if (!ctx->session.has_input("pixel_values")) {
                if (ctx->session.has_input("input")) input_name = "input";
                else input_name = ctx->session.input_names()[0].c_str();
            }

            auto outputs = ctx->session.run_float(
                input_name, batch_data.data(), shape, 4);

            auto& output = outputs[0];
            int out_dim = (output.shape.size() >= 2)
                ? (int)output.shape[output.shape.size() - 1] : dim;

            /* L2 normalize */
            lembed::detail::l2_normalize(output.data.data(), bsz, out_dim);

            std::memcpy(result->data + out_offset * dim,
                       output.data.data(), (size_t)bsz * dim * sizeof(float));
            out_offset += bsz;
        }

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (result->data) { free(result->data); result->data = NULL; }
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_image_embedding_embed_bytes(
        lembed_image_embedding_t* ctx,
        const unsigned char* const* image_data,
        const int* image_sizes,
        int num_images,
        int batch_size,
        lembed_embeddings_t* result) {
    if (!ctx || !image_data || !image_sizes || num_images <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = 32;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_images;
        result->data = (float*)malloc((size_t)num_images * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

        int out_offset = 0;
        int num_batches = lembed::detail::batch_count(num_images, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_images, batch_size);
            int bsz = range.end - range.start;
            int ts = ctx->target_size;

            std::vector<float> batch_data(bsz * 3 * ts * ts);
            for (int i = 0; i < bsz; i++) {
                int idx = range.start + i;
                auto img = lembed::detail::load_and_preprocess_image_bytes(
                    image_data[idx], image_sizes[idx], ts);
                std::memcpy(batch_data.data() + i * 3 * ts * ts,
                           img.data.data(), 3 * ts * ts * sizeof(float));
            }

            int64_t shape[4] = { bsz, 3, ts, ts };
            const char* input_name = "pixel_values";
            if (!ctx->session.has_input("pixel_values")) {
                if (ctx->session.has_input("input")) input_name = "input";
                else input_name = ctx->session.input_names()[0].c_str();
            }

            auto outputs = ctx->session.run_float(
                input_name, batch_data.data(), shape, 4);

            auto& output = outputs[0];
            lembed::detail::l2_normalize(output.data.data(), bsz, dim);

            std::memcpy(result->data + out_offset * dim,
                       output.data.data(), (size_t)bsz * dim * sizeof(float));
            out_offset += bsz;
        }

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (result->data) { free(result->data); result->data = NULL; }
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

int lembed_image_embedding_dim(const lembed_image_embedding_t* ctx) {
    return ctx ? ctx->dim : 0;
}

void lembed_image_embedding_free(lembed_image_embedding_t* ctx) {
    delete ctx;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_NO_IMAGE */
#endif /* LIBEMBEDDING_IMAGE_EMBEDDING_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_IMAGE_EMBEDDING_H */
