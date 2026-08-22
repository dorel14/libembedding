/*
 * libembedding - downloader.h
 * Model download C API
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DOWNLOADER_H
#define LIBEMBEDDING_DOWNLOADER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Download/ensure a text model is available in cache.
 * Returns the cache directory path (caller must free with lembed_free_string). */
lembed_status_t lembed_ensure_text_model(
    lembed_text_model_t model,
    const char* cache_dir,      /* NULL = default */
    int show_progress,
    int offline,                /* 1 = skip downloads, cache only */
    char** model_dir_out);

lembed_status_t lembed_ensure_sparse_model(
    lembed_sparse_model_t model,
    const char* cache_dir,
    int show_progress,
    int offline,
    char** model_dir_out);

lembed_status_t lembed_ensure_image_model(
    lembed_image_model_t model,
    const char* cache_dir,
    int show_progress,
    int offline,
    char** model_dir_out);

lembed_status_t lembed_ensure_reranker_model(
    lembed_reranker_model_t model,
    const char* cache_dir,
    int show_progress,
    int offline,
    char** model_dir_out);

void lembed_free_string(char* s);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_DOWNLOADER_IMPL
#define LIBEMBEDDING_DOWNLOADER_IMPL

#include "model_registry.h"
#include "detail/downloader_impl.hpp"
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

/* Helper to get additional files for a model */
static const char* const* lembed__get_additional_files(int model_enum, int model_type) {
    for (int i = 0; i < lembed__additional_files_count; i++) {
        if (lembed__additional_files[i].model_enum == model_enum &&
            lembed__additional_files[i].model_type == model_type) {
            return lembed__additional_files[i].files;
        }
    }
    return NULL;
}

lembed_status_t lembed_ensure_text_model(
        lembed_text_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_model_info_t info;
    lembed_status_t s = lembed_get_text_model_info(model, &info);
    if (s != LEMBED_OK) return s;

    try {
        std::string dir = lembed::detail::ensure_model(
            info.model_code, info.model_file,
            lembed__get_additional_files((int)model, LEMBED__MODEL_TYPE_TEXT),
            lembed::detail::get_cache_dir(cache_dir),
            show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

lembed_status_t lembed_ensure_sparse_model(
        lembed_sparse_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_model_info_t info;
    lembed_status_t s = lembed_get_sparse_model_info(model, &info);
    if (s != LEMBED_OK) return s;

    try {
        std::string dir = lembed::detail::ensure_model(
            info.model_code, info.model_file,
            lembed__get_additional_files((int)model, LEMBED__MODEL_TYPE_SPARSE),
            lembed::detail::get_cache_dir(cache_dir),
            show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

lembed_status_t lembed_ensure_image_model(
        lembed_image_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_model_info_t info;
    lembed_status_t s = lembed_get_image_model_info(model, &info);
    if (s != LEMBED_OK) return s;

    try {
        std::string dir = lembed::detail::ensure_model(
            info.model_code, info.model_file,
            NULL,
            lembed::detail::get_cache_dir(cache_dir),
            show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

lembed_status_t lembed_ensure_reranker_model(
        lembed_reranker_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_model_info_t info;
    lembed_status_t s = lembed_get_reranker_model_info(model, &info);
    if (s != LEMBED_OK) return s;

    try {
        std::string dir = lembed::detail::ensure_model(
            info.model_code, info.model_file,
            lembed__get_additional_files((int)model, LEMBED__MODEL_TYPE_RERANKER),
            lembed::detail::get_cache_dir(cache_dir),
            show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

void lembed_free_string(char* s) {
    free(s);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_DOWNLOADER_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_DOWNLOADER_H */
