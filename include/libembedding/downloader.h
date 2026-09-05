/*
 * libembedding - downloader.h
 * Model download C API
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DOWNLOADER_H
#define LIBEMBEDDING_DOWNLOADER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lookup additional files for a model from the registry.
 * Declaration only — implementation in detail/downloader_impl.hpp */
const char* const* lembed__get_additional_files(int model_enum, int model_type);

/* Download/ensure a text model is available in cache */
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

lembed_status_t lembed_ensure_gguf_model(
    const char* repo,
    const char* filename,
    const char* cache_dir,      /* NULL = default */
    int show_progress,
    int offline,                /* 1 = skip downloads, cache only */
    char** model_path_out);

void lembed_free_string(char* s);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION

#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>
#include "detail/downloader_impl.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Lookup additional files for a model from the registry.
 * Returns nullptr if no additional files are needed. */
const char* const* lembed__get_additional_files(int model_enum, int model_type) {
    (void)model_enum;
    (void)model_type;
    return nullptr;
}

/* =========================================================================
 * Progress callback for GGUF downloads
 * ========================================================================= */
static void gguf_download_progress(float fraction, void* userdata) {
    (void)userdata;
    int bar_width = 40;
    int pos = (int)(bar_width * fraction);
    fprintf(stderr, "\r[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) fprintf(stderr, "=");
        else if (i == pos) fprintf(stderr, ">");
        else fprintf(stderr, " ");
    }
    fprintf(stderr, "] %3.0f%%", fraction * 100.0f);
    if (fraction >= 1.0f) fprintf(stderr, "\n");
    fflush(stderr);
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

lembed_status_t lembed_ensure_gguf_model(
        const char* repo, const char* filename,
        const char* cache_dir, int show_progress, int offline,
        char** model_path_out) {
    if (!repo || !repo[0] || !filename || !filename[0] || !model_path_out)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        std::string cache = lembed::detail::get_cache_dir(cache_dir);
        std::string repo_dir = lembed::detail::repo_to_dirname(repo);
        std::string model_dir = cache + "/models--" + repo_dir;
        lembed::detail::mkdirs(model_dir);

        std::string dest = model_dir + "/" + filename;

        /* Check if already cached */
        if (lembed::detail::file_exists(dest)) {
            *model_path_out = strdup(dest.c_str());
            return LEMBED_OK;
        }

        /* In offline mode, never attempt download */
        if (offline) {
            lembed::detail::set_error("GGUF model not in cache (offline mode): " + dest);
            return LEMBED_ERROR_DOWNLOAD;
        }

#ifndef LIBEMBEDDING_NO_DOWNLOAD
        /* Download the file with optional progress callback */
        lembed::detail::download_progress_fn pfn = show_progress ? &gguf_download_progress : nullptr;
        bool ok = lembed::detail::download_hf_file(repo, filename, dest, pfn, nullptr);
        if (!ok) {
            lembed::detail::set_error(std::string("Failed to download GGUF model: ") + repo + "/" + filename);
            return LEMBED_ERROR_DOWNLOAD;
        }
#else
        lembed::detail::set_error("GGUF model not in cache and downloading is disabled: " + dest);
        return LEMBED_ERROR_DOWNLOAD;
#endif

        *model_path_out = strdup(dest.c_str());
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

#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_DOWNLOADER_H */
