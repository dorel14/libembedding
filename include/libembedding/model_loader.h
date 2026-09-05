/*
 * libembedding - model_loader.h
 * Local model loading from directory paths (create_from_path API)
 *
 * Convention: dir_path contains model.onnx, tokenizer.json, and optionally
 * config.json for metadata.
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_MODEL_LOADER_H
#define LIBEMBEDDING_MODEL_LOADER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load a text embedding model from a local directory.
 * The directory should contain: model.onnx, tokenizer.json, and optionally
 * config.json (for auto-detecting dimension, max_length, pooling).
 *
 * If config.json is absent, dim/pooling must be specified in options. */
lembed_status_t lembed_text_embedding_create_from_path(
    const char* dir_path,
    const lembed_text_options_t* options,
    lembed_text_embedding_t** out);

/* Load a sparse embedding model from a local directory. */
lembed_status_t lembed_sparse_text_embedding_create_from_path(
    const char* dir_path,
    const lembed_sparse_options_t* options,
    lembed_sparse_embedding_ctx_t** out);

/* Load an image embedding model from a local directory. */
lembed_status_t lembed_image_embedding_create_from_path(
    const char* dir_path,
    const lembed_image_options_t* options,
    lembed_image_embedding_t** out);

/* Load a reranker model from a local directory. */
lembed_status_t lembed_reranker_create_from_path(
    const char* dir_path,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_MODEL_LOADER_IMPL
#define LIBEMBEDDING_MODEL_LOADER_IMPL

#include "cJSON.h"
#include "detail/downloader_impl.hpp"

#include <string>

namespace lembed { namespace detail {

/*
 * Parse config.json blob to extract model dimension and max position embeddings.
 * Uses cJSON (bundled). Returns true if the blob was parsed successfully.
 */
inline bool parse_config_json(const std::string& config_blob,
                              int* out_dim, int* out_max_length) {
    cJSON* root = cJSON_Parse(config_blob.c_str());
    if (!root) return false;

    cJSON* hs = cJSON_GetObjectItem(root, "hidden_size");
    if (cJSON_IsNumber(hs) && hs->valueint > 0) {
        *out_dim = hs->valueint;
    }

    cJSON* mpe = cJSON_GetObjectItem(root, "max_position_embeddings");
    if (cJSON_IsNumber(mpe) && mpe->valueint > 0) {
        *out_max_length = mpe->valueint;
    }

    cJSON_Delete(root);
    return true;
}

/*
 * Infer pooling strategy from the directory path basename.
 * Heuristic: BGE, Snowflake Arctic, GTE, MXBAI models use CLS pooling.
 * All others default to MEAN.
 */
inline lembed_pooling_t infer_pooling_from_path(const std::string& path) {
    std::string basename = path;
    size_t pos = basename.find_last_of("/\\");
    if (pos != std::string::npos) basename = basename.substr(pos + 1);
    for (auto& c : basename) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }

    if (basename.find("bge") != std::string::npos ||
        basename.find("snowflake") != std::string::npos ||
        basename.find("arctic") != std::string::npos ||
        basename.find("gte") != std::string::npos ||
        basename.find("mxbai") != std::string::npos) {
        return LEMBED_POOLING_CLS;
    }
    return LEMBED_POOLING_MEAN;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_MODEL_LOADER_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_MODEL_LOADER_H */




