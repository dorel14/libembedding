#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

/* Autotuner C API implementation */
#include "libembedding/detail/autotuner_impl.hpp"

#include <cstdio>

lembed_status_t lembed_autotune(
        const char* model_name,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    int idx = lembed_find_text_model_by_code(model_name);
    if (idx < 0) return LEMBED_ERROR_MODEL_NOT_FOUND;

    return lembed::detail::autotune_impl(
        (lembed_text_model_t)idx,
        {},  /* use generated corpus */
        mode,
        result
    );
}

lembed_status_t lembed_autotune_custom(
        const char* model_name,
        const char* const* texts,
        int n_texts,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    int idx = lembed_find_text_model_by_code(model_name);
    if (idx < 0) return LEMBED_ERROR_MODEL_NOT_FOUND;

    std::vector<std::string> corpus;
    corpus.reserve(n_texts);
    for (int i = 0; i < n_texts; i++)
        corpus.push_back(texts[i]);

    return lembed::detail::autotune_impl(
        (lembed_text_model_t)idx,
        corpus,
        mode,
        result
    );
}

#include "libembedding/detail/model_selector.hpp"

lembed_status_t lembed_auto_select_model(
        const char* use_case,
        lembed_model_selection_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    if (!use_case) use_case = "balanced";

    lembed::detail::ModelSelection sel;
    lembed_status_t s = lembed::detail::auto_select_impl(use_case, sel);
    if (s != LEMBED_OK) return s;

    result->model_code = sel.model_code;
    result->model_name = sel.model_name;
    result->dim = sel.dim;
    result->workers = sel.workers;
    result->threads = sel.threads;
    result->batch_size = sel.batch_size;
    result->throughput_docs_sec = sel.throughput_docs_sec;
    result->latency_ms = sel.latency_ms;
    result->memory_mb = sel.memory_mb;
    result->score = sel.score;

    return LEMBED_OK;
}

/* Autotune cache management */
void lembed_autotune_clear_cache(const char* model_name) {
    lembed::detail::clear_cache(model_name);
}

/* Sparse auto-tuner wrapper */
lembed_status_t lembed_sparse_autotune(
        const char* model_name,
        lembed_autotune_mode_t mode,
        lembed_sparse_tuning_result_t* result);

/* Force linker to include sparse_autotune */
static const void* const _sparse_autotune_ref = (const void*)&lembed_sparse_autotune;

lembed_status_t lembed_sparse_autotune(
        const char* model_name,
        lembed_autotune_mode_t mode,
        lembed_sparse_tuning_result_t* result) {
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    return lembed::detail::lembed_sparse_autotune(model_name, mode, result);
}
