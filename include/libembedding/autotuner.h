/*
 * libembedding - autotuner.h
 * Auto-tuning for optimal embedding performance
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_AUTOTUNER_H
#define LIBEMBEDDING_AUTOTUNER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tuning result */
typedef struct {
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
} lembed_tuning_result_t;

/* Autotune modes */
typedef enum {
    LEMBED_AUTOTUNE_QUICK = 0,  /* 5-15s */
    LEMBED_AUTOTUNE_FULL        /* 30-120s */
} lembed_autotune_mode_t;

/* Run auto-tuning for a model
 * Returns optimal configuration for current hardware.
 * mode: QUICK (5-15s) or FULL (30-120s)
 * result: output configuration
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result);

/* Run auto-tuning with custom corpus
 * texts: array of text samples
 * n_texts: number of texts
 * avg_tokens: average token count (for calibration) */
lembed_status_t lembed_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result);

/* Clear autotune cache for a model (or all if model_name=NULL) */
void lembed_autotune_clear_cache(const char* model_name);

/* Auto model selection result */
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

/* Auto-select best model for hardware and use case
 * use_case: "speed", "quality", or "balanced" (default)
 * Returns optimal model + configuration */
lembed_status_t lembed_auto_select_model(
    const char* use_case,
    lembed_model_selection_t* result);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_AUTOTUNER_H */
