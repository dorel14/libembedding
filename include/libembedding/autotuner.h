/*
 * libembedding - autotuner.h
 * Auto-tuning API for finding optimal workers/threads/batch_size
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
    char model_name[256];
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
} lembed_tuning_result_t;

/* Autotune mode */
typedef enum {
    LEMBED_AUTOTUNE_QUICK = 0,   /* 5-15s */
    LEMBED_AUTOTUNE_FULL         /* 30-120s */
} lembed_autotune_mode_t;

/* Run autotune for a specific model */
int lembed_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result
);

/* Auto-select model + run autotune (model_name = "auto") */
int lembed_autotune_auto(
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result
);

/* Get cached autotune result if available */
int lembed_autotune_cached(
    const char* model_name,
    lembed_tuning_result_t* result
);

/* Save autotune result to cache */
int lembed_autotune_save_cache(
    const char* model_name,
    const lembed_tuning_result_t* result
);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_AUTOTUNER_H */
