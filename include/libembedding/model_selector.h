/*
 * libembedding - model_selector.h
 * Automatic model selection based on hardware and use case
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_MODEL_SELECTOR_H
#define LIBEMBEDDING_MODEL_SELECTOR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Use case for model selection */
typedef enum {
    LEMBED_USE_CASE_SPEED = 0,    /* Prioritize throughput */
    LEMBED_USE_CASE_QUALITY,      /* Prioritize accuracy */
    LEMBED_USE_CASE_BALANCED      /* Balance speed and quality */
} lembed_use_case_t;

/* Model candidate info */
typedef struct {
    char model_name[256];      /* e.g. "BAAI/bge-small-en-v1.5" */
    int dim;                   /* embedding dimension */
    int max_length;            /* max token length */
    int pooling;               /* LEMBED_POOLING_CLS or MEAN */
    int estimated_ram_mb;      /* estimated RAM usage */
    double estimated_throughput; /* estimated docs/sec */
} lembed_model_candidate_t;

/* Select the best model for given hardware and use case */
int lembed_model_select(
    int logical_cores,
    int ram_mb,
    lembed_use_case_t use_case,
    lembed_model_candidate_t* out_selected
);

/* Detect hardware capabilities */
typedef struct {
    char cpu_model[256];
    int physical_cores;
    int logical_cores;
    int ram_mb;
} lembed_hardware_info_t;

int lembed_detect_hardware(lembed_hardware_info_t* out_info);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_MODEL_SELECTOR_H */
