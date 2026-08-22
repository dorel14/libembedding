/*
 * libembedding - C Header-Only Embedding Library
 * config.h - Version and feature configuration
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CONFIG_H
#define LIBEMBEDDING_CONFIG_H

#define LIBEMBEDDING_VERSION_MAJOR 0
#define LIBEMBEDDING_VERSION_MINOR 2
#define LIBEMBEDDING_VERSION_PATCH 0
#define LIBEMBEDDING_VERSION_STRING "0.2.0"

/* Feature toggles (can be defined before including headers) */

/* Define to disable model downloading (offline-only mode) */
/* #define LIBEMBEDDING_NO_DOWNLOAD */

/* Define to disable image embedding support */
/* #define LIBEMBEDDING_NO_IMAGE */

/* Default configuration values */
#define LEMBED_DEFAULT_BATCH_SIZE    256
#define LEMBED_DEFAULT_MAX_LENGTH    512
#define LEMBED_DEFAULT_CACHE_SUBDIR  "libembedding_cache"

#endif /* LIBEMBEDDING_CONFIG_H */
