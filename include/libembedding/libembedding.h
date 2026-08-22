/*
 * libembedding - Single umbrella include
 *
 * Usage:
 *   // In exactly ONE .cpp file in your project:
 *   #define LIBEMBEDDING_IMPLEMENTATION
 *   #include <libembedding/libembedding.h>
 *
 *   // In all other files:
 *   #include <libembedding/libembedding.h>
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_H
#define LIBEMBEDDING_H

#include "config.h"
#include "error.h"
#include "types.h"
#include "model_registry.h"
#include "downloader.h"
#include "model_loader.h"
#include "text_embedding.h"
#include "sparse_text_embedding.h"

#ifndef LIBEMBEDDING_NO_IMAGE
#include "image_embedding.h"
#endif

#include "reranker.h"
#include "similarity.h"

#endif /* LIBEMBEDDING_H */
