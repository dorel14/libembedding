/*
 * libembedding - C Header-Only Embedding Library
 * error.h - Error codes and error reporting
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_ERROR_H
#define LIBEMBEDDING_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LEMBED_OK = 0,
    LEMBED_ERROR_INVALID_ARGUMENT,
    LEMBED_ERROR_OUT_OF_MEMORY,
    LEMBED_ERROR_ONNX_RUNTIME,
    LEMBED_ERROR_TOKENIZER,
    LEMBED_ERROR_DOWNLOAD,
    LEMBED_ERROR_IO,
    LEMBED_ERROR_MODEL_NOT_FOUND,
    LEMBED_ERROR_UNSUPPORTED,
    LEMBED_ERROR_BATCH_SIZE,
} lembed_status_t;

/* Return a static string for the given status code */
const char* lembed_status_message(lembed_status_t status);

/* Return thread-local detailed error string (set by the last failing call) */
const char* lembed_last_error(void);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_ERROR_IMPL
#define LIBEMBEDDING_ERROR_IMPL

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static _Thread_local char lembed__error_buf[1024] = {0};

const char* lembed_status_message(lembed_status_t status) {
    switch (status) {
        case LEMBED_OK:                     return "Success";
        case LEMBED_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case LEMBED_ERROR_OUT_OF_MEMORY:    return "Out of memory";
        case LEMBED_ERROR_ONNX_RUNTIME:     return "ONNX Runtime error";
        case LEMBED_ERROR_TOKENIZER:        return "Tokenizer error";
        case LEMBED_ERROR_DOWNLOAD:         return "Download error";
        case LEMBED_ERROR_IO:               return "I/O error";
        case LEMBED_ERROR_MODEL_NOT_FOUND:  return "Model not found";
        case LEMBED_ERROR_UNSUPPORTED:      return "Unsupported operation";
        case LEMBED_ERROR_BATCH_SIZE:       return "Invalid batch size for quantization mode";
        default:                            return "Unknown error";
    }
}

const char* lembed_last_error(void) {
    return lembed__error_buf;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

/* Internal helper to set the error message (C++ linkage for detail headers) */
#ifdef __cplusplus
#include <string>
namespace lembed { namespace detail {
    inline void set_error(const char* msg) {
        strncpy(lembed__error_buf, msg, sizeof(lembed__error_buf) - 1);
        lembed__error_buf[sizeof(lembed__error_buf) - 1] = '\0';
    }
    inline void set_error(const std::string& msg) {
        set_error(msg.c_str());
    }
}} /* namespace lembed::detail */
#else
static inline void lembed__set_error(const char* msg) {
    strncpy(lembed__error_buf, msg, sizeof(lembed__error_buf) - 1);
    lembed__error_buf[sizeof(lembed__error_buf) - 1] = '\0';
}
#endif

#endif /* LIBEMBEDDING_ERROR_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_ERROR_H */
