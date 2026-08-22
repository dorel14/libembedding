/*
 * libembedding - cpp/provider.hpp
 * Shared C++ helpers for execution provider string mapping.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_PROVIDER_HPP
#define LIBEMBEDDING_CPP_PROVIDER_HPP

#include <libembedding/types.h>
#include <stdexcept>
#include <string>

namespace lembed {

inline lembed_execution_provider_t parse_provider(const std::string& name) {
    if (name == "cpu" || name.empty()) return LEMBED_PROVIDER_CPU;
    if (name == "cuda") return LEMBED_PROVIDER_CUDA;
    if (name == "coreml") return LEMBED_PROVIDER_COREML;
    if (name == "directml") return LEMBED_PROVIDER_DIRECTML;
    if (name == "tensorrt") return LEMBED_PROVIDER_TENSORRT;
    throw std::invalid_argument("Unknown execution provider: " + name);
}

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_PROVIDER_HPP */
