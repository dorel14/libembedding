/*
 * libembedding - detail/model_selector_impl.hpp
 * Model selection implementation skeleton
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_MODEL_SELECTOR_IMPL_HPP
#define LIBEMBEDDING_DETAIL_MODEL_SELECTOR_IMPL_HPP

#include "model_selector.h"
#include "model_registry.h"
#include "detail/downloader_impl.hpp"

#include <algorithm>
#include <cstring>

namespace lembed { namespace detail {

/* Candidate models for auto-selection (ordered by preference) */
static const struct {
    const char* model_code;
    int dim;
    int max_length;
    int pooling;
    int estimated_ram_mb;
    double estimated_throughput;
} CANDIDATE_MODELS[] = {
    /* Speed-first candidates (384-dim, small) */
    { "Xenova/all-MiniLM-L6-v2",           384,  256, LEMBED_POOLING_MEAN,    120,  600.0 },
    { "Xenova/bge-small-en-v1.5",           384,  512, LEMBED_POOLING_CLS,     150,  550.0 },
    { "Xenova/paraphrase-MiniLM-L12-v2",    384,  256, LEMBED_POOLING_MEAN,    130,  500.0 },
    /* Balanced candidates (384-768-dim) */
    { "Xenova/bge-base-en-v1.5",           768,  512, LEMBED_POOLING_CLS,     300,  400.0 },
    { "Xenova/gpt-small",                  768,  512, LEMBED_POOLING_MEAN,    320,  380.0 },
    /* Quality candidates (1024-dim) */
    { "Xenova/bge-large-en-v1.5",         1024,  512, LEMBED_POOLING_CLS,     600,  250.0 },
    { "Xenova/e5-large-v2",               1024,  512, LEMBED_POOLING_MEAN,    650,  220.0 },
};

inline int lembed_detect_hardware(lembed_hardware_info_t* out_info) {
    if (!out_info) return LEMBED_ERROR_INVALID_ARGUMENT;

    memset(out_info, 0, sizeof(*out_info));

#ifdef _WIN32
    /* Windows: use GetSystemInfo */
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    out_info->logical_cores = sysinfo.dwNumberOfProcessors;
    out_info->physical_cores = out_info->logical_cores;  /* approximation */
#elif defined(__APPLE__)
    /* macOS: sysctlbyname */
    int ncpu = 1;
    size_t len = sizeof(ncpu);
    sysctlbyname("hw.logicalcpu", &ncpu, &len, NULL, 0);
    out_info->logical_cores = ncpu;
    out_info->physical_cores = ncpu;
#else
    /* Linux: sysconf */
    out_info->logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    out_info->physical_cores = out_info->logical_cores;
#endif

    /* RAM detection (platform-specific) */
#ifdef _WIN32
    MEMORYSTATUSEX meminfo;
    meminfo.dwLength = sizeof(meminfo);
    if (GlobalMemoryStatusEx(&meminfo)) {
        out_info->ram_mb = (int)(meminfo.ullTotalPhys / (1024 * 1024));
    }
#elif defined(__APPLE__)
    int64_t mem = 0;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    out_info->ram_mb = (int)(mem / (1024 * 1024));
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    out_info->ram_mb = (int)((pages * page_size) / (1024 * 1024));
#endif

    /* CPU model string */
#ifdef _WIN32
    int cpuinfo[4] = {0};
    __cpuid(cpuinfo, 0x80000000);
    if (cpuinfo[0] >= 0x80000004) {
        __cpuid(cpuinfo, 0x80000002);
        memcpy(out_info->cpu_model, cpuinfo, 16);
        __cpuid(cpuinfo, 0x80000003);
        memcpy(out_info->cpu_model + 16, cpuinfo, 16);
        __cpuid(cpuinfo, 0x80000004);
        memcpy(out_info->cpu_model + 32, cpuinfo, 16);
    }
#elif defined(__APPLE__)
    size_t len = sizeof(out_info->cpu_model);
    sysctlbyname("machdep.cpu.brand_string", out_info->cpu_model, &len, NULL, 0);
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    strncpy(out_info->cpu_model, colon + 2, sizeof(out_info->cpu_model) - 1);
                    out_info->cpu_model[strcspn(out_info->cpu_model, "\n")] = 0;
                }
                break;
            }
        }
        fclose(f);
    }
#endif

    return LEMBED_OK;
}

inline int lembed_model_select(
    int logical_cores,
    int ram_mb,
    lembed_use_case_t use_case,
    lembed_model_candidate_t* out_selected) {

    if (!out_selected || logical_cores <= 0 || ram_mb <= 0)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    memset(out_selected, 0, sizeof(*out_selected));

    /* Filter candidates based on hardware */
    std::vector<int> candidates;
    int num_candidates = (int)(sizeof(CANDIDATE_MODELS) / sizeof(CANDIDATE_MODELS[0]));

    for (int i = 0; i < num_candidates; i++) {
        /* Skip models that don't fit in RAM */
        if (CANDIDATE_MODELS[i].estimated_ram_mb > ram_mb * 0.5)  /* max 50% of RAM */
            continue;

        /* Skip very large models on low-core machines */
        if (CANDIDATE_MODELS[i].dim >= 1024 && logical_cores < 4)
            continue;

        candidates.push_back(i);
    }

    if (candidates.empty())
        return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Select based on use case */
    int selected = candidates[0];

    if (use_case == LEMBED_USE_CASE_SPEED) {
        /* Pick highest throughput */
        double best_tp = 0;
        for (int idx : candidates) {
            if (CANDIDATE_MODELS[idx].estimated_throughput > best_tp) {
                best_tp = CANDIDATE_MODELS[idx].estimated_throughput;
                selected = idx;
            }
        }
    } else if (use_case == LEMBED_USE_CASE_QUALITY) {
        /* Pick highest dimension (largest model) */
        int best_dim = 0;
        for (int idx : candidates) {
            if (CANDIDATE_MODELS[idx].dim > best_dim) {
                best_dim = CANDIDATE_MODELS[idx].dim;
                selected = idx;
            }
        }
    } else {
        /* BALANCED: score = throughput / (dim / 384.0) */
        double best_score = 0;
        for (int idx : candidates) {
            double score = CANDIDATE_MODELS[idx].estimated_throughput /
                          (CANDIDATE_MODELS[idx].dim / 384.0);
            if (score > best_score) {
                best_score = score;
                selected = idx;
            }
        }
    }

    /* Fill output */
    strncpy(out_selected->model_name, CANDIDATE_MODELS[selected].model_code,
            sizeof(out_selected->model_name) - 1);
    out_selected->dim = CANDIDATE_MODELS[selected].dim;
    out_selected->max_length = CANDIDATE_MODELS[selected].max_length;
    out_selected->pooling = CANDIDATE_MODELS[selected].pooling;
    out_selected->estimated_ram_mb = CANDIDATE_MODELS[selected].estimated_ram_mb;
    out_selected->estimated_throughput = CANDIDATE_MODELS[selected].estimated_throughput;

    return LEMBED_OK;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_MODEL_SELECTOR_IMPL_HPP */
