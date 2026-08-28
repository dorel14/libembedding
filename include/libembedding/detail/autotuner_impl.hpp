/*
 * libembedding - detail/autotuner_impl.hpp
 * Auto-tuning implementation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP

#include "libembedding/autotuner.h"
#include "libembedding/text_embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lembed { namespace detail {

/* Get number of logical CPU cores */
static int cpu_logical_cores() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

/* =========================================================================
 * Cache system for autotune results
 * ========================================================================= */

static std::string autotune_cache_dir() {
#ifdef _WIN32
    const char* local_appdata = std::getenv("LOCALAPPDATA");
    if (local_appdata) {
        return std::string(local_appdata) + "\\libembedding\\autotune";
    }
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        return std::string(userprofile) + "\\AppData\\Local\\libembedding\\autotune";
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.cache/libembedding/autotune";
    }
#endif
    return "./libembedding_autotune_cache";
}

static std::string get_cache_key(const char* model_name) {
    /* Key: CPU_cores_model_ort_version_lib_version
     * Plan: CPU model + logical_cores + physical_cores + model_name + libembedding_version + ORT_version
     */
    int logical_cores = cpu_logical_cores();

    /* Get physical cores */
    int physical_cores = logical_cores;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    /* dwNumberOfProcessors is logical; need to calculate physical */
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len > 0) {
        auto* buf = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(len);
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
            int n_phys = 0;
            uint8_t* ptr = (uint8_t*)buf;
            uint8_t* end = ptr + len;
            while (ptr < end) {
                auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
                if (info->Relationship == RelationProcessorCore) {
                    n_phys++;
                }
                ptr += info->Size;
            }
            if (n_phys > 0) physical_cores = n_phys;
        }
        free(buf);
    }
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        int max_phys = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "cpu cores", 9) == 0) {
                int n = 0;
                if (sscanf(line, "cpu cores : %d", &n) == 1 && n > max_phys)
                    max_phys = n;
            }
        }
        if (max_phys > 0) physical_cores = max_phys;
        fclose(f);
    }
#endif

    /* Get CPU brand string */
    char cpu_brand[64] = "unknown";
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD size = sizeof(cpu_brand);
        RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
                        (LPBYTE)cpu_brand, &size);
        RegCloseKey(hKey);
    }
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    strncpy(cpu_brand, colon + 2, sizeof(cpu_brand) - 1);
                    cpu_brand[sizeof(cpu_brand) - 1] = '\0';
                    /* Remove newline */
                    char* nl = strchr(cpu_brand, '\n');
                    if (nl) *nl = '\0';
                }
                break;
            }
        }
        fclose(f);
    }
#endif

    /* Sanitize: replace spaces, slashes and special chars */
    for (int i = 0; cpu_brand[i]; i++) {
        char c = cpu_brand[i];
        if (c == ' ' || c == '(' || c == ')' || c == '@' || c == '.' || c == '/')
            cpu_brand[i] = '_';
    }

    /* Sanitize model name too */
    std::string safe_model = model_name;
    for (auto& c : safe_model) {
        if (c == '/' || c == '\\' || c == ':' || c == ' ')
            c = '_';
    }

    /* ORT version (major.minor only) */
    const OrtApiBase* ort_base = OrtGetApiBase();
    const char* ort_ver = ort_base->GetVersionString();
    /* Extract major.minor from version string like "1.29.0" */
    int ort_major = 0, ort_minor = 0;
    if (ort_ver) {
        sscanf(ort_ver, "%d.%d", &ort_major, &ort_minor);
    }

    std::ostringstream key;
    key << logical_cores << "x" << physical_cores << "_" << cpu_brand << "_" << safe_model
        << "_ort" << ort_major << "." << ort_minor
        << "_v" << LIBEMBEDDING_VERSION_MAJOR << "." << LIBEMBEDDING_VERSION_MINOR;
    return key.str();
}

static std::string get_cache_path(const char* model_name) {
    std::string dir = autotune_cache_dir();
    std::filesystem::create_directories(dir);
    return dir + "/" + get_cache_key(model_name) + ".json";
}

static bool read_cache(const char* model_name, lembed_tuning_result_t& result) {
    std::string path = get_cache_path(model_name);
    std::ifstream f(path);
    if (!f.is_open()) return false;

    /* Simple JSON parsing */
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);

        /* Trim */
        auto trim = [](std::string& s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '"' || s.front() == ','))
                s.erase(s.begin());
            while (!s.empty() && (s.back() == ' ' || s.back() == '"' || s.back() == ','))
                s.pop_back();
        };
        trim(key);
        trim(val);

        if (key == "workers") result.workers = std::stoi(val);
        else if (key == "threads") result.threads = std::stoi(val);
        else if (key == "batch_size") result.batch_size = std::stoi(val);
        else if (key == "throughput_docs_sec") result.throughput_docs_sec = std::stod(val);
        else if (key == "latency_ms") result.latency_ms = std::stod(val);
        else if (key == "memory_mb") result.memory_mb = std::stod(val);
    }

    f.close();
    return true;
}

static void write_cache(const char* model_name, const lembed_tuning_result_t& result) {
    std::string path = get_cache_path(model_name);
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "{\n";
    f << "  \"workers\": " << result.workers << ",\n";
    f << "  \"threads\": " << result.threads << ",\n";
    f << "  \"batch_size\": " << result.batch_size << ",\n";
    f << "  \"throughput_docs_sec\": " << result.throughput_docs_sec << ",\n";
    f << "  \"latency_ms\": " << result.latency_ms << ",\n";
    f << "  \"memory_mb\": " << result.memory_mb << "\n";
    f << "}\n";

    f.close();
}

static void clear_cache(const char* model_name) {
    if (model_name) {
        std::string path = get_cache_path(model_name);
        try { std::filesystem::remove(path); } catch (...) {}
    } else {
        /* Clear all cache */
        try {
    std::string dir = autotune_cache_dir();
            if (std::filesystem::exists(dir)) {
                std::filesystem::remove_all(dir);
            }
        } catch (...) {}
    }
}

/* Generate synthetic benchmark corpus with varied lengths */
static std::vector<std::string> generate_corpus(int n_samples) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "algorithms", "process", "data", "efficiently",
        "natural", "language", "processing", "enables", "understanding", "semantic",
        "embeddings", "represent", "meaningful", "vector", "representations",
        "transformer", "models", "utilize", "attention", "mechanisms",
        "deep", "neural", "networks", "learn", "patterns", "from", "training",
        "optimization", "techniques", "improve", "convergence", "accuracy",
        "inference", "latency", "throughput", "benchmark", "performance",
        "production", "deployment", "scalable", "reliable", "systems"
    };
    int nwords = sizeof(words) / sizeof(words[0]);

    /* Distribution: 40% short, 40% medium, 20% long */
    int lengths[] = {16, 16, 16, 16, 64, 64, 64, 64, 128, 128};
    int n_lengths = sizeof(lengths) / sizeof(lengths[0]);

    std::vector<std::string> corpus;
    corpus.reserve(n_samples);

    for (int i = 0; i < n_samples; i++) {
        int ntok = lengths[i % n_lengths];
        std::string text;
        for (int j = 0; j < ntok; j++) {
            if (j > 0) text += " ";
            text += words[(i * 7 + j) % nwords];
        }
        corpus.push_back(text);
    }

    return corpus;
}

/* Benchmark a single configuration */
static double benchmark_config(
        const std::vector<std::string>& corpus,
        int workers, int threads, int batch_size,
        double& out_latency_ms) {

    if (corpus.empty()) return 0.0;

    /* Create workers */
    std::vector<lembed_text_embedding_t*> emb(workers);
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.offline = 1;
    opts.show_download_progress = 0;

    for (int i = 0; i < workers; i++) {
        if (lembed_text_embedding_create(&opts, &emb[i]) != LEMBED_OK) {
            for (int j = 0; j < i; j++) lembed_text_embedding_free(emb[j]);
            return 0.0;
        }
    }

    int n = (int)corpus.size();
    int per = n / workers;

    /* Warmup */
    for (int w = 0; w < 3; w++) {
        for (int i = 0; i < workers; i++) {
            std::vector<const char*> ct;
            int start = i * per;
            int cnt = (i == workers - 1) ? n - start : per;
            for (int k = 0; k < cnt; k++)
                ct.push_back(corpus[start + k].c_str());
            lembed_embeddings_t r = {0};
            lembed_text_embedding_embed(emb[i], ct.data(), cnt, cnt, &r);
            lembed_embeddings_free(&r);
        }
    }

    /* Measure */
    auto t0 = std::chrono::high_resolution_clock::now();
    int n_batches = 3;  /* reduced for speed */

    for (int b = 0; b < n_batches; b++) {
        std::vector<std::thread> th;
        for (int i = 0; i < workers; i++) {
            th.emplace_back([&emb, i, &corpus, per, n, workers]() {
                std::vector<const char*> ct;
                int start = i * per;
                int cnt = (i == workers - 1) ? n - start : per;
                for (int k = 0; k < cnt; k++)
                    ct.push_back(corpus[start + k].c_str());
                lembed_embeddings_t r = {0};
                lembed_text_embedding_embed(emb[i], ct.data(), cnt, cnt, &r);
                lembed_embeddings_free(&r);
            });
        }
        for (auto& t : th) t.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double total_texts = n_batches * corpus.size();
    double docs_per_sec = total_texts / (total_ms / 1000.0);
    out_latency_ms = total_ms / total_texts;

    for (int i = 0; i < workers; i++)
        lembed_text_embedding_free(emb[i]);

    return docs_per_sec;
}

/* Scoring function: higher is better */
static double score_config(double throughput, double latency_ms, double memory_mb) {
    /* Penalty for high memory (>500MB) */
    double mem_penalty = 0.0;
    if (memory_mb > 500.0) {
        mem_penalty = (memory_mb - 500.0) / 100.0;
    }

    /* Penalty for high latency (>20ms per text) */
    double lat_penalty = 0.0;
    if (latency_ms > 20.0) {
        lat_penalty = (latency_ms - 20.0) / 5.0;
    }

    return throughput - mem_penalty - lat_penalty;
}

/* Main autotune implementation */
static lembed_status_t autotune_impl(
        lembed_text_model_t model,
        const std::vector<std::string>& corpus,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {

    /* Get model code for cache key */
    lembed_model_info_t info;
    if (lembed_get_text_model_info(model, &info) != LEMBED_OK) {
        return LEMBED_ERROR_MODEL_NOT_FOUND;
    }
    const char* model_code = info.model_code;

    /* Check cache first */
    lembed_tuning_result_t cached;
    if (read_cache(model_code, cached)) {
        fprintf(stderr, "autotune: cache hit for %s (workers=%d, threads=%d, batch=%d)\n",
                model_code, cached.workers, cached.threads, cached.batch_size);
        if (result) *result = cached;
        return LEMBED_OK;
    }

    fprintf(stderr, "autotune: cache miss for %s (key=%s), running benchmark...\n",
            model_code, get_cache_path(model_code).c_str());

    int cores = cpu_logical_cores();
    int n_samples = (mode == LEMBED_AUTOTUNE_QUICK) ? 16 : 64;

    /* Generate corpus if not provided */
    std::vector<std::string> bench_corpus;
    if (corpus.empty()) {
        bench_corpus = generate_corpus(n_samples);
    } else {
        bench_corpus = corpus;
    }

    /* Configurations to test */
    struct Config { int workers; int threads; int batch; };

    std::vector<Config> configs;

    if (mode == LEMBED_AUTOTUNE_QUICK) {
        /* Quick: test key configurations */
        int worker_opts[] = {1, std::min(4, cores), std::min(8, cores)};
        for (int w : worker_opts) {
            if (w > cores) continue;
            configs.push_back({w, 1, 32});
        }
    } else {
        /* Full: exhaustive search */
        int worker_opts[] = {1, 2, 4, 8, 16};
        int thread_opts[] = {1, 2, 4};
        int batch_opts[] = {16, 32, 64, 128, 256};

        for (int w : worker_opts) {
            if (w > cores) continue;
            for (int t : thread_opts) {
                if (w * t > cores) continue;
                for (int b : batch_opts) {
                    configs.push_back({w, t, b});
                }
            }
        }
    }

    /* Benchmark each config */
    double best_score = -1e18;  /* very negative initial score */
    lembed_tuning_result_t best = {1, 1, 64, 0, 0, 0};

    int configs_tested = 0;
    for (const auto& cfg : configs) {
        double latency = 0.0;
        double throughput = benchmark_config(bench_corpus, cfg.workers, cfg.threads, cfg.batch, latency);

        /* Estimate memory: ~100MB per worker */
        double memory = cfg.workers * 100.0;

        double score = score_config(throughput, latency, memory);

        fprintf(stderr, "  autotune: workers=%d threads=%d batch=%d -> %.1f docs/s\n",
                cfg.workers, cfg.threads, cfg.batch, throughput);

        if (score > best_score) {
            best_score = score;
            best.workers = cfg.workers;
            best.threads = cfg.threads;
            best.batch_size = cfg.batch;
            best.throughput_docs_sec = throughput;
            best.latency_ms = latency;
            best.memory_mb = memory;
        }
    }

    if (result) {
        *result = best;
    }

    /* Write to cache */
    write_cache(model_code, best);
    fprintf(stderr, "autotune: cached result for %s at %s\n", model_code, get_cache_path(model_code).c_str());

    return LEMBED_OK;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP */
