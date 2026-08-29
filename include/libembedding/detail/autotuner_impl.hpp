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
#include "libembedding/sparse_text_embedding.h"

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
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
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
        fclose(fp);
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

/* =========================================================================
 * Reranker Auto-Tuner Implementation
 * ========================================================================= */

/* Generate synthetic documents with target token count */
static std::string generate_synthetic_doc(int target_tokens, int seed) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "data", "model", "system", "algorithm",
        "search", "query", "document", "text", "information", "result",
        "process", "analysis", "method", "approach", "technique",
        "application", "performance", "evaluation", "research", "study",
    };
    int n_words = sizeof(words) / sizeof(words[0]);

    std::string doc;
    int target_words = std::max(1, (int)(target_tokens * 0.75));
    for (int i = 0; i < target_words; i++) {
        if (i > 0) doc += " ";
        doc += words[(seed + i) % n_words];
    }
    return doc;
}

/* Get cache directory for reranker autotune */
static std::string reranker_autotune_cache_dir() {
    return autotune_cache_dir() + "/reranker";
}

/* Get cache file path for a reranker model */
static std::string get_reranker_cache_path(const char* model_name) {
    std::string dir = reranker_autotune_cache_dir();
    std::filesystem::create_directories(dir);

    /* Build cache key: model_name + cpu_cores */
    int cores = cpu_logical_cores();
    std::string key = std::string(model_name) + "_cores_" + std::to_string(cores);
    return dir + "/" + get_cache_key(key.c_str()) + ".json";
}

/* Write reranker tune result to cache */
static void write_reranker_cache(const char* model_name, const lembed_reranker_tuning_result_t& result) {
    std::string path = get_reranker_cache_path(model_name);
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "{\n";
    f << "  \"threads\": " << result.threads << ",\n";
    f << "  \"batch_size\": " << result.batch_size << ",\n";
    f << "  \"max_tokens\": " << result.max_tokens << ",\n";
    f << "  \"throughput_docs_sec\": " << result.throughput_docs_sec << ",\n";
    f << "  \"latency_ms\": " << result.latency_ms << ",\n";
    f << "  \"p95_latency_ms\": " << result.p95_latency_ms << ",\n";
    f << "  \"memory_mb\": " << result.memory_mb << "\n";
    f << "}\n";
}

/* Read reranker tune result from cache */
static bool read_reranker_cache(const char* model_name, lembed_reranker_tuning_result_t* out) {
    std::string path = get_reranker_cache_path(model_name);
    std::ifstream f(path);
    if (!f.is_open()) return false;

    /* Simple JSON parsing */
    std::string line;
    while (std::getline(f, line)) {
        auto find_val = [](const std::string& s, const char* key) -> double {
            std::string search = std::string("\"") + key + "\": ";
            size_t pos = s.find(search);
            if (pos == std::string::npos) return -1;
            return std::stod(s.substr(pos + search.length()));
        };
        if (line.find("\"threads\"") != std::string::npos) out->threads = (int)find_val(line, "threads");
        if (line.find("\"batch_size\"") != std::string::npos) out->batch_size = (int)find_val(line, "batch_size");
        if (line.find("\"max_tokens\"") != std::string::npos) out->max_tokens = (int)find_val(line, "max_tokens");
        if (line.find("\"throughput_docs_sec\"") != std::string::npos) out->throughput_docs_sec = find_val(line, "throughput_docs_sec");
        if (line.find("\"latency_ms\"") != std::string::npos) out->latency_ms = find_val(line, "latency_ms");
        if (line.find("\"p95_latency_ms\"") != std::string::npos) out->p95_latency_ms = find_val(line, "p95_latency_ms");
        if (line.find("\"memory_mb\"") != std::string::npos) out->memory_mb = find_val(line, "memory_mb");
    }
    return true;
}

/* Benchmark a single reranker configuration */
static lembed_reranker_tuning_result_t bench_reranker_config(
    const char* model_name,
    int threads,
    int batch_size,
    int max_tokens,
    int n_docs,
    int warmup_iters,
    int bench_iters)
{
    lembed_reranker_tuning_result_t res = {0};
    res.threads = threads;
    res.batch_size = batch_size;
    res.max_tokens = max_tokens;

    /* Generate synthetic documents */
    std::vector<std::string> docs;
    for (int i = 0; i < n_docs; i++) {
        docs.push_back(generate_synthetic_doc(max_tokens, i * 7));
    }
    const char* query = "What is deep learning?";

    /* Create reranker */
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.max_length = max_tokens;
    opts.show_download_progress = 0;

    /* Resolve model by name or code */
    int model_idx = -1;
    {
        const lembed_model_info_t* models = nullptr;
        int count = 0;
        lembed_list_reranker_models(&models, &count);
        for (int i = 0; i < count; i++) {
            std::string name = models[i].model_name;
            std::string code = models[i].model_code;
            if (model_name == name || model_name == code) {
                model_idx = i;
                break;
            }
        }
    }
    if (model_idx < 0) model_idx = 0;
    opts.model = static_cast<lembed_reranker_model_t>(model_idx);

    lembed_reranker_t* ctx = nullptr;
    lembed_status_t s = lembed_reranker_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        res.latency_ms = 999999;
        return res;
    }

    /* Warmup */
    /* Build C string array for rerank API */
    std::vector<const char*> c_docs;
    for (const auto& d : docs) c_docs.push_back(d.c_str());

    for (int i = 0; i < warmup_iters; i++) {
        lembed_rerank_results_t result = {0};
        lembed_reranker_rerank(ctx, query, c_docs.data(), n_docs, batch_size, &result);
        lembed_rerank_results_free(&result);
    }

    /* Benchmark */
    std::vector<double> times;
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_rerank_results_t result = {0};
        lembed_reranker_rerank(ctx, query, c_docs.data(), n_docs, batch_size, &result);
        lembed_rerank_results_free(&result);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    lembed_reranker_free(ctx);

    /* Compute stats */
    std::sort(times.begin(), times.end());
    double p50 = times[times.size() / 2];
    double p95 = times[(int)(0.95 * times.size())];
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();

    res.latency_ms = p50;
    res.p95_latency_ms = p95;
    res.throughput_docs_sec = (p50 > 0) ? (1000.0 / p50) * n_docs : 0;
    res.memory_mb = 0;  /* TODO: measure RSS */

    return res;
}

/* Main reranker auto-tune function */
lembed_status_t lembed_reranker_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Check cache first */
    lembed_reranker_tuning_result_t cached;
    if (read_reranker_cache(model_name, &cached)) {
        fprintf(stderr, "reranker_autotune: using cached result for %s\n", model_name);
        *result = cached;
        return LEMBED_OK;
    }

    int cores = cpu_logical_cores();
    int n_docs = 20;
    int warmup = 1;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    /* Configurations to test */
    /* QUICK: fewer configs (3 threads x 2 batch x 2 tokens = 12) */
    /* FULL: more configs (4 threads x 3 batch x 4 tokens = 48) */
    std::vector<int> threads_vec, batch_vec, tokens_vec;

    if (mode == LEMBED_AUTOTUNE_QUICK) {
        threads_vec = {1, 4, 8};
        batch_vec = {4, 16};
        tokens_vec = {64, 256};
    } else {
        threads_vec = {1, 2, 4, 8};
        batch_vec = {4, 8, 16};
        tokens_vec = {32, 64, 128, 256};
    }

    /* Filter threads > cores */
    std::vector<int> valid_threads;
    for (int t : threads_vec) {
        if (t <= cores) valid_threads.push_back(t);
    }

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total_configs = valid_threads.size() * batch_vec.size() * tokens_vec.size();
    int current = 0;

    fprintf(stderr, "reranker_autotune: testing %d configurations (mode=%s)...\n",
            total_configs, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL");

    for (int t : valid_threads) {
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                current++;
                int threads = t;
                int batch = b;
                int tokens = k;

                fprintf(stderr, "  [%d/%d] threads=%d batch=%d tokens=%d\n",
                        current, total_configs, threads, batch, tokens);

                auto r = bench_reranker_config(model_name, threads, batch, tokens,
                                               n_docs, warmup, bench_iters);

                /* Score: prefer lower latency, with penalty for high P95 */
                double score = r.latency_ms + (r.p95_latency_ms - r.latency_ms) * 0.5;
                double best_score = best.latency_ms + (best.p95_latency_ms - best.latency_ms) * 0.5;

                if (score < best_score) {
                    best = r;
                }
            }
        }
    }

    fprintf(stderr, "reranker_autotune: best config: threads=%d batch=%d tokens=%d (P50=%.1fms, P95=%.1fms)\n",
            best.threads, best.batch_size, best.max_tokens, best.latency_ms, best.p95_latency_ms);

    /* Write to cache */
    write_reranker_cache(model_name, best);

    *result = best;
    return LEMBED_OK;
}

/* Reranker auto-tune with custom corpus */
lembed_status_t lembed_reranker_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_reranker_tuning_result_t* result)
{
    /* For now, delegate to standard autotune */
    return lembed::detail::lembed_reranker_autotune(model_name, mode, result);
}

/* Auto-configure reranker for target latency */
lembed_status_t lembed_reranker_auto_config(
    const char* model_name,
    double target_latency_ms,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Run quick autotune first */
    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int cores = cpu_logical_cores();
    int n_docs = 20;
    int warmup = 2;
    int bench_iters = 10;

    /* Use fewer configs for faster search */
    std::vector<int> threads_vec = {1, 4, 8};
    std::vector<int> batch_vec = {4, 16};
    std::vector<int> tokens_vec = {64, 256};

    for (int t : threads_vec) {
        if (t > cores) continue;
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                auto r = bench_reranker_config(model_name, t, b, k, n_docs, warmup, bench_iters);

                /* Must fit within latency budget */
                if (r.p95_latency_ms > target_latency_ms) continue;

                /* Among valid configs, prefer highest throughput */
                if (r.throughput_docs_sec > best.throughput_docs_sec) {
                    best = r;
                }
            }
        }
    }

    if (best.latency_ms >= 999999) {
        /* No config fits budget — try with more aggressive settings */
        tokens_vec = {32, 64};
        batch_vec = {4, 8};
        for (int t : threads_vec) {
            if (t > cores) continue;
            for (int b : batch_vec) {
                for (int k : tokens_vec) {
                    auto r = bench_reranker_config(model_name, t, b, k, n_docs, warmup, bench_iters);
                    if (r.p95_latency_ms > target_latency_ms) continue;
                    if (r.throughput_docs_sec > best.throughput_docs_sec) {
                        best = r;
                    }
                }
            }
        }
    }

    if (best.latency_ms >= 999999) {
        /* No config fits budget — return fastest */
        return lembed::detail::lembed_reranker_autotune(model_name, LEMBED_AUTOTUNE_QUICK, result);
    }

    *result = best;
    return LEMBED_OK;
}

/* Profile-based auto-config */
lembed_status_t lembed_reranker_auto_config_profile(
    const char* model_name,
    lembed_reranker_profile_t profile,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Map profiles to target latencies */
    double target_ms = 300;  /* default balanced */
    switch (profile) {
        case LEMBED_PROFILE_INTERACTIVE: target_ms = 100; break;
        case LEMBED_PROFILE_BALANCED:    target_ms = 300; break;
        case LEMBED_PROFILE_QUALITY:     target_ms = 1000; break;
    }

    return lembed::detail::lembed_reranker_auto_config(model_name, target_ms, result);
}

/* Clear reranker autotune cache */
void lembed_reranker_autotune_clear_cache(const char* model_name) {
    if (model_name) {
        std::string path = get_reranker_cache_path(model_name);
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
    } else {
        std::string dir = reranker_autotune_cache_dir();
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
        }
    }
}

}} /* namespace lembed::detail */

/* =========================================================================
 * Global scope wrappers for reranker autotune
 * ========================================================================= */

extern "C" {

lembed_status_t lembed_reranker_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_reranker_tuning_result_t* result)
{
    return lembed::detail::lembed_reranker_autotune(model_name, mode, result);
}

lembed_status_t lembed_reranker_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_reranker_tuning_result_t* result)
{
    return lembed::detail::lembed_reranker_autotune_custom(model_name, texts, n_texts, mode, result);
}

lembed_status_t lembed_reranker_auto_config(
    const char* model_name,
    double target_latency_ms,
    lembed_reranker_tuning_result_t* result)
{
    return lembed::detail::lembed_reranker_auto_config(model_name, target_latency_ms, result);
}

void lembed_reranker_autotune_clear_cache(const char* model_name)
{
    lembed::detail::lembed_reranker_autotune_clear_cache(model_name);
}

lembed_status_t lembed_reranker_auto_config_profile(
    const char* model_name,
    lembed_reranker_profile_t profile,
    lembed_reranker_tuning_result_t* result)
{
    return lembed::detail::lembed_reranker_auto_config_profile(model_name, profile, result);
}

} /* extern "C" */

/* =========================================================================
 * Sparse Auto-Tuner Implementation
 * ========================================================================= */

/* Benchmark a single sparse configuration */
static lembed_sparse_tuning_result_t bench_sparse_config(
    const char* model_name,
    int top_k,
    float min_weight,
    int storage_format,
    int threads,
    int batch_size,
    const std::vector<std::string>& texts,
    int warmup_iters,
    int bench_iters)
{
    lembed_sparse_tuning_result_t res = {0};
    res.top_k = top_k;
    res.min_weight = min_weight;
    res.storage_format = storage_format;
    res.threads = threads;
    res.batch_size = batch_size;

    /* Create sparse model */
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.top_k = top_k;
    opts.min_weight = min_weight;
    opts.storage_format = storage_format;
    opts.show_download_progress = 0;

    /* Resolve model */
    int model_idx = lembed_find_sparse_model_by_code(model_name);
    if (model_idx < 0) model_idx = 0;
    opts.model = static_cast<lembed_sparse_model_t>(model_idx);

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    lembed_status_t s = lembed_sparse_text_embedding_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        res.latency_ms = 999999;
        return res;
    }

    /* Prepare texts */
    const char** c_texts = new const char*[texts.size()];
    std::vector<std::string> encoded;
    for (size_t i = 0; i < texts.size(); i++) {
        encoded.push_back(texts[i]);
        c_texts[i] = encoded[i].c_str();
    }

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        lembed_sparse_embeddings_t result = {0};
        lembed_sparse_text_embedding_embed(ctx, c_texts, texts.size(), batch_size, &opts, &result);
        lembed_sparse_embeddings_free(&result);
    }

    /* Benchmark */
    std::vector<double> times;
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_sparse_embeddings_t result = {0};
        lembed_sparse_text_embedding_embed(ctx, c_texts, texts.size(), batch_size, &opts, &result);
        lembed_sparse_embeddings_free(&result);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    lembed_sparse_text_embedding_free(ctx);
    delete[] c_texts;

    /* Compute stats */
    std::sort(times.begin(), times.end());
    double p50 = times[times.size() / 2];

    res.latency_ms = p50;
    res.throughput_docs_sec = (p50 > 0) ? (1000.0 / p50) * texts.size() : 0;

    return res;
}

/* Main sparse auto-tune function */
lembed_status_t lembed_sparse_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_sparse_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    int threads = 4;
    int batch_size = 256;
    int n_docs = 20;
    int warmup = 2;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    /* Generate synthetic texts */
    std::vector<std::string> texts;
    for (int i = 0; i < n_docs; i++) {
        texts.push_back("Machine learning is a branch of artificial intelligence that enables systems to learn from data.");
    }

    /* Configurations to test */
    int top_k_options[] = {32, 64, 128};
    float min_weight_options[] = {0.0f, 0.01f, 0.05f};
    int storage_options[] = {0, 1}; /* dict, CSR */

    int n_top_k = sizeof(top_k_options) / sizeof(int);
    int n_weight = sizeof(min_weight_options) / sizeof(float);
    int n_storage = sizeof(storage_options) / sizeof(int);

    lembed_sparse_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total = n_top_k * n_weight * n_storage;
    int current = 0;

    fprintf(stderr, "sparse_autotune: testing %d configurations (mode=%s)...\n",
            total, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL");

    for (int t = 0; t < n_top_k; t++) {
        for (int w = 0; w < n_weight; w++) {
            for (int s = 0; s < n_storage; s++) {
                current++;
                fprintf(stderr, "  [%d/%d] top_k=%d min_weight=%.2f storage=%d\n",
                        current, total, top_k_options[t], min_weight_options[w], storage_options[s]);

                auto r = bench_sparse_config(
                    model_name, top_k_options[t], min_weight_options[w],
                    storage_options[s], threads, batch_size,
                    texts, warmup, bench_iters);

                if (r.latency_ms < best.latency_ms) {
                    best = r;
                }
            }
        }
    }

    fprintf(stderr, "sparse_autotune: best config: top_k=%d min_weight=%.2f storage=%d (P50=%.1fms)\n",
            best.top_k, best.min_weight, best.storage_format, best.latency_ms);

    *result = best;
    return LEMBED_OK;
}

/* =========================================================================
 * Unified Auto-Tuner Implementation
 * ========================================================================= */

extern "C" {

lembed_status_t lembed_autotune_unified(
    lembed_task_t task,
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_unified_tuning_result_t* result)
{
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(result, 0, sizeof(*result));
    result->task = task;

    switch (task) {
        case LEMBED_TASK_EMBEDDING: {
            lembed_tuning_result_t emb_result = {0};
            lembed_status_t s = lembed_autotune(model_name, mode, &emb_result);
            if (s != LEMBED_OK) return s;
            result->threads = emb_result.threads;
            result->batch_size = emb_result.batch_size;
            result->workers = emb_result.workers;
            result->throughput_docs_sec = emb_result.throughput_docs_sec;
            result->latency_ms = emb_result.latency_ms;
            result->memory_mb = emb_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_RERANKING: {
            lembed_reranker_tuning_result_t rerank_result = {0};
            lembed_status_t s = lembed::detail::lembed_reranker_autotune(model_name, mode, &rerank_result);
            if (s != LEMBED_OK) return s;
            result->threads = rerank_result.threads;
            result->batch_size = rerank_result.batch_size;
            result->max_tokens = rerank_result.max_tokens;
            result->throughput_docs_sec = rerank_result.throughput_docs_sec;
            result->latency_ms = rerank_result.latency_ms;
            result->p95_latency_ms = rerank_result.p95_latency_ms;
            result->memory_mb = rerank_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_IMAGE:
            return LEMBED_ERROR_UNSUPPORTED;
        case LEMBED_TASK_SPARSE: {
            lembed_sparse_tuning_result_t sparse_result = {0};
            lembed_status_t s = lembed_sparse_autotune(model_name, mode, &sparse_result);
            if (s != LEMBED_OK) return s;
            result->threads = sparse_result.threads;
            result->batch_size = sparse_result.batch_size;
            result->top_k = sparse_result.top_k;
            result->min_weight = sparse_result.min_weight;
            result->storage_format = sparse_result.storage_format;
            result->throughput_docs_sec = sparse_result.throughput_docs_sec;
            result->latency_ms = sparse_result.latency_ms;
            result->memory_mb = sparse_result.memory_mb;
            return LEMBED_OK;
        }
        default:
            return LEMBED_ERROR_INVALID_ARGUMENT;
    }
}

lembed_status_t lembed_autotune_unified_config(
    lembed_task_t task,
    const char* model_name,
    double target_latency_ms,
    lembed_unified_tuning_result_t* result)
{
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(result, 0, sizeof(*result));
    result->task = task;

    switch (task) {
        case LEMBED_TASK_RERANKING: {
            lembed_reranker_tuning_result_t rerank_result = {0};
            lembed_status_t s = lembed::detail::lembed_reranker_auto_config(
                model_name, target_latency_ms, &rerank_result);
            if (s != LEMBED_OK) return s;
            result->threads = rerank_result.threads;
            result->batch_size = rerank_result.batch_size;
            result->max_tokens = rerank_result.max_tokens;
            result->throughput_docs_sec = rerank_result.throughput_docs_sec;
            result->latency_ms = rerank_result.latency_ms;
            result->p95_latency_ms = rerank_result.p95_latency_ms;
            result->memory_mb = rerank_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_EMBEDDING:
            /* For embedding, use standard autotune (no latency budget concept yet) */
            return lembed_autotune_unified(task, model_name, LEMBED_AUTOTUNE_QUICK, result);
        case LEMBED_TASK_IMAGE:
        case LEMBED_TASK_SPARSE:
            return LEMBED_ERROR_UNSUPPORTED;
        default:
            return LEMBED_ERROR_INVALID_ARGUMENT;
    }
}

void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name) {
    switch (task) {
        case LEMBED_TASK_EMBEDDING:
            lembed_autotune_clear_cache(model_name);
            break;
        case LEMBED_TASK_RERANKING:
            lembed::detail::lembed_reranker_autotune_clear_cache(model_name);
            break;
        case LEMBED_TASK_IMAGE:
        case LEMBED_TASK_SPARSE:
            break;
    }
}

} /* extern "C" */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP */
