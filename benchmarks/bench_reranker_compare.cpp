/*
 * libembedding - bench_reranker_compare.cpp
 * Benchmark: ONNX reranker vs llama.cpp reranker
 *
 * Measures:
 *   - Latency (ms) per query for Top-K ∈ {20, 50, 100}
 *   - Throughput (queries/sec)
 *   - Memory footprint (MB)
 *   - Quality metrics (nDCG@10, MRR) if ground truth available
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

#ifndef LEMBED_RERANKER_MODEL_DEFAULT
#define LEMBED_RERANKER_MODEL_DEFAULT 0
#endif

/* =========================================================================
 * Synthetic benchmark data (MS MARCO-like passage reranking)
 * ========================================================================= */
static const char* QUERIES[] = {
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Explain the theory of relativity",
    "What are the benefits of exercise?",
    "How do vaccines work?",
    "What is quantum computing?",
    "Describe the water cycle",
    "What causes climate change?",
    "How does the internet work?",
    "What is DNA replication?",
    "Explain neural networks",
    "What is the speed of light?",
    "How do airplanes fly?",
    "What is the human genome?",
    "Describe plate tectonics",
    "What is artificial intelligence?",
    "How does GPS work?",
    "What are black holes?",
    "Explain the immune system",
};

static const char* PASSAGES[] = {
    "The capital of France is Paris.",
    "Paris is the largest city in France.",
    "France is a country in Western Europe.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Chlorophyll absorbs light during photosynthesis.",
    "Plants use photosynthesis to produce oxygen.",
    "Machine learning is a subset of artificial intelligence.",
    "Algorithms learn patterns from data in machine learning.",
    "Deep learning uses neural networks for machine learning.",
    "Einstein developed the theory of relativity.",
    "Relativity describes gravity as curvature of spacetime.",
    "Special relativity introduced E=mc^2.",
    "Exercise improves cardiovascular health.",
    "Regular exercise reduces stress and anxiety.",
    "Physical activity strengthens muscles and bones.",
    "Vaccines stimulate the immune system to recognize pathogens.",
    "mRNA vaccines teach cells to produce antigens.",
    "Vaccination has eradicated smallpox.",
    "Quantum computing uses qubits instead of classical bits.",
    "Qubits can exist in superposition states.",
    "Quantum algorithms can solve certain problems exponentially faster.",
    "The water cycle includes evaporation, condensation, and precipitation.",
    "Water vapor rises and cools in the atmosphere.",
    "Rain and snow return water to Earth's surface.",
    "Burning fossil fuels releases greenhouse gases.",
    "CO2 traps heat in the atmosphere.",
    "Deforestation contributes to climate change.",
    "The internet connects billions of devices worldwide.",
    "TCP/IP protocols enable internet communication.",
    "Data travels through routers and switches on the internet.",
    "DNA replication copies genetic material before cell division.",
    "Helicase unwinds the DNA double helix.",
    "DNA polymerase synthesizes new DNA strands.",
    "Neural networks are inspired by biological neurons.",
    "Backpropagation trains neural networks.",
    "Convolutional networks excel at image recognition.",
    "Light travels at approximately 299,792 kilometers per second.",
    "Nothing can travel faster than light in vacuum.",
    "The speed of light is a fundamental constant.",
    "Airplanes generate lift using wing shape and angle of attack.",
    "Bernoulli's principle explains aerodynamic lift.",
    "Jet engines provide thrust for aircraft.",
    "The human genome contains about 3 billion base pairs.",
    "DNA encodes the instructions for life.",
    "The Human Genome Project mapped all human genes.",
    "Plate tectonics explains the movement of Earth's crust.",
    "Continental drift was proposed by Alfred Wegener.",
    "Earthquakes occur at plate boundaries.",
    "AI enables machines to perform tasks requiring intelligence.",
    "Machine learning is a subset of AI.",
    "AI applications include robotics and natural language processing.",
    "GPS satellites orbit Earth and transmit timing signals.",
    "Receivers calculate position using trilateration.",
    "GPS accuracy depends on satellite geometry.",
    "Black holes have extreme gravitational pull.",
    "Event horizons mark the point of no return around black holes.",
    "Hawking radiation is theoretical emission from black holes.",
    "The immune system defends against pathogens.",
    "Antibodies recognize and neutralize antigens.",
    "T cells and B cells are key immune cells.",
};

static const int N_QUERIES = 20;
static const int N_PASSAGES_PER_QUERY = 3;
static const int TOTAL_PASSAGES = 60;

/* =========================================================================
 * Relevance judgments (binary relevance for nDCG@10 computation)
 * For each query, we mark the first N_PASSAGES_PER_QUERY as relevant (1)
 * and the rest as non-relevant (0). This simulates MS MARCO format where
 * passages are annotated with binary relevance.
 * ========================================================================= */
static std::vector<std::vector<int>> generate_relevance_judgments() {
    std::vector<std::vector<int>> relevance(N_QUERIES, std::vector<int>(TOTAL_PASSAGES, 0));
    for (int q = 0; q < N_QUERIES; q++) {
        /* Mark the passages for this query as relevant */
        int start = q * N_PASSAGES_PER_QUERY;
        for (int i = 0; i < N_PASSAGES_PER_QUERY && (start + i) < TOTAL_PASSAGES; i++) {
            relevance[q][start + i] = 1;
        }
    }
    return relevance;
}

/* =========================================================================
 * Benchmark result structure
 * ========================================================================= */
struct BenchmarkResult {
    std::string backend;
    int top_k;
    double avg_latency_ms;
    double p50_latency_ms;
    double p95_latency_ms;
    double p99_latency_ms;
    double throughput_qps;
    size_t memory_mb;
    double ndcg_at_10;
    double mrr;
    int num_queries;
    int num_documents_per_query;
};

/* =========================================================================
 * Timing helpers
 * ========================================================================= */
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static std::vector<double> measure_latencies(
        lembed_reranker_t* reranker,
        const std::vector<std::string>& queries,
        const std::vector<std::string>& documents,
        int top_k,
        int num_iterations) {

    std::vector<double> latencies;
    latencies.reserve(num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
        for (const auto& query : queries) {
            auto t_start = now_ms();
            lembed_rerank_results_t results = {0};
            std::vector<const char*> doc_ptrs;
            doc_ptrs.reserve(documents.size());
            for (const auto& d : documents) doc_ptrs.push_back(d.c_str());

            lembed_status_t s = lembed_reranker_rerank(
                reranker, query.c_str(), doc_ptrs.data(),
                (int)doc_ptrs.size(), 32, &results);

            double t_end = now_ms();
            if (s == LEMBED_OK && results.items) {
                latencies.push_back(t_end - t_start);
                lembed_rerank_results_free(&results);
            }
        }
    }
    return latencies;
}

/* =========================================================================
 * Compute nDCG@10 from relevance judgments and reranked order
 * ========================================================================= */
static double compute_ndcg_at_10(const std::vector<int>& relevance,
                                  const lembed_rerank_result_t* results,
                                  int result_count) {
    if (result_count == 0) return 0.0;

    /* DCG@10 */
    double dcg = 0.0;
    int limit = std::min(10, result_count);
    for (int i = 0; i < limit; i++) {
        int idx = results[i].index;
        if (idx >= 0 && idx < (int)relevance.size()) {
            double rel = relevance[idx];
            dcg += (rel > 0) ? (1.0 / std::log2(i + 2)) : 0.0;
        }
    }

    /* IDCG@10 (ideal ordering) */
    std::vector<double> ideal;
    for (int r : relevance) ideal.push_back(r);
    std::sort(ideal.begin(), ideal.end(), std::greater<double>());
    double idcg = 0.0;
    for (int i = 0; i < limit; i++) {
        idcg += (ideal[i] > 0) ? (1.0 / std::log2(i + 2)) : 0.0;
    }

    return (idcg > 0.0) ? (dcg / idcg) : 0.0;
}

/* =========================================================================
 * Compute MRR (Mean Reciprocal Rank) from relevance judgments
 * ========================================================================= */
static double compute_mrr(const std::vector<int>& relevance,
                          const lembed_rerank_result_t* results,
                          int result_count) {
    if (result_count == 0) return 0.0;
    for (int i = 0; i < result_count; i++) {
        int idx = results[i].index;
        if (idx >= 0 && idx < (int)relevance.size() && relevance[idx] > 0) {
            return 1.0 / (i + 1);
        }
    }
    return 0.0;
}

/* =========================================================================
 * Run quality benchmark with ground truth
 * ========================================================================= */
static std::pair<double, double> measure_quality(
        lembed_reranker_t* reranker,
        const std::vector<std::string>& queries,
        const std::vector<std::string>& documents,
        const std::vector<std::vector<int>>& relevance,
        int top_k) {

    double total_ndcg = 0.0;
    double total_mrr = 0.0;
    int num_queries = (int)queries.size();

    for (int q = 0; q < num_queries; q++) {
        lembed_rerank_results_t results = {0};
        std::vector<const char*> doc_ptrs;
        doc_ptrs.reserve(documents.size());
        for (const auto& d : documents) doc_ptrs.push_back(d.c_str());

        lembed_status_t s = lembed_reranker_rerank(
            reranker, queries[q].c_str(), doc_ptrs.data(),
            (int)doc_ptrs.size(), 32, &results);

        if (s == LEMBED_OK && results.items) {
            total_ndcg += compute_ndcg_at_10(relevance[q], results.items, results.count);
            total_mrr += compute_mrr(relevance[q], results.items, results.count);
            lembed_rerank_results_free(&results);
        }
    }

    return {total_ndcg / num_queries, total_mrr / num_queries};
}

/* =========================================================================
 * Extract latency percentiles
 * ========================================================================= */
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

/* =========================================================================
 * Get memory usage (Windows-specific)
 * ========================================================================= */
static size_t get_process_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (size_t)(pmc.WorkingSetSize / (1024 * 1024));
    }
#endif
    return 0;
}

/* =========================================================================
 * Main benchmark
 * ========================================================================= */
int main(int argc, char* argv[]) {
    printf("=============================================================\n");
    printf("libembedding Reranker Benchmark: ONNX vs llama.cpp\n");
    printf("=============================================================\n\n");

    /* Parse arguments */
    const char* onnx_model = "BAAI/bge-reranker-base";
    const char* gguf_model = nullptr;
    int num_iterations = 3;
    int warmup_iterations = 1;
    std::vector<int> top_k_values = {20, 50, 100};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--onnx") == 0 && i + 1 < argc) {
            onnx_model = argv[++i];
        } else if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_model = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            num_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: bench_reranker_compare [options]\n");
            printf("Options:\n");
            printf("  --onnx MODEL      ONNX model name or path (default: BAAI/bge-reranker-base)\n");
            printf("  --gguf PATH       GGUF model path for llama.cpp backend\n");
            printf("  --iterations N    Number of benchmark iterations (default: 3)\n");
            printf("  --top-k K,K,K     Top-K values to test (default: 20,50,100)\n");
            return 0;
        }
    }

    /* Generate benchmark data */
    printf("Generating benchmark data...\n");
    std::vector<std::string> queries;
    for (int i = 0; i < N_QUERIES; i++) {
        queries.push_back(QUERIES[i]);
    }

    std::vector<std::string> documents;
    for (int i = 0; i < TOTAL_PASSAGES; i++) {
        documents.push_back(PASSAGES[i]);
    }

    auto relevance = generate_relevance_judgments();
    printf("  Queries: %d\n", (int)queries.size());
    printf("  Documents per query: %d\n", TOTAL_PASSAGES);
    printf("  Relevance judgments generated\n\n");

    std::vector<BenchmarkResult> results;

    /* =========================================================
     * ONNX Backend
     * ========================================================= */
    printf("-------------------------------------------------------------\n");
    printf("Backend 1: ONNX\n");
    printf("-------------------------------------------------------------\n");

    lembed_reranker_t* onnx_reranker = nullptr;
    {
        lembed_reranker_options_t opts = lembed_reranker_options_default();
        opts.model = LEMBED_RERANKER_MODEL_DEFAULT;
        opts.batch_size = 32;
        opts.num_threads = 4;

        printf("Loading ONNX model: %s...\n", onnx_model);

        /* Try to resolve model name to registry index */
        int idx = lembed_find_reranker_model_by_code(onnx_model);
        if (idx >= 0) {
            opts.model = (lembed_reranker_model_t)idx;
        }

        lembed_status_t s = lembed_reranker_create(&opts, &onnx_reranker);
        if (s != LEMBED_OK) {
            printf("WARNING: Failed to load ONNX model: %s\n", lembed_last_error());
            printf("Skipping ONNX backend.\n\n");
        } else {
            printf("ONNX model loaded successfully.\n\n");
        }
    }

    if (onnx_reranker) {
        size_t mem_before = get_process_memory_mb();

        for (int top_k : top_k_values) {
            printf("  Testing Top-K = %d...\n", top_k);

            /* Warmup */
            for (int w = 0; w < warmup_iterations; w++) {
                for (const auto& query : queries) {
                    lembed_rerank_results_t r = {0};
                    std::vector<const char*> doc_ptrs;
                    for (const auto& d : documents) doc_ptrs.push_back(d.c_str());
                    lembed_reranker_rerank(onnx_reranker, query.c_str(), doc_ptrs.data(),
                                           (int)doc_ptrs.size(), 32, &r);
                    if (r.items) lembed_rerank_results_free(&r);
                }
            }

            /* Benchmark */
            auto latencies = measure_latencies(onnx_reranker, queries, documents, top_k, num_iterations);
            auto quality = measure_quality(onnx_reranker, queries, documents, relevance, top_k);

            size_t mem_after = get_process_memory_mb();
            size_t mem_used = (mem_after > mem_before) ? (mem_after - mem_before) : mem_after;

            double total_time = 0.0;
            for (double l : latencies) total_time += l;

            BenchmarkResult res;
            res.backend = "ONNX";
            res.top_k = top_k;
            res.avg_latency_ms = total_time / latencies.size();
            res.p50_latency_ms = percentile(latencies, 0.50);
            res.p95_latency_ms = percentile(latencies, 0.95);
            res.p99_latency_ms = percentile(latencies, 0.99);
            res.throughput_qps = latencies.size() > 0 ? (1000.0 / res.avg_latency_ms) : 0.0;
            res.memory_mb = mem_used;
            res.ndcg_at_10 = quality.first;
            res.mrr = quality.second;
            res.num_queries = (int)queries.size();
            res.num_documents_per_query = TOTAL_PASSAGES;
            results.push_back(res);

            printf("    Avg latency: %.2f ms\n", res.avg_latency_ms);
            printf("    P50 latency: %.2f ms\n", res.p50_latency_ms);
            printf("    P95 latency: %.2f ms\n", res.p95_latency_ms);
            printf("    Throughput: %.1f QPS\n", res.throughput_qps);
            printf("    nDCG@10: %.4f\n", res.ndcg_at_10);
            printf("    MRR: %.4f\n", res.mrr);
            printf("    Memory: %zu MB\n\n", res.memory_mb);
        }

        lembed_reranker_free(onnx_reranker);
    }

    /* =========================================================
     * llama.cpp Backend
     * ========================================================= */
    if (gguf_model) {
        printf("-------------------------------------------------------------\n");
        printf("Backend 2: llama.cpp (GGUF)\n");
        printf("-------------------------------------------------------------\n");

        lembed_reranker_t* llama_reranker = nullptr;
        {
            lembed_reranker_options_t opts = lembed_reranker_options_default();
            opts.batch_size = 32;
            opts.num_threads = 4;

            printf("Loading GGUF model: %s...\n", gguf_model);

            lembed_status_t s = lembed_reranker_create_from_gguf_path(gguf_model, &opts, &llama_reranker);
            if (s != LEMBED_OK) {
                printf("WARNING: Failed to load GGUF model: %s\n", lembed_last_error());
                printf("Skipping llama.cpp backend.\n\n");
            } else {
                printf("GGUF model loaded successfully.\n\n");
            }
        }

        if (llama_reranker) {
            size_t mem_before = get_process_memory_mb();

            for (int top_k : top_k_values) {
                printf("  Testing Top-K = %d...\n", top_k);

                /* Warmup */
                for (int w = 0; w < warmup_iterations; w++) {
                    for (const auto& query : queries) {
                        lembed_rerank_results_t r = {0};
                        std::vector<const char*> doc_ptrs;
                        for (const auto& d : documents) doc_ptrs.push_back(d.c_str());
                        lembed_reranker_rerank(llama_reranker, query.c_str(), doc_ptrs.data(),
                                               (int)doc_ptrs.size(), 32, &r);
                        if (r.items) lembed_rerank_results_free(&r);
                    }
                }

                auto latencies = measure_latencies(llama_reranker, queries, documents, top_k, num_iterations);
                auto quality = measure_quality(llama_reranker, queries, documents, relevance, top_k);

                size_t mem_after = get_process_memory_mb();
                size_t mem_used = (mem_after > mem_before) ? (mem_after - mem_before) : mem_after;

                double total_time = 0.0;
                for (double l : latencies) total_time += l;

                BenchmarkResult res;
                res.backend = "llama.cpp";
                res.top_k = top_k;
                res.avg_latency_ms = total_time / latencies.size();
                res.p50_latency_ms = percentile(latencies, 0.50);
                res.p95_latency_ms = percentile(latencies, 0.95);
                res.p99_latency_ms = percentile(latencies, 0.99);
                res.throughput_qps = latencies.size() > 0 ? (1000.0 / res.avg_latency_ms) : 0.0;
                res.memory_mb = mem_used;
                res.ndcg_at_10 = quality.first;
                res.mrr = quality.second;
                res.num_queries = (int)queries.size();
                res.num_documents_per_query = TOTAL_PASSAGES;
                results.push_back(res);

                printf("    Avg latency: %.2f ms\n", res.avg_latency_ms);
                printf("    P50 latency: %.2f ms\n", res.p50_latency_ms);
                printf("    P95 latency: %.2f ms\n", res.p95_latency_ms);
                printf("    Throughput: %.1f QPS\n", res.throughput_qps);
                printf("    nDCG@10: %.4f\n", res.ndcg_at_10);
                printf("    MRR: %.4f\n", res.mrr);
                printf("    Memory: %zu MB\n\n", res.memory_mb);
            }

            lembed_reranker_free(llama_reranker);
        }
    } else {
        printf("-------------------------------------------------------------\n");
        printf("Backend 2: llama.cpp (skipped - no GGUF path provided)\n");
        printf("-------------------------------------------------------------\n");
        printf("Usage: bench_reranker_compare --gguf path/to/model.gguf\n\n");
    }

    /* =========================================================
     * Summary
     * ========================================================= */
    printf("=============================================================\n");
    printf("BENCHMARK SUMMARY\n");
    printf("=============================================================\n\n");

    printf("%-12s %6s %10s %10s %10s %10s %10s %8s %8s\n",
           "Backend", "Top-K", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)", "QPS", "Mem(MB)", "nDCG@10");
    printf("%-12s %6s %10s %10s %10s %10s %10s %8s %8s\n",
           "-------", "-----", "--------", "--------", "--------", "--------", "-----", "-------", "-------");

    for (const auto& r : results) {
        printf("%-12s %6d %10.2f %10.2f %10.2f %10.2f %10.1f %8zu %8.4f\n",
               r.backend.c_str(), r.top_k, r.avg_latency_ms, r.p50_latency_ms,
               r.p95_latency_ms, r.p99_latency_ms, r.throughput_qps,
               r.memory_mb, r.ndcg_at_10);
    }

    printf("\n=============================================================\n");
    printf("Quality Comparison\n");
    printf("=============================================================\n\n");

    for (const auto& r : results) {
        printf("%s (Top-K=%d): nDCG@10=%.4f, MRR=%.4f\n",
               r.backend.c_str(), r.top_k, r.ndcg_at_10, r.mrr);
    }

    printf("\n=============================================================\n");
    printf("Recommendations\n");
    printf("=============================================================\n\n");

    if (results.size() >= 2) {
        /* Compare first ONNX vs first llama.cpp result */
        const auto& onnx_res = results[0];
        const auto& llama_res = results[1];

        printf("Latency: %s is %.2fx %s than %s\n",
               onnx_res.avg_latency_ms < llama_res.avg_latency_ms ? "ONNX" : "llama.cpp",
               std::max(onnx_res.avg_latency_ms, llama_res.avg_latency_ms) /
               std::min(onnx_res.avg_latency_ms, llama_res.avg_latency_ms),
               onnx_res.avg_latency_ms < llama_res.avg_latency_ms ? "faster" : "slower",
               onnx_res.avg_latency_ms < llama_res.avg_latency_ms ? "llama.cpp" : "ONNX");

        printf("Memory: %s uses %.2fx %s memory than %s\n",
               onnx_res.memory_mb < llama_res.memory_mb ? "ONNX" : "llama.cpp",
               (float)std::max(onnx_res.memory_mb, llama_res.memory_mb) /
               (float)std::min(onnx_res.memory_mb, llama_res.memory_mb),
               onnx_res.memory_mb < llama_res.memory_mb ? "less" : "more",
               onnx_res.memory_mb < llama_res.memory_mb ? "llama.cpp" : "ONNX");

        printf("Quality: nDCG@10 gap = %.4f (%s)\n",
               std::abs(onnx_res.ndcg_at_10 - llama_res.ndcg_at_10),
               onnx_res.ndcg_at_10 >= llama_res.ndcg_at_10 ? "ONNX >= llama.cpp" : "llama.cpp > ONNX");
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
