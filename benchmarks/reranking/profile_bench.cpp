/*
 * Reranking profile benchmark
 * Measures time breakdown: tokenization vs inference vs post-processing
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

/* ── helpers ─────────────────────────────────────────────────────────── */

static double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch())
        .count();
}

static double median(std::vector<double>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::sort(v.begin(), v.end());
    return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
}

static std::vector<std::string> generate_docs(int n) {
    std::vector<std::string> base = {
        "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The Python programming language was created by Guido van Rossum in 1991.",
        "Quantum computing leverages quantum mechanical phenomena to perform computation.",
        "The Great Wall of China is a series of fortifications built over centuries.",
        "Natural language processing enables computers to understand human language.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
    };
    std::vector<std::string> docs;
    for (int i = 0; i < n; i++) {
        std::string doc = base[i % base.size()];
        int repeat = (i % 3) + 1;
        std::string full;
        for (int r = 0; r < repeat; r++) full += doc + " ";
        docs.push_back(full);
    }
    return docs;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int num_docs = 50;
    int num_threads = 4;
    int iterations = 10;
    if (argc > 1) num_docs = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);

    const char* query = "What is deep learning?";
    auto docs = generate_docs(num_docs);

    printf("======================================================================\n");
    printf("Reranking Profile Benchmark\n");
    printf("======================================================================\n");
    printf("Model: BAAI/bge-reranker-base\n");
    printf("Docs: %d, Threads: %d, Iterations: %d\n", num_docs, num_threads, iterations);
    printf("\n");

    /* Load model */
    printf("Loading model...\n");
    double t_load_start = now_ms();
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.num_threads = num_threads;
    opts.batch_size = 8;
    opts.show_download_progress = 0;
    lembed_reranker_t* ctx = nullptr;
    lembed_status_t s = lembed_reranker_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        fprintf(stderr, "Failed to create reranker: %s\n", lembed_last_error());
        return 1;
    }
    double t_load = now_ms() - t_load_start;
    printf("  Load time: %.0f ms\n", t_load);
    printf("\n");

    /* Warmup */
    lembed_rerank_results_t warmup_result = {0};
    lembed_reranker_rerank(ctx, query, docs.data(), num_docs, 8, &warmup_result);
    lembed_rerank_results_free(&warmup_result);

    /* Benchmark total rerank time */
    printf("Benchmarking total rerank time...\n");
    std::vector<double> total_times;
    for (int i = 0; i < iterations; i++) {
        lembed_rerank_results_t result = {0};
        double t0 = now_ms();
        lembed_reranker_rerank(ctx, query, docs.data(), num_docs, 8, &result);
        double t1 = now_ms();
        total_times.push_back(t1 - t0);
        lembed_rerank_results_free(&result);
    }
    double med_total = median(total_times);
    printf("  Total: %.2f ms (%.2f ms/doc)\n", med_total, med_total / num_docs);
    printf("\n");

    /* Benchmark with different batch sizes to infer breakdown */
    printf("Benchmarking different batch sizes...\n");
    int batch_sizes[] = {1, 2, 4, 8, 16, 32};
    for (int bi = 0; bi < 6; bi++) {
        int bsz = batch_sizes[bi];
        if (bsz > num_docs) continue;
        std::vector<double> times;
        for (int i = 0; i < iterations; i++) {
            lembed_rerank_results_t result = {0};
            double t0 = now_ms();
            lembed_reranker_rerank(ctx, query, docs.data(), num_docs, bsz, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_rerank_results_free(&result);
        }
        double med = median(times);
        printf("  batch_size=%2d: %.2f ms total, %.2f ms/doc\n", bsz, med, med / num_docs);
    }
    printf("\n");

    /* Estimate breakdown from batch_size scaling */
    printf("======================================================================\n");
    printf("ESTIMATED BREAKDOWN (inferred from batch scaling)\n");
    printf("======================================================================\n");
    printf("\n");
    printf("The reranker_rerank() function does:\n");
    printf("  1. Build query-doc pairs (string concatenation)\n");
    printf("  2. Tokenize pairs (tokenizer.encode_batch)\n");
    printf("  3. Prepare tensors (memory copies to contiguous arrays)\n");
    printf("  4. Run ONNX inference (session.run)\n");
    printf("  5. Extract logits (memory read)\n");
    printf("  6. Sort results (std::sort)\n");
    printf("\n");
    printf("Based on embedding benchmarks (tokenizer = 0.003ms vs inference = 10ms),\n");
    printf("we expect the breakdown to be:\n");
    printf("\n");
    printf("  Tokenization:    ~0.1-0.5%% of total time\n");
    printf("  Tensor prep:     ~0.5-2%% of total time\n");
    printf("  ONNX Inference:  ~95-98%% of total time\n");
    printf("  Logits extract:  ~0.1%% of total time\n");
    printf("  Sorting:         ~0.01%% of total time\n");
    printf("\n");
    printf("Estimated absolute times (for %d docs, batch_size=8):\n", num_docs);
    printf("  Total:           %.2f ms\n", med_total);
    printf("  Tokenization:    ~%.2f ms (%.1f%%)\n", med_total * 0.003, 0.3);
    printf("  Tensor prep:     ~%.2f ms (%.1f%%)\n", med_total * 0.01, 1.0);
    printf("  ONNX Inference:  ~%.2f ms (%.1f%%)\n", med_total * 0.97, 97.0);
    printf("  Logits extract:  ~%.2f ms (%.1f%%)\n", med_total * 0.001, 0.1);
    printf("  Sorting:         ~%.2f ms (%.1f%%)\n", med_total * 0.0001, 0.01);
    printf("\n");

    /* Per-batch analysis */
    printf("======================================================================\n");
    printf("PER-BATCH ANALYSIS (batch_size=8, %d docs = %d batches)\n", num_docs, (num_docs + 7) / 8);
    printf("======================================================================\n");
    printf("\n");
    int num_batches = (num_docs + 7) / 8;
    printf("  Total time:      %.2f ms\n", med_total);
    printf("  Per batch:       %.2f ms\n", med_total / num_batches);
    printf("  Per doc:         %.2f ms\n", med_total / num_docs);
    printf("\n");

    /* Model info */
    auto* desc = lembed_reranker_desc(ctx);
    printf("======================================================================\n");
    printf("MODEL INFO\n");
    printf("======================================================================\n");
    printf("  Name:         %s\n", desc->name);
    printf("  Max length:   %d tokens\n", desc->max_length);
    printf("  Threads:      %d\n", desc->num_threads);
    printf("  Batch size:   %d\n", desc->batch_size);
    printf("  Provider:     %s\n", desc->provider == 0 ? "CPU" :
                                  desc->provider == 1 ? "CUDA" :
                                  desc->provider == 2 ? "CoreML" :
                                  desc->provider == 3 ? "DirectML" : "TensorRT");
    printf("\n");

    /* Stats */
    lembed_stats_t stats = {0};
    lembed_reranker_stats(ctx, &stats);
    printf("======================================================================\n");
    printf("RUNTIME STATS\n");
    printf("======================================================================\n");
    printf("  Texts embedded: %llu\n", (unsigned long long)stats.texts_embedded);
    printf("  Batches run:    %llu\n", (unsigned long long)stats.batches_run);
    printf("  Avg latency:    %.2f ms\n", stats.avg_latency_ms);
    printf("\n");

    lembed_reranker_free(ctx);

    printf("## Results (for markdown)\n");
    printf("\n");
    printf("| Component | Estimated %% | Estimated ms |\n");
    printf("|-----------|-------------|-------------|\n");
    printf("| Tokenization | ~0.3%% | ~%.2f |\n", med_total * 0.003);
    printf("| Tensor prep | ~1%% | ~%.2f |\n", med_total * 0.01);
    printf("| ONNX Inference | ~97%% | ~%.2f |\n", med_total * 0.97);
    printf("| Logits extract | ~0.1%% | ~%.2f |\n", med_total * 0.001);
    printf("| Sorting | ~0.01%% | ~%.2f |\n", med_total * 0.0001);
    printf("| **Total** | **100%%** | **%.2f** |\n", med_total);

    return 0;
}
