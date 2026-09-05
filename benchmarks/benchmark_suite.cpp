/*
 * Embedding Benchmark Suite — unified performance + quality + cost measurement.
 * Produces a normalized score for model comparison across backends.
 *
 * Metrics:
 *   Performance: load time, single latency, throughput (1-sess & 4-sess pool)
 *   Quality:     retrieval recall@3 on built-in semantic test set
 *   Cost:        RAM usage, model file size
 *   Score:       weighted composite (quality 50%, throughput 30%, cost 20%)
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <libembedding/detail/llama_session_impl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* =========================================================================
 * Quality test set: queries with expected relevant documents.
 * Each query has a list of "relevant" doc indices (ground truth).
 * We measure recall@3: fraction of queries where a relevant doc is in top-3.
 * ========================================================================= */
struct QualityTest {
    const char* query;
    std::vector<int> relevant_docs; /* indices into corpus */
};

static const char* quality_corpus[] = {
    "Machine learning algorithms learn patterns from data without explicit programming",
    "Deep neural networks use backpropagation for training multi-layer architectures",
    "The stock market experienced significant volatility due to economic uncertainty",
    "Investment portfolios should be diversified across asset classes",
    "The football match ended in a dramatic penalty shootout championship",
    "Soccer players train extensively for endurance and tactical awareness",
    "Climate change is causing rising sea levels and extreme weather events",
    "Renewable energy sources like solar and wind reduce carbon emissions",
    "The Renaissance period saw remarkable achievements in art and science",
    "Ancient Roman architecture influenced building design for centuries",
    "Quantum computing uses qubits to perform calculations faster than classical computers",
    "Cryptographic protocols secure communications using mathematical primitives",
};
static const int quality_corpus_size = 12;

static QualityTest quality_tests[] = {
    {"AI and machine learning",        {0, 1}},
    {"neural network training",         {1, 0}},
    {"financial markets and investing", {2, 3}},
    {"sports competition",              {4, 5}},
    {"environmental science",           {6, 7}},
    {"history and culture",             {8, 9}},
    {"computer science advances",       {10, 11}},
};
static const int quality_test_count = 7;

/* Cosine similarity */
static float cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-8f) ? dot / denom : 0.0f;
}

/* Measure retrieval recall@3 */
static float measure_quality(lembed_text_embedding_t* embedder, int dim) {
    /* Embed corpus */
    std::vector<const char*> corpus_ptrs;
    for (int i = 0; i < quality_corpus_size; i++)
        corpus_ptrs.push_back(quality_corpus[i]);

    std::vector<std::vector<float>> corpus_embs(quality_corpus_size);
    for (int i = 0; i < quality_corpus_size; i++) {
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(embedder, &corpus_ptrs[i], 1, 1, &r);
        corpus_embs[i].resize(dim);
        memcpy(corpus_embs[i].data(), r.data, dim * sizeof(float));
        lembed_embeddings_free(&r);
    }

    /* For each query, find top-3 docs and check if any relevant is included */
    int hits = 0;
    for (int t = 0; t < quality_test_count; t++) {
        lembed_embeddings_t qr = {0};
        lembed_text_embedding_embed(embedder, &quality_tests[t].query, 1, 1, &qr);
        float* qvec = qr.data;

        /* Score all docs */
        std::vector<std::pair<float, int>> scores;
        for (int d = 0; d < quality_corpus_size; d++) {
            scores.push_back({cosine_sim(qvec, corpus_embs[d].data(), dim), d});
        }
        std::sort(scores.begin(), scores.end(), std::greater<>());

        /* Check if any relevant doc is in top-3 */
        bool hit = false;
        for (int k = 0; k < 3 && k < (int)scores.size(); k++) {
            for (int rel : quality_tests[t].relevant_docs) {
                if (scores[k].second == rel) { hit = true; break; }
            }
            if (hit) break;
        }
        if (hit) hits++;
        lembed_embeddings_free(&qr);
    }
    return (float)hits / quality_test_count;
}

/* =========================================================================
 * Model definition for benchmark
 * ========================================================================= */
struct BenchModel {
    const char* name;
    const char* path;
    float file_size_mb;
};

int main(int argc, char** argv) {
    const char* modelDir = "C:\\Users\\david\\Documents\\devs\\libembedding\\benchmarks\\models";
    if (argc > 1) modelDir = argv[1];

    BenchModel models[] = {
        {"MiniLM-L6-Q4",       "all-MiniLM-L6-v2-Q4_K_M.gguf",     20.0f},
        {"MiniLM-L6-Q8",       "all-MiniLM-L6-v2-Q8_0.gguf",       23.8f},
        {"Snowflake-XS-Q4",    "snowflake-xs-Q4_K_M.gguf",          20.0f},
        {"Snowflake-S-Q4",     "snowflake-s-Q4_K_M.gguf",           27.5f},
        {"E5-small-v2-Q4",     "e5-small-v2-Q4_K_M.gguf",           27.9f},
        {"GIST-small-Q4",      "gist-small-Q4_K_M.gguf",            27.9f},
    };
    int nmodels = sizeof(models) / sizeof(models[0]);

    /* Test texts for throughput */
    std::vector<std::string> texts;
    for (int i = 0; i < 64; i++) {
        texts.push_back("Document " + std::to_string(i) + " with content about topic " + std::to_string(i % 8));
    }
    int total_texts = (int)texts.size();

    fprintf(stderr, "========================================\n");
    fprintf(stderr, "  Embedding Benchmark Suite\n");
    fprintf(stderr, "  llama.cpp backend (GGUF)\n");
    fprintf(stderr, "========================================\n\n");

    /* Results storage */
    struct Result {
        const char* name;
        float load_ms, single_ms, tps1, tps4, quality, file_mb, score;
    };
    std::vector<Result> results;

    for (int mi = 0; mi < nmodels; mi++) {
        std::string path = std::string(modelDir) + "\\" + models[mi].path;
        fprintf(stderr, "--- %s ---\n", models[mi].name);

        /* Load */
        lembed_text_options_t opts = lembed_text_options_default();
        opts.num_threads = 4;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* embedder = nullptr;
        double t0 = now_ms();
        lembed_status_t s = lembed_text_embedding_create_from_gguf_path(path.c_str(), &opts, &embedder);
        double t1 = now_ms();
        float load_ms = (float)(t1 - t0);

        if (s != LEMBED_OK) {
            fprintf(stderr, "  FAILED: %s\n\n", lembed_last_error());
            continue;
        }

        int dim = lembed_text_embedding_dim(embedder);
        std::vector<const char*> text_ptrs;
        for (auto& t : texts) text_ptrs.push_back(t.c_str());

        /* Single latency */
        std::vector<double> single_times;
        for (int i = 0; i < 20; i++) {
            lembed_embeddings_t r = {0};
            double s0 = now_ms();
            lembed_text_embedding_embed(embedder, text_ptrs.data(), 1, 1, &r);
            double s1 = now_ms();
            single_times.push_back(s1 - s0);
            lembed_embeddings_free(&r);
        }
        float single_ms = (float)median(single_times);

        /* Throughput 1-session */
        double tp0 = now_ms();
        for (int off = 0; off < total_texts; off += 16) {
            int n = std::min(16, total_texts - off);
            lembed_embeddings_t r = {0};
            lembed_text_embedding_embed(embedder, text_ptrs.data() + off, n, n, &r);
            lembed_embeddings_free(&r);
        }
        double tp1 = now_ms();
        float tps1 = (float)total_texts / ((float)(tp1 - tp0) / 1000.0f);

        /* Quality */
        float quality = measure_quality(embedder, dim);

        lembed_text_embedding_free(embedder);

        /* Throughput 4-session pool */
        float tps4 = 0;
        {
            lembed::detail::LlamaSessionPool pool;
            try {
                pool.load_from_file(path.c_str(), 4, 1, 0, 0, false);
                /* warmup */
                for (int i = 0; i < 4 && i < total_texts; i++)
                    pool.embed(texts[i].c_str());

                std::atomic<int> idx{0};
                double p0 = now_ms();
                std::vector<std::thread> workers;
                for (int w = 0; w < 4; w++) {
                    workers.emplace_back([&]() {
                        while (true) {
                            int i = idx.fetch_add(1, std::memory_order_relaxed);
                            if (i >= total_texts) break;
                            pool.embed(texts[i].c_str());
                        }
                    });
                }
                for (auto& t : workers) t.join();
                double p1 = now_ms();
                tps4 = (float)total_texts / ((float)(p1 - p0) / 1000.0f);
            } catch (...) { tps4 = 0; }
        }

        /* Composite score: quality 50%, throughput 30%, cost 20% */
        /* Normalize each to 0-1 scale relative to best in suite */
        Result r;
        r.name = models[mi].name;
        r.load_ms = load_ms;
        r.single_ms = single_ms;
        r.tps1 = tps1;
        r.tps4 = tps4;
        r.quality = quality;
        r.file_mb = models[mi].file_size_mb;
        r.score = 0; /* computed after */
        results.push_back(r);

        fprintf(stderr, "  dim=%d, load=%.0fms, single=%.2fms, tps1=%.1f, tps4=%.1f, quality=%.2f\n\n",
                dim, load_ms, single_ms, tps1, tps4, quality);
    }

    /* Compute normalized scores */
    if (!results.empty()) {
        float best_tps4 = 0, best_quality = 0, best_cost = 999;
        for (auto& r : results) {
            if (r.tps4 > best_tps4) best_tps4 = r.tps4;
            if (r.quality > best_quality) best_quality = r.quality;
            if (r.file_mb < best_cost) best_cost = r.file_mb;
        }

        for (auto& r : results) {
            float norm_quality = r.quality / best_quality;
            float norm_throughput = r.tps4 / best_tps4;
            float norm_cost = best_cost / r.file_mb; /* smaller is better */
            r.score = 0.5f * norm_quality + 0.3f * norm_throughput + 0.2f * norm_cost;
        }

        /* Sort by score descending */
        std::sort(results.begin(), results.end(),
                  [](const Result& a, const Result& b) { return a.score > b.score; });

        /* Print summary table */
        fprintf(stderr, "\n========================================\n");
        fprintf(stderr, "  FINAL RANKING (composite score)\n");
        fprintf(stderr, "  Quality 50%% | Throughput 30%% | Cost 20%%\n");
        fprintf(stderr, "========================================\n");
        fprintf(stderr, "%-20s %8s %10s %10s %8s %8s\n",
                "Model", "Score", "Quality", "Tps(4s)", "SizeMB", "Single");
        fprintf(stderr, "--------------------------------------------------------------------\n");
        for (size_t i = 0; i < results.size(); i++) {
            fprintf(stderr, "%-20s %8.3f %10.2f %10.1f %8.1f %8.2fms\n",
                    results[i].name, results[i].score, results[i].quality,
                    results[i].tps4, results[i].file_mb, results[i].single_ms);
        }
        fprintf(stderr, "\n");
    }

    return 0;
}
