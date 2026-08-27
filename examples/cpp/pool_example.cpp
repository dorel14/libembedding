/*
 * Example: EmbeddingPool - performance test
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <libembedding/cpp/embedding_pool.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    fprintf(stderr, "=== EmbeddingPool Performance ===\n\n");

    /* Configure pool */
    lembed::PoolOptions opts;
    opts.model_path = "C:/Users/david/.cache/libembedding/models--Qdrant-all-MiniLM-L6-v2-onnx";
    opts.num_workers = 8;
    opts.threads_per_worker = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;
    opts.dim = 384;

    fprintf(stderr, "Creating pool with %d workers...\n", opts.num_workers);

    auto t0 = std::chrono::high_resolution_clock::now();
    lembed::EmbeddingPool pool(opts);
    auto t1 = std::chrono::high_resolution_clock::now();

    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Pool created in %.0f ms (dim=%d, workers=%d)\n\n", load_ms, pool.dimension(), pool.num_workers());

    /* Prepare texts */
    int n_texts = 256;
    std::vector<std::string> texts;
    texts.reserve(n_texts);
    for (int i = 0; i < n_texts; i++) {
        texts.push_back("The quick brown fox jumps over the lazy dog. Sentence number " + std::to_string(i));
    }

    /* Warmup */
    pool.embed({texts[0], texts[1], texts[2]}, 3);

    /* Benchmark */
    printf("Embedding %d texts...\n", n_texts);

    t0 = std::chrono::high_resolution_clock::now();
    auto embeddings = pool.embed(texts, 64);
    t1 = std::chrono::high_resolution_clock::now();

    double embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double texts_per_sec = (double)n_texts / (embed_ms / 1000.0);

    printf("Result: %d embeddings in %.1f ms = %.1f texts/sec\n",
           (int)embeddings.size(), embed_ms, texts_per_sec);

    /* Print first few values of first embedding */
    if (!embeddings.empty()) {
        printf("First embedding [0:5]: ");
        for (int i = 0; i < 5 && i < (int)embeddings[0].size(); i++) {
            printf("%.4f ", embeddings[0][i]);
        }
        printf("\n");
    }

    /* Test with different worker counts */
    printf("\n=== Scaling workers ===\n");
    int worker_counts[] = {1, 2, 4, 8};
    for (int wi = 0; wi < 4; wi++) {
        int nw = worker_counts[wi];
        lembed::PoolOptions opts2 = opts;
        opts2.num_workers = nw;

        lembed::EmbeddingPool pool2(opts2);

        /* warmup */
        pool2.embed({texts[0]}, 1);

        t0 = std::chrono::high_resolution_clock::now();
        auto emb2 = pool2.embed(texts, 64);
        t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double tps = (double)n_texts / (ms / 1000.0);
        printf("  %d workers: %.1f texts/sec\n", nw, tps);
    }

    printf("\nDone.\n");
    return 0;
}
