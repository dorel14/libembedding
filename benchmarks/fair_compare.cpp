/*
 * Fair comparison: C++ vs Python with identical corpus
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
static double now_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
static double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        clk::now().time_since_epoch()
    );
    return (double)us.count() / 1000.0;
}
#endif

int main() {
    printf("=== Fair C++ Benchmark ===\n\n");

    const char* texts[] = {
        "Hello world.",
        "Bonjour le monde.",
        "Hallo Welt.",
        "Hola mundo.",
    };
    int ntexts = sizeof(texts) / sizeof(texts[0]);

    printf("Corpus: %d texts\n\n", ntexts);

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;
    opts.num_threads = 4;
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &emb);
    if (s != LEMBED_OK) {
        printf("Error: %s\n", lembed_last_error());
        return 1;
    }

    printf("Dimension: %d\n", lembed_text_embedding_dim(emb));

    std::vector<const char*> ctexts;
    for (int i = 0; i < ntexts; i++) ctexts.push_back(texts[i]);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        lembed_embeddings_t w = {0};
        lembed_text_embedding_embed(emb, ctexts.data(), ntexts, ntexts, &w);
        lembed_embeddings_free(&w);
    }

    /* Benchmark: 100 batches */
    double t0 = now_ms();
    int nbatches = 100;
    for (int b = 0; b < nbatches; b++) {
        lembed_embeddings_t res = {0};
        lembed_text_embedding_embed(emb, ctexts.data(), ntexts, ntexts, &res);
        lembed_embeddings_free(&res);
    }
    double t1 = now_ms();

    double total_ms = t1 - t0;
    double per_batch = total_ms / nbatches;
    printf("Result: %d batches of %d texts in %.1f ms = %.2f ms/batch = %.0f docs/s\n",
           nbatches, ntexts, total_ms, per_batch, (double)ntexts / (per_batch/1000.0));

    lembed_text_embedding_free(emb);

    printf("=== Done ===\n");
    return 0;
}
