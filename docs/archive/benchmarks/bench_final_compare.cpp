/*
 * Final Comparison: ONNX (with bucketing) vs llama.cpp
 * Tests mixed corpus with both backends
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_USE_LLAMACPP
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

int main() {
    fprintf(stderr, "=== Final Comparison: ONNX (bucketing) vs llama.cpp ===\n\n");

    const char* onnx_dir = "C:\\Users\\david\\.cache\\libembedding\\models--Qdrant-all-MiniLM-L6-v2-onnx";
    const char* gguf_path = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";

    /* Create mixed corpus */
    const char* short_texts[] = {"Hello world.", "Machine learning.", "AI is great.", "Fast embeddings."};
    const char* medium_texts[] = {
        "Machine learning algorithms can identify patterns in large datasets automatically.",
        "The transformer architecture has become the foundation for modern NLP.",
        "Deep learning is part of a broader family of machine learning methods.",
    };
    const char* long_texts[] = {
        "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing.",
        "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing.",
    };

    std::vector<const char*> corpus;
    for (auto t : short_texts) corpus.push_back(t);
    for (auto t : medium_texts) corpus.push_back(t);
    for (auto t : long_texts) corpus.push_back(t);
    int total = (int)corpus.size();

    fprintf(stderr, "Corpus: %d texts (mixed lengths)\n\n", total);

    /* ===== ONNX with bucketing (use_bucketing = 1) ===== */
    fprintf(stderr, "--- ONNX (with length bucketing) ---\n");
    lembed_text_options_t onnx_opts = lembed_text_options_default();
    onnx_opts.num_threads = 4;
    onnx_opts.batch_size = 16;
    onnx_opts.use_bucketing = 1;
    onnx_opts.show_download_progress = 0;

    lembed_text_embedding_t* onnx_emb = nullptr;
    if (lembed_text_embedding_create_from_path(onnx_dir, &onnx_opts, &onnx_emb) == LEMBED_OK) {
        lembed_embeddings_t onnx_result = {0};
        double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        lembed_text_embedding_embed(onnx_emb, corpus.data(), total, 16, &onnx_result);
        double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        float onnx_tps = (float)total / ((float)(t1 - t0) / 1000.0f);
        fprintf(stderr, "  Throughput: %.1f docs/s (%.2f ms total)\n", onnx_tps, (float)(t1-t0));
        fprintf(stderr, "  Dim: %d\n", onnx_result.dim);
        lembed_embeddings_free(&onnx_result);
        lembed_text_embedding_free(onnx_emb);
    }

    /* ===== ONNX without bucketing (use_bucketing = 0) ===== */
    fprintf(stderr, "\n--- ONNX (no bucketing, naive) ---\n");
    lembed_text_options_t onnx_opts2 = lembed_text_options_default();
    onnx_opts2.num_threads = 4;
    onnx_opts2.batch_size = 16;
    onnx_opts2.use_bucketing = 0;
    onnx_opts2.show_download_progress = 0;

    lembed_text_embedding_t* onnx_emb2 = nullptr;
    if (lembed_text_embedding_create_from_path(onnx_dir, &onnx_opts2, &onnx_emb2) == LEMBED_OK) {
        lembed_embeddings_t onnx_result2 = {0};
        double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        lembed_text_embedding_embed(onnx_emb2, corpus.data(), total, 16, &onnx_result2);
        double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        float onnx_tps2 = (float)total / ((float)(t1 - t0) / 1000.0f);
        fprintf(stderr, "  Throughput: %.1f docs/s (%.2f ms total)\n", onnx_tps2, (float)(t1-t0));
        lembed_embeddings_free(&onnx_result2);
        lembed_text_embedding_free(onnx_emb2);
    }

    /* ===== llama.cpp with session pool ===== */
    fprintf(stderr, "\n--- llama.cpp (session pool) ---\n");
    lembed_text_options_t llama_opts = lembed_text_options_default();
    llama_opts.num_threads = 1;
    llama_opts.show_download_progress = 0;

    lembed_text_embedding_t* llama_emb = nullptr;
    if (lembed_text_embedding_create_from_gguf_path(gguf_path, &llama_opts, &llama_emb) == LEMBED_OK) {
        lembed_embeddings_t llama_result = {0};
        double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        lembed_text_embedding_embed(llama_emb, corpus.data(), total, 1, &llama_result);
        double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        float llama_tps = (float)total / ((float)(t1 - t0) / 1000.0f);
        fprintf(stderr, "  Throughput: %.1f docs/s (%.2f ms total)\n", llama_tps, (float)(t1-t0));
        fprintf(stderr, "  Dim: %d\n", llama_result.dim);
        lembed_embeddings_free(&llama_result);
        lembed_text_embedding_free(llama_emb);
    }

    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "ONNX with bucketing should be significantly faster than naive ONNX.\n");
    fprintf(stderr, "llama.cpp is competitive but processes texts sequentially.\n");

    return 0;
}
