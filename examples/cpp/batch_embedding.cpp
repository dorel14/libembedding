/*
 * batch_embedding.c - Batch embedding and reranking example
 *
 * Demonstrates:
 *   - Text embedding with batching
 *   - Sparse embedding
 *   - Reranking
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <stdio.h>
#include <math.h>

int main(void) {
    lembed_status_t s;

    /* ============================================================
     * 1. Dense Text Embedding with custom model selection
     * ============================================================ */
    printf("=== Dense Text Embedding ===\n");
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;  /* 384-dim, fast */
        opts.batch_size = 32;

        /* Print model info */
        lembed_model_info_t info;
        lembed_get_text_model_info(opts.model, &info);
        printf("Model: %s (dim=%d)\n", info.model_name, info.dim);

        lembed_text_embedding_t* embedder = NULL;
        s = lembed_text_embedding_create(&opts, &embedder);
        if (s != LEMBED_OK) {
            printf("Error: %s\n", lembed_last_error());
            return 1;
        }

        /* Print runtime descriptor */
        const lembed_model_desc_t* desc = lembed_text_embedding_desc(embedder);
        if (desc) {
            printf("  threads=%d, batch_size=%d, provider=%d\n",
                   desc->num_threads, desc->batch_size, desc->provider);
        }

        /* Batch of texts */
        const char* texts[] = {
            "Machine learning enables computers to learn from data",
            "Deep learning uses neural networks with many layers",
            "Natural language processing helps computers understand text",
            "Computer vision allows machines to interpret images",
            "Reinforcement learning trains agents through rewards"
        };

        lembed_embeddings_t result = {0};
        s = lembed_text_embedding_embed(embedder, texts, 5, 2 /* batch_size */, &result);
        if (s == LEMBED_OK) {
            printf("Embedded %d texts, dim=%d\n", result.num_embeddings, result.dim);

            /* Show first few values of first embedding */
            printf("First embedding (first 5 dims): [");
            for (int i = 0; i < 5; i++) {
                printf("%.4f%s", result.data[i], i < 4 ? ", " : "");
            }
            printf("]\n");
        }

        lembed_embeddings_free(&result);
        lembed_text_embedding_free(embedder);
    }

    /* ============================================================
     * 2. Sparse Text Embedding
     * ============================================================ */
    printf("\n=== Sparse Text Embedding ===\n");
    {
        lembed_sparse_options_t opts = lembed_sparse_options_default();

        lembed_sparse_embedding_ctx_t* embedder = NULL;
        s = lembed_sparse_text_embedding_create(&opts, &embedder);
        if (s != LEMBED_OK) {
            printf("Sparse error: %s\n", lembed_last_error());
        } else {
            const char* texts[] = {
                "What is the capital of France?",
                "Paris is a beautiful city in Europe"
            };

            lembed_sparse_embeddings_t result = {0};
            s = lembed_sparse_text_embedding_embed(embedder, texts, 2, 0, &result);
            if (s == LEMBED_OK) {
                for (int i = 0; i < result.count; i++) {
                    printf("Text %d: %d non-zero dimensions\n",
                           i, result.items[i].length);
                }
            }

            lembed_sparse_embeddings_free(&result);
            lembed_sparse_text_embedding_free(embedder);
        }
    }

    /* ============================================================
     * 3. Reranking
     * ============================================================ */
    printf("\n=== Reranking ===\n");
    {
        lembed_reranker_options_t opts = lembed_reranker_options_default();

        lembed_reranker_t* reranker = NULL;
        s = lembed_reranker_create(&opts, &reranker);
        if (s != LEMBED_OK) {
            printf("Reranker error: %s\n", lembed_last_error());
        } else {
            const char* query = "How does photosynthesis work?";
            const char* docs[] = {
                "Photosynthesis converts light energy into chemical energy in plants",
                "The stock market fluctuated wildly today",
                "Chlorophyll absorbs sunlight during photosynthesis",
                "Football is the most popular sport worldwide",
                "Plants use CO2 and water to produce glucose and oxygen"
            };

            lembed_rerank_results_t result = {0};
            s = lembed_reranker_rerank(reranker, query, docs, 5, 0, &result);
            if (s == LEMBED_OK) {
                printf("Query: \"%s\"\n\nRanked results:\n", query);
                for (int i = 0; i < result.count; i++) {
                    printf("  %d. [score=%.4f] %s\n",
                           i + 1, result.items[i].score,
                           docs[result.items[i].index]);
                }
            }

            lembed_rerank_results_free(&result);
            lembed_reranker_free(reranker);
        }
    }

    printf("\nDone!\n");
    return 0;
}
