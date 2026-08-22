/*
 * basic_embedding.c - Minimal example of using libembedding
 *
 * Demonstrates:
 *   - Creating a text embedder with default model
 *   - Embedding a few texts
 *   - Computing cosine similarity
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <math.h>
#include <stdio.h>

static float cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

int main(void) {
    /* List available models */
    const lembed_model_info_t* models;
    int count;
    lembed_list_text_models(&models, &count);
    printf("Available text embedding models: %d\n\n", count);

    /* Create embedder with defaults (BGE-small-en-v1.5, 384-dim, CPU) */
    lembed_text_options_t opts = lembed_text_options_default();
    opts.batch_size = 32;
    lembed_text_embedding_t* embedder = NULL;

    printf("Loading model: %s\n", models[opts.model].model_name);
    lembed_status_t status = lembed_text_embedding_create(&opts, &embedder);
    if (status != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n%s\n",
                lembed_status_message(status), lembed_last_error());
        return 1;
    }

    /* Print model info via introspection */
    const lembed_model_desc_t* desc = lembed_text_embedding_desc(embedder);
    printf("Model loaded! Dimension: %d, batch_size: %d\n\n",
           lembed_text_embedding_dim(embedder), desc ? desc->batch_size : 0);

    /* Embed some texts */
    const char* texts[] = {
        "The cat sat on the mat",
        "A kitten rested on the rug",
        "Quantum physics describes subatomic particles"
    };

    lembed_embeddings_t result = {0};
    status = lembed_text_embedding_embed(embedder, texts, 3, 0, &result);
    if (status != LEMBED_OK) {
        fprintf(stderr, "Embed error: %s\n", lembed_last_error());
        lembed_text_embedding_free(embedder);
        return 1;
    }

    /* Compute pairwise cosine similarities */
    printf("Cosine similarities:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            float sim = cosine_sim(
                result.data + i * result.dim,
                result.data + j * result.dim,
                result.dim);
            printf("  [%d] vs [%d]: %.4f\n", i, j, sim);
        }
    }

    /* Cleanup */
    lembed_embeddings_free(&result);
    lembed_text_embedding_free(embedder);

    printf("\nDone!\n");
    return 0;
}
