/* Ultra minimal test with file output */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#define LIBEMBEDDING_NO_IMAGE
#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    FILE* f = fopen("test_output.txt", "w");
    if (!f) return 1;

    fprintf(f, "STEP 1: Starting\n");
    fflush(f);

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 1;
    opts.offline = 1;

    fprintf(f, "STEP 2: Creating embedder\n");
    fflush(f);

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    fprintf(f, "STEP 3: Status=%d, embedder=%p\n", s, (void*)embedder);
    fflush(f);

    if (s != LEMBED_OK) {
        fprintf(f, "ERROR: %s\n", lembed_last_error());
        fflush(f);
        fclose(f);
        return 1;
    }

    fprintf(f, "STEP 4: Model loaded OK. dim=%d\n", lembed_text_embedding_dim(embedder));
    fflush(f);

    const char* text = "Hello world";
    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(embedder, &text, 1, 1, &result);
    fprintf(f, "STEP 5: Embed status=%d\n", s);
    fflush(f);

    if (s == LEMBED_OK) {
        fprintf(f, "Result: %d texts, dim=%d\n", result.num_embeddings, result.dim);
        fprintf(f, "First 5 values: %.4f %.4f %.4f %.4f %.4f\n",
               result.data[0], result.data[1], result.data[2], result.data[3], result.data[4]);
        fflush(f);
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(embedder);
    fprintf(f, "STEP 6: Done\n");
    fflush(f);
    fclose(f);
    return 0;
}
