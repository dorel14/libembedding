/*
 * Test the enriched autotune cache (hardware+software+model fingerprint)
 * with all configurations stored.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_USE_LLAMACPP
#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    fprintf(stderr, "=== Enriched Autotune Cache Test ===\n\n");

    /* Hardware detection */
    fprintf(stderr, "--- Hardware ---\n");
    lembed_hardware_info_t hw = {0};
    lembed_detect_hardware(&hw);
    fprintf(stderr, "  CPU: %s\n", hw.cpu_name);
    fprintf(stderr, "  Cores: %dP/%dL\n", hw.physical_cores, hw.logical_cores);
    fprintf(stderr, "  RAM: %d MB\n", hw.ram_mb);
    fprintf(stderr, "  Features: %s\n", hw.features);
    fprintf(stderr, "  OS: %s\n", hw.os_name);

    /* Software detection */
    fprintf(stderr, "\n--- Software ---\n");
    lembed_software_info_t sw = {0};
    lembed_detect_software(&sw);
    fprintf(stderr, "  libembedding: %s\n", sw.libembedding);
    fprintf(stderr, "  llama.cpp: %s\n", sw.llama_cpp);

    /* Model info */
    fprintf(stderr, "\n--- Model ---\n");
    lembed_model_fingerprint_t model = {0};
    strncpy(model.model_id, "snowflake-xs-Q4", sizeof(model.model_id) - 1);
    strncpy(model.quantization, "Q4_K_M", sizeof(model.quantization) - 1);
    model.dim = 384;
    model.file_size_bytes = 20971520;
    fprintf(stderr, "  ID: %s\n", model.model_id);
    fprintf(stderr, "  Quantization: %s\n", model.quantization);
    fprintf(stderr, "  Dim: %d\n", model.dim);

    /* Cache key */
    fprintf(stderr, "\n--- Cache Key ---\n");
    char key[256];
    lembed_tune_cache_key(&hw, &sw, &model, "llama.cpp", key);
    fprintf(stderr, "  Key: %s\n", key);

    /* Build entry with all configs */
    fprintf(stderr, "\n--- Save (all configs) ---\n");
    lembed_tune_cache_entry_t entry = {0};
    entry.cache_schema_version = 1;
    entry.hardware = hw;
    entry.software = sw;
    entry.model = model;
    strncpy(entry.backend, "llama.cpp", sizeof(entry.backend) - 1);
    /* Add all measured configs */
    lembed_tune_config_result_t c1 = {1, 4, 0, 95.0f, 10.5f, 12.0f};
    lembed_tune_config_result_t c2 = {2, 1, 0, 186.0f, 5.4f, 7.1f};
    lembed_tune_config_result_t c3 = {3, 1, 0, 198.0f, 5.2f, 6.8f};
    lembed_tune_config_result_t c4 = {4, 1, 0, 204.0f, 5.1f, 6.5f};
    lembed_tune_cache_add_config(&entry, &c1);
    lembed_tune_cache_add_config(&entry, &c2);
    lembed_tune_cache_add_config(&entry, &c3);
    lembed_tune_cache_add_config(&entry, &c4);
    lembed_tune_cache_set_best(&entry, 2); /* 3 sessions = best efficient optimum */
    fprintf(stderr, "  Configs stored: %d\n", entry.num_configs);
    fprintf(stderr, "  Best index: %d (3 sessions, %.1f docs/s)\n", entry.best_idx, entry.configs[entry.best_idx].throughput_docs_sec);
    lembed_status_t s = lembed_tune_cache_save(&entry);
    fprintf(stderr, "  Save: %s\n", s == LEMBED_OK ? "OK" : "FAILED");

    /* Load */
    fprintf(stderr, "\n--- Load ---\n");
    lembed_tune_cache_entry_t loaded = {0};
    s = lembed_tune_cache_load(&hw, &sw, &model, "llama.cpp", &loaded);
    if (s == LEMBED_OK) {
        fprintf(stderr, "  Model: %s\n", loaded.model.model_id);
        fprintf(stderr, "  Backend: %s\n", loaded.backend);
        fprintf(stderr, "  Configs loaded: %d\n", loaded.num_configs);
        fprintf(stderr, "  Best index: %d\n", loaded.best_idx);
        fprintf(stderr, "  Best throughput: %.1f docs/s\n", loaded.configs[loaded.best_idx].throughput_docs_sec);
    } else {
        fprintf(stderr, "  FAILED\n");
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
