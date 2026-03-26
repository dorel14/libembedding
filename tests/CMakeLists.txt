# Tests for libembedding
# Unit tests can run without model downloads
# Integration tests require network access

# =========================================================================
# Unit Tests (no downloads required)
# =========================================================================

# Test model registry
add_executable(test_model_registry test_model_registry.cpp)
target_link_libraries(test_model_registry PRIVATE libembedding)
add_test(NAME test_model_registry COMMAND test_model_registry)

# Test pooling and normalization
add_executable(test_pooling test_pooling.cpp)
target_link_libraries(test_pooling PRIVATE libembedding)
add_test(NAME test_pooling COMMAND test_pooling)

# =========================================================================
# Integration Tests (require model downloads)
# =========================================================================
option(LIBEMBEDDING_INTEGRATION_TESTS "Enable integration tests (require network)" OFF)

if(LIBEMBEDDING_INTEGRATION_TESTS)
    add_executable(test_text_embedding test_text_embedding.cpp)
    target_link_libraries(test_text_embedding PRIVATE libembedding)
    add_test(NAME test_text_embedding COMMAND test_text_embedding)

    add_executable(test_sparse_embedding test_sparse_embedding.cpp)
    target_link_libraries(test_sparse_embedding PRIVATE libembedding)
    add_test(NAME test_sparse_embedding COMMAND test_sparse_embedding)

    add_executable(test_reranker test_reranker.cpp)
    target_link_libraries(test_reranker PRIVATE libembedding)
    add_test(NAME test_reranker COMMAND test_reranker)

    if(NOT LIBEMBEDDING_NO_IMAGE)
        add_executable(test_image_embedding test_image_embedding.cpp)
        target_link_libraries(test_image_embedding PRIVATE libembedding)
        add_test(NAME test_image_embedding COMMAND test_image_embedding)
    endif()
endif()
