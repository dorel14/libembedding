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

# Test similarity functions (no download required)
add_executable(test_similarity test_similarity.cpp)
target_link_libraries(test_similarity PRIVATE libembedding)
add_test(NAME test_similarity COMMAND test_similarity)

# Test local loading (no network required — tests error paths)
add_executable(test_local_loading test_local_loading.cpp)
target_link_libraries(test_local_loading PRIVATE libembedding)
# This test only exercises create_from_path error paths; no download code needed.
# Defining LIBEMBEDDING_NO_DOWNLOAD avoids pulling in libcurl.dll dependency.
target_compile_definitions(test_local_loading PRIVATE LIBEMBEDDING_NO_DOWNLOAD)
add_test(NAME test_local_loading COMMAND test_local_loading)

# On Windows, copy runtime DLLs (onnxruntime, libcurl) next to test executables
if(WIN32)
    foreach(_t test_model_registry test_pooling test_similarity test_local_loading)
        if(TARGET ${_t})
            copy_runtime_dlls(${_t})
        endif()
    endforeach()
endif()

# =========================================================================
# Integration Tests (require model downloads)
# =========================================================================
option(LIBEMBEDDING_INTEGRATION_TESTS "Enable integration tests (require network)" OFF)

if(LIBEMBEDDING_INTEGRATION_TESTS)
    add_executable(test_text_embedding test_text_embedding.cpp)
    target_link_libraries(test_text_embedding PRIVATE libembedding)
    add_test(NAME test_text_embedding COMMAND test_text_embedding)

    add_executable(test_introspection test_introspection.cpp)
    target_link_libraries(test_introspection PRIVATE libembedding)
    add_test(NAME test_introspection COMMAND test_introspection)

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

    if(WIN32)
        foreach(_t test_text_embedding test_introspection test_sparse_embedding test_reranker)
            copy_runtime_dlls(${_t})
        endforeach()
        if(NOT LIBEMBEDDING_NO_IMAGE)
            copy_runtime_dlls(test_image_embedding)
        endif()
    endif()
endif()
