add_executable(bench_libembedding bench_libembedding.cpp)
target_link_libraries(bench_libembedding PRIVATE libembedding)

add_executable(quick_bench quick_bench.cpp)
target_link_libraries(quick_bench PRIVATE libembedding)
target_compile_definitions(quick_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(bench_cached bench_cached.cpp)
target_link_libraries(bench_cached PRIVATE libembedding)
target_compile_definitions(bench_cached PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(profile_bench profile_bench.cpp)
target_link_libraries(profile_bench PRIVATE libembedding)
target_compile_definitions(profile_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(detail_profile detail_profile.cpp)
target_link_libraries(detail_profile PRIVATE libembedding)
target_compile_definitions(detail_profile PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(token_profile token_profile.cpp)
target_link_libraries(token_profile PRIVATE libembedding)
target_compile_definitions(token_profile PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(memcpy_profile memcpy_profile.cpp)
target_link_libraries(memcpy_profile PRIVATE libembedding)
target_compile_definitions(memcpy_profile PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(parallel_profile parallel_profile.cpp)
target_link_libraries(parallel_profile PRIVATE libembedding)
target_compile_definitions(parallel_profile PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(inference_profile inference_profile.cpp)
target_link_libraries(inference_profile PRIVATE libembedding)
target_compile_definitions(inference_profile PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(optimal_profile optimal_profile.cpp)
target_link_libraries(optimal_profile PRIVATE libembedding)
target_compile_definitions(optimal_profile PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(matrix_bench matrix_bench.cpp)
target_link_libraries(matrix_bench PRIVATE libembedding)
target_compile_definitions(matrix_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(minilm_bench minilm_bench.cpp)
target_link_libraries(minilm_bench PRIVATE libembedding)
target_compile_definitions(minilm_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(model_compare model_compare.cpp)
target_link_libraries(model_compare PRIVATE libembedding)

add_executable(verify_bench verify_bench.cpp)
target_link_libraries(verify_bench PRIVATE libembedding)
target_compile_definitions(verify_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(repro_bench repro_bench.cpp)
target_link_libraries(repro_bench PRIVATE libembedding)
target_compile_definitions(repro_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(long_bench long_bench.cpp)
target_link_libraries(long_bench PRIVATE libembedding)
target_compile_definitions(long_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(prod_bench prod_bench.cpp)
target_link_libraries(prod_bench PRIVATE libembedding)
target_compile_definitions(prod_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(robustness_bench robustness_bench.cpp)
target_link_libraries(robustness_bench PRIVATE libembedding)
target_compile_definitions(robustness_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(rigorous_bench rigorous_bench.cpp)
target_link_libraries(rigorous_bench PRIVATE libembedding)
target_compile_definitions(rigorous_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(stress_mem stress_mem.cpp)
target_link_libraries(stress_mem PRIVATE libembedding)
target_compile_definitions(stress_mem PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(debug_consistency debug_consistency.cpp)
target_link_libraries(debug_consistency PRIVATE libembedding)
target_compile_definitions(debug_consistency PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(fair_compare fair_compare.cpp)
target_link_libraries(fair_compare PRIVATE libembedding)
target_compile_definitions(fair_compare PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(long_text_bench long_text_bench.cpp)
target_link_libraries(long_text_bench PRIVATE libembedding)
target_compile_definitions(long_text_bench PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

add_executable(autotuner_test autotuner_test.cpp)
target_link_libraries(autotuner_test PRIVATE libembedding)
target_compile_definitions(autotuner_test PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

# llama.cpp backend benchmarks
add_executable(bench_llamacpp bench_llamacpp.cpp)
target_link_libraries(bench_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(bench_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(quick_bench_llamacpp quick_bench_llamacpp.cpp)
target_link_libraries(quick_bench_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(quick_bench_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(profile_bench_llamacpp profile_bench_llamacpp.cpp)
target_link_libraries(profile_bench_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(profile_bench_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(inference_profile_llamacpp inference_profile_llamacpp.cpp)
target_link_libraries(inference_profile_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(inference_profile_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(matrix_bench_llamacpp matrix_bench_llamacpp.cpp)
target_link_libraries(matrix_bench_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(matrix_bench_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(diag_llamacpp diag_llamacpp.cpp)
target_link_libraries(diag_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(diag_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(pool_bench_llamacpp pool_bench_llamacpp.cpp)
target_link_libraries(pool_bench_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(pool_bench_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(model_compare_llamacpp model_compare_llamacpp.cpp)
target_link_libraries(model_compare_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(model_compare_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(test_autotuner test_autotuner.cpp)
target_link_libraries(test_autotuner PRIVATE libembedding llama ggml)
target_compile_definitions(test_autotuner PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Critical experiment: batch vs pool
add_executable(batch_vs_pool batch_vs_pool.cpp)
target_link_libraries(batch_vs_pool PRIVATE libembedding llama ggml)
target_compile_definitions(batch_vs_pool PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Unified benchmark
add_executable(test_unified_benchmark test_unified_benchmark.cpp)
target_link_libraries(test_unified_benchmark PRIVATE libembedding llama ggml)
target_compile_definitions(test_unified_benchmark PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Benchmark sweep
add_executable(bench_sweep bench_sweep.cpp)
target_link_libraries(bench_sweep PRIVATE libembedding llama ggml)
target_compile_definitions(bench_sweep PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Global benchmark (ONNX vs llama.cpp)
add_executable(bench_global bench_global.cpp)
target_link_libraries(bench_global PRIVATE libembedding llama ggml)
target_compile_definitions(bench_global PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# ONNX diagnostic
add_executable(bench_diagnostic bench_diagnostic.cpp)
target_link_libraries(bench_diagnostic PRIVATE libembedding llama ggml)
target_compile_definitions(bench_diagnostic PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(baseline_bench baseline_bench.cpp)
target_link_libraries(baseline_bench PRIVATE libembedding llama ggml)
target_compile_definitions(baseline_bench PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Batch strategy comparison
add_executable(bench_batch_strategy bench_batch_strategy.cpp)
target_link_libraries(bench_batch_strategy PRIVATE libembedding llama ggml)
target_compile_definitions(bench_batch_strategy PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Quality benchmark (ONNX vs llama.cpp)
add_executable(bench_quality bench_quality.cpp)
target_link_libraries(bench_quality PRIVATE libembedding llama ggml)
target_compile_definitions(bench_quality PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Reranker comparison benchmark (ONNX vs llama.cpp)
add_executable(bench_reranker_compare bench_reranker_compare.cpp)
target_link_libraries(bench_reranker_compare PRIVATE libembedding llama ggml)
target_compile_definitions(bench_reranker_compare PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Bucketing benchmark for llama.cpp
add_executable(bench_bucket_llamacpp bench_bucket_llamacpp.cpp)
target_link_libraries(bench_bucket_llamacpp PRIVATE libembedding llama ggml)
target_compile_definitions(bench_bucket_llamacpp PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# llama.cpp headers needed for direct LlamaSession access
set(_llama_h_dir "${CMAKE_BINARY_DIR}/_deps/llama.cpp-src/include")
set(_ggml_h_dir "${CMAKE_BINARY_DIR}/_deps/llama.cpp-src/ggml/include")
if(EXISTS "${_llama_h_dir}/llama.h")
    foreach(_t bench_llamacpp quick_bench_llamacpp profile_bench_llamacpp inference_profile_llamacpp matrix_bench_llamacpp bench_bucket_llamacpp)
        target_include_directories(${_t} PRIVATE "${_llama_h_dir}" "${_ggml_h_dir}")
    endforeach()
endif()

if(NOT WIN32 AND ONNXRuntime_LIBRARIES)
    set(_all_benchmarks
        bench_libembedding quick_bench bench_cached profile_bench detail_profile
        token_profile memcpy_profile parallel_profile inference_profile optimal_profile
        matrix_bench minilm_bench model_compare verify_bench repro_bench long_bench
        prod_bench robustness_bench rigorous_bench stress_mem debug_consistency
        fair_compare long_text_bench autotuner_test
    )
    foreach(_t IN LISTS _all_benchmarks)
        if(TARGET ${_t})
            copy_runtime_libs(${_t})
        endif()
    endforeach()
endif()
