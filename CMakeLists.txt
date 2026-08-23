cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0144 NEW)
project(libembedding
    VERSION 0.2.0
    DESCRIPTION "C header-only embedding library using ONNX Runtime"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_UNUSED_CLI_VARIABLES ON)

if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# =========================================================================
# Options
# =========================================================================
option(LIBEMBEDDING_NO_DOWNLOAD   "Disable model downloading (offline only)"  OFF)
option(LIBEMBEDDING_NO_IMAGE      "Disable image embedding support"           OFF)
option(LIBEMBEDDING_BUILD_TESTS   "Build tests"                               ON)
option(LIBEMBEDDING_BUILD_EXAMPLES "Build examples"                           ON)
option(LIBEMBEDDING_BUILD_BENCHMARKS "Build benchmarks"                       OFF)
option(LIBEMBEDDING_BUILD_SHARED    "Build shared library for bindings"       OFF)

# =========================================================================
# Dependencies
# =========================================================================

if(WIN32)
    # Allow overriding bundled third_party paths via environment variable
    # (e.g. for CI builds downloading onnxruntime from GitHub releases)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        string(REPLACE "/" "\\" _ort_root "$ENV{ONNXRUNTIME_ROOT}")
        set(ONNXRuntime_INCLUDE_DIR "${_ort_root}\\include")
        set(ONNXRuntime_LIBRARY "${_ort_root}\\lib\\onnxruntime.lib")
    elseif(NOT ONNXRuntime_INCLUDE_DIR)
        set(ONNXRuntime_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
    endif()
    if(NOT ONNXRuntime_LIBRARY)
        set(ONNXRuntime_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.lib")
    endif()
    include_directories(${ONNXRuntime_INCLUDE_DIR})

    if(NOT CURL_INCLUDE_DIR)
        set(CURL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/include")
    endif()
    if(NOT CURL_LIBRARY)
        set(CURL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/lib/libcurl.dll.a")
    endif()
    include_directories(${CURL_INCLUDE_DIR})

    # 🟢 AJOUT CRUCIAL : Créer la cible manquante attendue par les tests et exemples
    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
        )
    endif()

    # Désactiver globalement les macros min/max conflictuelles de Windows
    add_compile_definitions(NOMINMAX)
else()
    find_package(CURL REQUIRED)
endif()

# ONNX Runtime (required)
# Use our custom FindONNXRuntime module which handles downloaded archives
# and system installs correctly
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Hint Homebrew paths on macOS
if(APPLE AND EXISTS "/opt/homebrew/opt/onnxruntime")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/onnxruntime")
endif()

find_package(ONNXRuntime REQUIRED)

# cJSON (bundled, for parsing tokenizer.json and config files)
add_library(cjson STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON/cJSON.c"
)
target_include_directories(cjson PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(cjson PROPERTIES POSITION_INDEPENDENT_CODE ON)

# stb (bundled image loading/resizing, optional)
if(NOT LIBEMBEDDING_NO_IMAGE)
    add_library(stb STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/stb_impl.c"
    )
    target_include_directories(stb PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_target_properties(stb PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# libcurl (optional, for downloading)
if(NOT LIBEMBEDDING_NO_DOWNLOAD)
    find_package(CURL REQUIRED)
endif()



# =========================================================================
# Main library target (Forcé en DLL SHARED pour Windows/Python cffi)
# =========================================================================
if(WIN32)
    # On crée une vraie DLL. Si un fichier cpp d'implémentation existe (ex: src/libembedding.cpp), 
    # ajoutez-le ici à la place de ""
    add_library(libembedding SHARED "${CMAKE_CURRENT_SOURCE_DIR}/python/csrc/libembedding_shlib.cpp")

    set_target_properties(libembedding PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    
    # On force CMake à accepter une bibliothèque partagée sans fichiers sources initiaux 
    # car elle va lier ONNXRuntime et CURL
    set_target_properties(libembedding PROPERTIES LINKER_LANGUAGE CXX)
else()
    # On laisse la configuration d'origine pour Linux / macOS
    add_library(libembedding INTERFACE)
endif()

add_library(libembedding::libembedding ALIAS libembedding)

# =========================================================================
# CPU Optimizations
# =========================================================================
# Note: /GL (whole-program opt) and /LTCG are intentionally omitted on MSVC
# because they are incompatible with WINDOWS_EXPORT_ALL_SYMBOLS, which
# CMake needs to parse object files to auto-generate the export definition.
# To use LTCG, provide an explicit .def file instead.
if(MSVC)
    target_compile_options(libembedding PRIVATE
        /O2       # Maximize speed
        /Oi       # Favor fast code (intrinsics)
        /Ot       # Favor fast over small code
        /fp:fast  # Fast floating-point model
    )
    # AVX2 for Ice Lake and newer (safe baseline for modern x86)
    target_compile_options(libembedding PRIVATE /arch:AVX2)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        target_compile_definitions(libembedding PRIVATE NDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libembedding INTERFACE
        -O3
        -ffast-math
    )
    # Enable AVX2/AVX-512 where available
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" _has_avx2)
        if(_has_avx2)
            target_compile_options(libembedding INTERFACE -mavx2)
        endif()
    endif()
endif()

# =========================================================================
# Configuration des Inclusions et Liaisons (Adaptée Windows/DLL)
# =========================================================================
if(WIN32)
    # Mode DLL (SHARED) pour Windows
    target_include_directories(libembedding PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(libembedding PUBLIC
        ONNXRuntime::ONNXRuntime
        cjson
    )

    if(NOT LIBEMBEDDING_NO_IMAGE)
        target_link_libraries(libembedding PUBLIC stb)
    endif()

    if(NOT LIBEMBEDDING_NO_DOWNLOAD)
        target_link_libraries(libembedding PUBLIC ${CURL_LIBRARY})
    else()
        target_compile_definitions(libembedding PUBLIC LIBEMBEDDING_NO_DOWNLOAD)
    endif()

    if(LIBEMBEDDING_NO_IMAGE)
        target_compile_definitions(libembedding PUBLIC LIBEMBEDDING_NO_IMAGE)
    endif()
else()
    # Mode d'origine (INTERFACE) pour Linux / macOS
    target_include_directories(libembedding INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(libembedding INTERFACE
        ONNXRuntime::ONNXRuntime
        cjson
    )

    if(NOT LIBEMBEDDING_NO_IMAGE)
        target_link_libraries(libembedding INTERFACE stb)
    endif()

    if(NOT LIBEMBEDDING_NO_DOWNLOAD)
        target_link_libraries(libembedding INTERFACE CURL::libcurl)
    else()
        target_compile_definitions(libembedding INTERFACE LIBEMBEDDING_NO_DOWNLOAD)
    endif()

    if(LIBEMBEDDING_NO_IMAGE)
        target_compile_definitions(libembedding INTERFACE LIBEMBEDDING_NO_IMAGE)
    endif()
endif()

# =========================================================================
# Helper: copy runtime DLLs next to executables on Windows
# =========================================================================
if(WIN32)
    set(_LEMBED_ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib")
    function(copy_runtime_dlls target_name)
        # ONNX Runtime DLLs
        set(_dll_dirs
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib"
        )
        set(_dll_names
            onnxruntime.dll
            onnxruntime_providers_shared.dll
        )
        # Also copy libembedding.dll if it exists in build output
        set(_lib_dir "${CMAKE_BINARY_DIR}/$<CONFIG>")
        if(EXISTS "${_lib_dir}/libembedding.dll")
            list(APPEND _dll_names libembedding.dll)
        endif()

        # Curl DLL (from CURL_LIBRARY or CURL_DIR)
        if(DEFINED CURL_LIBRARY)
            get_filename_component(_curl_lib_dir "${CURL_LIBRARY}" DIRECTORY)
            if(EXISTS "${_curl_lib_dir}/../bin/libcurl-x64.dll")
                list(APPEND _dll_dirs "${_curl_lib_dir}/../bin")
                list(APPEND _dll_names libcurl-x64.dll)
            endif()
        endif()

        foreach(_dll ${_dll_names})
            foreach(_dir ${_dll_dirs})
                if(EXISTS "${_dir}/${_dll}")
                    add_custom_command(TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${_dir}/${_dll}"
                            "$<TARGET_FILE_DIR:${target_name}>"
                    )
                endif()
            endforeach()
        endforeach()
    endfunction()
endif()

# =========================================================================
# Tests
# =========================================================================
if(LIBEMBEDDING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =========================================================================
# Examples
# =========================================================================
if(LIBEMBEDDING_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =========================================================================
# Benchmarks
# =========================================================================
if(LIBEMBEDDING_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# =========================================================================
# Shared library (for Python/FFI bindings)
# =========================================================================
if(LIBEMBEDDING_BUILD_SHARED)
    add_library(embedding_shared SHARED python/csrc/libembedding_shlib.cpp)
    target_link_libraries(embedding_shared PRIVATE libembedding)
    set_target_properties(embedding_shared PROPERTIES
        OUTPUT_NAME "embedding"
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
    if(APPLE)
        set_target_properties(embedding_shared PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(embedding_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
endif()

# =========================================================================
# Install
# =========================================================================
include(GNUInstallDirs)

install(DIRECTORY include/libembedding
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY third_party/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libembedding/detail
    FILES_MATCHING PATTERN "*.h"
)

set(_install_targets libembedding cjson)
if(NOT LIBEMBEDDING_NO_IMAGE AND TARGET stb)
    list(APPEND _install_targets stb)
endif()
install(TARGETS ${_install_targets}
    EXPORT libembedding-targets
)

install(EXPORT libembedding-targets
    FILE libembeddingConfig.cmake
    NAMESPACE libembedding::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libembedding
)
