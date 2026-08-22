cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0144 NEW)
project(libembedding
    VERSION 0.1.0
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
    # Définition des dossiers
    set(ONNXRuntime_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
    set(ONNXRuntime_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.lib")
    include_directories(${ONNXRuntime_INCLUDE_DIR})
    
    set(CURL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/include")
    set(CURL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/lib/libcurl.dll.a")
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
    find_package(ONNXRuntime REQUIRED)
    find_package(CURL REQUIRED)
endif()

# ONNX Runtime (required)
# Try the official cmake config first (ships with Homebrew, vcpkg, etc.)
# Then fall back to our custom FindONNXRuntime module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Hint Homebrew paths on macOS
if(APPLE AND EXISTS "/opt/homebrew/opt/onnxruntime")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/onnxruntime")
endif()

find_package(onnxruntime CONFIG QUIET)
if(onnxruntime_FOUND)
    message(STATUS "Found ONNX Runtime via cmake config")
    # The Homebrew config exports onnxruntime::onnxruntime
    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime ALIAS onnxruntime::onnxruntime)
    endif()
else()
    find_package(ONNXRuntime REQUIRED)
endif()

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
