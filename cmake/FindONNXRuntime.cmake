# FindONNXRuntime.cmake
# Locates ONNX Runtime installation
#
# Sets:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES
#
# Creates imported target: ONNXRuntime::ONNXRuntime
#
# Hints:
#   ONNXRUNTIME_ROOT (env or cmake var)

# Search paths
set(_onnxruntime_search_paths "")

if(DEFINED ENV{ONNXRUNTIME_ROOT})
    list(APPEND _onnxruntime_search_paths "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(DEFINED ONNXRUNTIME_ROOT)
    list(APPEND _onnxruntime_search_paths "${ONNXRUNTIME_ROOT}")
endif()

# Common install locations
list(APPEND _onnxruntime_search_paths
    "/usr/local"
    "/usr"
    "/opt/onnxruntime"
    "$ENV{HOME}/.local"
)

# macOS Homebrew
if(APPLE)
    list(APPEND _onnxruntime_search_paths
        "/opt/homebrew"
        "/opt/homebrew/opt/onnxruntime"
    )
endif()

# Find include directory
find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATHS ${_onnxruntime_search_paths}
    PATH_SUFFIXES include include/onnxruntime
)

# Find library
find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    PATHS ${_onnxruntime_search_paths}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
    set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})

    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
