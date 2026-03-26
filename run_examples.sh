#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

echo "=== libembedding examples ==="

# Build
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBEMBEDDING_BUILD_TESTS=OFF \
    -DLIBEMBEDDING_BUILD_EXAMPLES=ON

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

EXAMPLE="${1:-basic_embedding}"

BINARY="${BUILD_DIR}/examples/${EXAMPLE}"
if [ ! -f "${BINARY}" ]; then
    echo "Error: example '${EXAMPLE}' not found."
    echo "Available examples:"
    ls "${BUILD_DIR}/examples/" 2>/dev/null | grep -v CMake | grep -v cmake || echo "  (none built)"
    exit 1
fi

echo ""
echo "--- Running: ${EXAMPLE} ---"
echo ""
exec "${BINARY}"
