#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# Parse arguments
BUILD_TYPE="${1:-Release}"
JOBS="${2:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

echo "=== libembedding build ==="
echo "  Build type:  ${BUILD_TYPE}"
echo "  Parallel:    ${JOBS} jobs"
echo "  Build dir:   ${BUILD_DIR}"
echo ""

# Configure
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLIBEMBEDDING_BUILD_TESTS=ON \
    -DLIBEMBEDDING_BUILD_EXAMPLES=ON \
    "$@"

# Build
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo ""
echo "=== Build complete ==="
echo "  Tests:    ${BUILD_DIR}/tests/"
echo "  Examples: ${BUILD_DIR}/examples/"
