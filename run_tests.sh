#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

INTEGRATION="${1:-off}"

echo "=== libembedding tests ==="

# Build first
EXTRA_FLAGS=""
if [ "${INTEGRATION}" = "--integration" ] || [ "${INTEGRATION}" = "-i" ]; then
    EXTRA_FLAGS="-DLIBEMBEDDING_INTEGRATION_TESTS=ON"
    echo "  Integration tests: ENABLED (requires network)"
else
    echo "  Integration tests: disabled (pass --integration to enable)"
fi

cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLIBEMBEDDING_BUILD_TESTS=ON \
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF \
    ${EXTRA_FLAGS}

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo ""
echo "--- Running tests ---"
cd "${BUILD_DIR}"
ctest --output-on-failure --test-dir . -j"${JOBS}"

echo ""
echo "=== All tests passed ==="
