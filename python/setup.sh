#!/usr/bin/env bash
#
# setup.sh — Build libembedding shared library and prepare the Python package.
#
# Usage:
#   ./setup.sh                  # Build shared lib + install package in editable mode
#   ./setup.sh --build-only     # Build shared lib only (no pip install)
#   ./setup.sh --wheel          # Build a wheel for distribution
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
PKG_DIR="$SCRIPT_DIR/src/libembedding"

# ── Detect platform ──────────────────────────────────────────────
case "$(uname -s)" in
    Darwin)  LIB_EXT=".dylib" ;;
    Linux)   LIB_EXT=".so" ;;
    MINGW*|MSYS*|CYGWIN*) LIB_EXT=".dll" ;;
    *)       LIB_EXT=".so" ;;
esac

# ── Step 1: Build the shared library ─────────────────────────────
echo "==> Building libembedding shared library..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBEMBEDDING_BUILD_SHARED=ON \
    -DLIBEMBEDDING_BUILD_TESTS=OFF \
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF \
    -DLIBEMBEDDING_BUILD_BENCHMARKS=OFF

cmake --build . --parallel --target embedding_shared

# ── Step 2: Copy shared library into the Python package ──────────
echo "==> Copying libembedding${LIB_EXT} into Python package..."
cp "$BUILD_DIR/libembedding${LIB_EXT}" "$PKG_DIR/"

# On macOS, fix the install name so cffi can find it by relative path
if [ "$LIB_EXT" = ".dylib" ]; then
    install_name_tool -id "@loader_path/libembedding.dylib" \
        "$PKG_DIR/libembedding.dylib" 2>/dev/null || true
fi

# ── Step 3: Copy LICENSE ─────────────────────────────────────────
if [ -f "$PROJECT_ROOT/LICENSE" ] && [ ! -f "$SCRIPT_DIR/LICENSE" ]; then
    cp "$PROJECT_ROOT/LICENSE" "$SCRIPT_DIR/"
fi

# ── Step 4: Install or build wheel ───────────────────────────────
cd "$SCRIPT_DIR"

case "${1:-}" in
    --build-only)
        echo "==> Done. Shared library built and copied."
        echo "    Run 'pip install -e .' in $SCRIPT_DIR to install."
        ;;
    --wheel)
        echo "==> Building wheel..."
        pip wheel . --no-deps -w dist/
        echo "==> Wheel built in $SCRIPT_DIR/dist/"
        ls -lh dist/*.whl
        ;;
    *)
        echo "==> Installing in editable mode..."
        pip install -e ".[dev]"
        echo "==> Done. Run 'pytest' to verify."
        ;;
esac
