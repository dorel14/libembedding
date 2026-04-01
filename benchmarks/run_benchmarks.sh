#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=== libembedding Benchmark Suite ==="
echo ""

# ── Check prerequisites ──────────────────────────────────────────
echo "Checking prerequisites..."

BENCH_BIN="$SCRIPT_DIR/../build/benchmarks/bench_libembedding"
if [ ! -f "$BENCH_BIN" ]; then
    echo "Building libembedding benchmark..."
    cd "$SCRIPT_DIR/.."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBEMBEDDING_BUILD_BENCHMARKS=ON
    cmake --build . --parallel --target bench_libembedding
    cd "$SCRIPT_DIR"
    BENCH_BIN="$SCRIPT_DIR/../build/benchmarks/bench_libembedding"
fi

HAS_PYTHON=0
if python3 -c "import fastembed" 2>/dev/null; then
    HAS_PYTHON=1
else
    echo "WARNING: fastembed (Python) not installed. Skipping."
    echo "  Install with: pip3 install fastembed"
fi

HAS_RUST=0
if command -v cargo &>/dev/null; then
    HAS_RUST=1
    if [ ! -f "$SCRIPT_DIR/bench_fastembed_rs/target/release/bench_fastembed_rs" ]; then
        echo "Building fastembed-rs benchmark..."
        cd "$SCRIPT_DIR/bench_fastembed_rs"
        cargo build --release 2>&1 | tail -3
        cd "$SCRIPT_DIR"
    fi
else
    echo "WARNING: cargo not found. Skipping fastembed-rs benchmark."
fi

echo ""

# ── Run benchmarks ───────────────────────────────────────────────
echo "=== Running libembedding (C++) benchmark ==="
"$BENCH_BIN" "$SCRIPT_DIR/corpus.txt" | tee "$RESULTS_DIR/results_libembedding.json"
echo ""

if [ "$HAS_PYTHON" -eq 1 ]; then
    echo "=== Running fastembed (Python) benchmark ==="
    python3 "$SCRIPT_DIR/bench_fastembed_py.py" "$SCRIPT_DIR/corpus.txt" | tee "$RESULTS_DIR/results_fastembed_py.json"
    echo ""
fi

if [ "$HAS_RUST" -eq 1 ]; then
    echo "=== Running fastembed-rs (Rust) benchmark ==="
    "$SCRIPT_DIR/bench_fastembed_rs/target/release/bench_fastembed_rs" "$SCRIPT_DIR/corpus.txt" | tee "$RESULTS_DIR/results_fastembed_rs.json"
    echo ""
fi

# ── Compare results ──────────────────────────────────────────────
echo "=== Comparison ==="
python3 "$SCRIPT_DIR/compare_results.py" "$RESULTS_DIR"
