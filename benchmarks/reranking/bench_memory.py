"""
Reranking memory benchmark - measures RSS per model.
"""
import argparse
import sys
import os
import subprocess
import re

sys.path.insert(0, os.path.dirname(__file__))
from libembedding import Reranker
from bench_common import generate_documents, benchmark_rerank


def get_rss_mb_tasklist():
    """Get RSS via tasklist (fallback when API fails)."""
    try:
        pid = os.getpid()
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse CSV: "name","pid","session","mem","status"
            match = re.search(r'"(\d[\d\s]*)\s*Ko"', result.stdout)
            if match:
                kb = int(match.group(1).replace(' ', ''))
                return kb / 1024
    except Exception:
        pass
    return 0


def get_rss_mb_wmic():
    """Get RSS via wmic."""
    try:
        pid = os.getpid()
        result = subprocess.run(
            ['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'WorkingSetSize'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.isdigit():
                    return int(line) / (1024 * 1024)
    except Exception:
        pass
    return 0


def get_rss_mb():
    """Get current process RSS in MB."""
    return get_rss_mb_wmic() or get_rss_mb_tasklist()


def main():
    parser = argparse.ArgumentParser(description='Reranking memory benchmark')
    parser.add_argument('--models', default='BAAI/bge-reranker-base')
    parser.add_argument('--docs', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    query = "What is deep learning?"
    docs = generate_documents(args.docs)

    print("=" * 70)
    print("Reranking Memory Benchmark")
    print("=" * 70)
    print(f"Docs: {args.docs}, Threads: {args.threads}")
    print()

    print(f"{'Model':<40} {'RSS load (MB)':>15} {'RSS inference (MB)':>20} {'Delta (MB)':>12}")
    print("-" * 90)

    results = []
    for model_name in model_names:
        try:
            rss_before = get_rss_mb()

            reranker = Reranker(
                model_name,
                threads=args.threads,
                offline=args.offline,
                show_download_progress=False,
            )
            rss_after_load = get_rss_mb()

            reranker.rerank(query, docs)
            rss_after_inference = get_rss_mb()

            delta = rss_after_load - rss_before
            print(f"{model_name:<40} {rss_after_load:>15.1f} {rss_after_inference:>20.1f} {delta:>12.1f}")

            results.append({
                'model': model_name,
                'rss_load': rss_after_load,
                'rss_inference': rss_after_inference,
                'delta': delta,
            })
            reranker.close()
        except Exception as e:
            print(f"{model_name:<40} SKIP: {e}")
            try:
                reranker.close()
            except Exception:
                pass
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| Modèle | RAM après load (MB) | RAM pendant inference (MB) |")
    print("|--------|---------------------|---------------------------|")
    for r in results:
        print(f"| {r['model']} | {r['rss_load']:.0f} | {r['rss_inference']:.0f} |")


if __name__ == '__main__':
    main()
