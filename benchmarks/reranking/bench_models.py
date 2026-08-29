"""
Reranking model comparison benchmark - compares MiniLM vs BGE vs Jina rerankers.
Tests if the bottleneck is the model itself or the code.
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))

from libembedding import Reranker, list_reranker_models


def generate_documents(n, seed=42):
    """Generate n synthetic documents."""
    import random
    random.seed(seed)
    base_docs = [
        "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The Python programming language was created by Guido van Rossum in 1991.",
        "Quantum computing leverages quantum mechanical phenomena to perform computation.",
        "The Great Wall of China is a series of fortifications built over centuries.",
        "Natural language processing enables computers to understand human language.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
    ]
    docs = []
    for i in range(n):
        doc = base_docs[i % len(base_docs)]
        repeat = (i % 3) + 1
        docs.append((doc + " ") * repeat)
    return docs


def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def main():
    parser = argparse.ArgumentParser(description='Reranking model comparison')
    parser.add_argument('--models', default='',
                        help='Comma-separated model names (empty = all)')
    parser.add_argument('--docs', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    query = "What is deep learning?"
    docs = generate_documents(args.docs)

    # Get available models
    available = list_reranker_models()
    available_names = [m.model_name for m in available]

    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
    else:
        model_names = available_names

    print("=" * 70)
    print("Reranking Model Comparison Benchmark")
    print("=" * 70)
    print(f"Docs: {args.docs}, Threads: {args.threads}, Batch: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()
    print("Objectif: comparer les modèles pour voir si le bottleneck est le modèle ou le code.")
    print()

    print("Available models:")
    for m in available:
        print(f"  - {m.model_name} ({m.description})")
    print()

    print("=" * 70)
    print(f"{'Model':<45} | {'ms/doc':>8} | {'docs/s':>8} | {'Status':>10}")
    print("-" * 80)

    results = []
    for model_name in model_names:
        try:
            # Load model
            t0 = time.perf_counter()
            reranker = Reranker(
                model_name,
                threads=args.threads,
                batch_size=args.batch_size,
                offline=args.offline,
                show_download_progress=False,
            )
            load_time = (time.perf_counter() - t0) * 1000

            # Warmup
            for _ in range(args.warmup):
                reranker.rerank(query, docs, batch_size=args.batch_size)

            # Benchmark
            times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                reranker.rerank(query, docs, batch_size=args.batch_size)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            med_ms = median(times)
            ms_per_doc = med_ms / args.docs
            docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0

            # Get model info
            info = reranker.info()

            print(f"{model_name:<45} | {ms_per_doc:>8.1f} | {docs_per_sec:>8.1f} | {'OK':>10}")

            results.append({
                'model': model_name,
                'ms_per_doc': ms_per_doc,
                'docs_per_sec': docs_per_sec,
                'load_ms': load_time,
                'max_length': info.max_length,
                'status': 'OK',
            })
            reranker.close()

        except Exception as e:
            print(f"{model_name:<45} | {'---':>8} | {'---':>8} | {'SKIP':>10}")
            print(f"  Reason: {e}")
            results.append({
                'model': model_name,
                'ms_per_doc': None,
                'docs_per_sec': None,
                'load_ms': None,
                'max_length': None,
                'status': f'SKIP: {e}',
            })

    print()

    # Summary
    ok_results = [r for r in results if r['status'] == 'OK']
    if len(ok_results) >= 2:
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print()
        baseline = ok_results[0]
        for r in ok_results[1:]:
            speedup = r['ms_per_doc'] / baseline['ms_per_doc']
            print(f"  {baseline['model']} vs {r['model']}:")
            print(f"    {baseline['ms_per_doc']:.1f} vs {r['ms_per_doc']:.1f} ms/doc")
            print(f"    Ratio: {speedup:.2f}x")
            print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| Modèle | ms/doc | docs/s | Load (ms) | Max tokens |")
    print("|--------|--------|--------|----------|------------|")
    for r in results:
        if r['status'] == 'OK':
            print(f"| {r['model']} | {r['ms_per_doc']:.1f} | {r['docs_per_sec']:.1f} | {r['load_ms']:.0f} | {r['max_length']} |")
        else:
            print(f"| {r['model']} | --- | --- | --- | --- |")


if __name__ == '__main__':
    main()
