"""Corpus sampling utilities for autotune benchmarking."""

from __future__ import annotations

import math
import random


def _sample_corpus(texts: list[str], max_size: int = 100) -> list[str]:
    """Sample a representative subset of texts for autotune benchmarking.

    Uses stratified sampling by text length to ensure the sample includes
    a mix of short, medium, and long texts.

    Args:
        texts: Full corpus (can be millions of texts)
        max_size: Maximum number of texts to sample (default 100)

    Returns:
        Sampled texts representative of the corpus distribution
    """
    n = len(texts)
    if n <= max_size:
        return texts

    n_buckets = 10
    buckets: list[list[str]] = [[] for _ in range(n_buckets)]

    for text in texts:
        word_count = len(text.split())
        if word_count <= 0:
            bucket_idx = 0
        else:
            log_count = math.log10(word_count)
            bucket_idx = min(int(log_count * 2.5), n_buckets - 1)
        buckets[bucket_idx].append(text)

    sampled = []
    sampled_set = set()
    per_bucket = max_size // n_buckets

    for bucket in buckets:
        if not bucket:
            continue
        if len(bucket) <= per_bucket:
            for t in bucket:
                if t not in sampled_set:
                    sampled.append(t)
                    sampled_set.add(t)
        else:
            step = len(bucket) / per_bucket
            for i in range(per_bucket):
                idx = int(i * step)
                t = bucket[idx]
                if t not in sampled_set:
                    sampled.append(t)
                    sampled_set.add(t)

    if len(sampled) < max_size:
        remaining = [t for t in texts if t not in sampled_set]
        if remaining:
            sampled.extend(random.sample(remaining, min(max_size - len(sampled), len(remaining))))

    return sampled[:max_size]
