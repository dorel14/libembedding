"""
libembedding — Python example
Demonstrates text embedding, similarity search, and model discovery.
"""

import numpy as np
from libembedding import TextEmbedding, list_text_models


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    # ── List available models ────────────────────────────────────
    print("Available text models:")
    for m in list_text_models()[:5]:
        print(f"  {m.model_name:45} dim={m.dim:<5} {m.pooling}")
    print(f"  ... and {len(list_text_models()) - 5} more\n")

    # ── Create embedder ──────────────────────────────────────────
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    print(f"Model loaded: dim={model.dim}\n")

    # ── Embed texts ──────────────────────────────────────────────
    texts = [
        "The cat sat on the mat",
        "A kitten was sitting on a rug",
        "Quantum physics describes subatomic particles",
        "Machine learning models learn from data",
        "Deep neural networks have many layers",
    ]

    embeddings = model.embed(texts)
    print(f"Embedded {len(texts)} texts -> shape {embeddings.shape}\n")

    # ── Pairwise similarity ──────────────────────────────────────
    print("Pairwise cosine similarities:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}  ({texts[i][:30]}... vs {texts[j][:30]}...)")
    print()

    # ── Simple semantic search ───────────────────────────────────
    query = "How do cats behave?"
    query_emb = model.embed([query])

    print(f'Semantic search for: "{query}"')
    scores = [(i, cosine_similarity(query_emb[0], embeddings[i])) for i in range(len(texts))]
    scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (idx, score) in enumerate(scores, 1):
        print(f"  #{rank} (score={score:.4f}): {texts[idx]}")

    model.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
