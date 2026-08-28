# Reranking Benchmarks — Memory

## Objectif
Mesurer la consommation mémoire pour chaque modèle de reranker.

## Méthodologie

- Mesurer peak RSS après chargement modèle
- Mesurer peak RSS pendant inference
- Platform: macOS (ru_maxrss), Linux (/proc/self/status), Windows (GetProcessMemoryInfo)

## Résultats attendus

| Modèle | RAM après load (MB) | RAM pendant inference (MB) |
|--------|---------------------|---------------------------|
| MiniLM-L6-v2 | ? | ? |
| BGE-base | ? | ? |
| BGE-large | ? | ? |

## Commandes

```bash
cd benchmarks/reranking
python bench_memory.py --models MiniLM-L6-v2,BGE-base,BGE-large
```
