"""Data types for libembedding results."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SparseEmbedding:
    """Sparse embedding with token indices and weights."""

    indices: np.ndarray  # int32
    values: np.ndarray  # float32


@dataclass(frozen=True)
class RerankResult:
    """Single document reranking result."""

    index: int
    score: float


@dataclass(frozen=True)
class ModelInfo:
    """Information about an available model."""

    model_name: str
    model_code: str
    model_file: str
    description: str
    dim: int
    max_tokens: int
    pooling: str  # "cls" or "mean"
    quantization: str  # "none", "static", "dynamic"


@dataclass(frozen=True)
class ModelDesc:
    """Runtime descriptor of a created model context."""

    name: str
    dimension: int
    max_length: int
    pooling: str  # "cls" or "mean"
    num_threads: int
    batch_size: int
    provider: str
    device_id: int
