"""libembedding — Fast ONNX-based text, image, and sparse embeddings for Python."""

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("libembedding-ng")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .text_embedding import TextEmbedding, TextEmbeddingPool, autotune, auto_select_model, clear_autotune_cache, autotune_unified
from .sparse_text_embedding import SparseTextEmbedding, sparse_autotune
from .image_embedding import ImageEmbedding, image_autotune
from .reranker import Reranker, reranker_autotune, reranker_auto_config, clear_reranker_autotune_cache, reranker_auto_config_profile, reranker_autotune_constrained
from .types import SparseEmbedding, RerankResult, ModelInfo, ModelDesc, Stats, TuningResult, RerankerTuningResult, SparseTuningResult, ImageTuningResult, UnifiedTuningResult, ModelSelectionResult
from .models import (
    list_text_models,
    list_sparse_models,
    list_image_models,
    list_reranker_models,
)
from .similarity import cosine_similarity, dot_product, euclidean_distance
from .exceptions import LembedError

# Version from C API (runtime)
try:
    from ._binding import lib
    __version__ = lib.lembed_version().decode("utf-8", errors="replace")
except Exception:
    pass  # fall back to importlib.metadata version above

# Autotune modes (re-exported for convenience)
LEMBED_AUTOTUNE_QUICK = lib.LEMBED_AUTOTUNE_QUICK
LEMBED_AUTOTUNE_FULL = lib.LEMBED_AUTOTUNE_FULL

# Optimization objectives
LEMBED_OBJECTIVE_LATENCY = lib.LEMBED_OBJECTIVE_LATENCY
LEMBED_OBJECTIVE_THROUGHPUT = lib.LEMBED_OBJECTIVE_THROUGHPUT
LEMBED_OBJECTIVE_BALANCED = lib.LEMBED_OBJECTIVE_BALANCED
LEMBED_OBJECTIVE_MEMORY = lib.LEMBED_OBJECTIVE_MEMORY

# Task types for unified auto-tuner
LEMBED_TASK_EMBEDDING = lib.LEMBED_TASK_EMBEDDING
LEMBED_TASK_RERANKING = lib.LEMBED_TASK_RERANKING
LEMBED_TASK_IMAGE = lib.LEMBED_TASK_IMAGE
LEMBED_TASK_SPARSE = lib.LEMBED_TASK_SPARSE

# Reranker profiles
LEMBED_PROFILE_INTERACTIVE = lib.LEMBED_PROFILE_INTERACTIVE
LEMBED_PROFILE_BALANCED = lib.LEMBED_PROFILE_BALANCED
LEMBED_PROFILE_QUALITY = lib.LEMBED_PROFILE_QUALITY

__all__ = [
    "TextEmbedding",
    "TextEmbeddingPool",
    "autotune",
    "auto_select_model",
    "clear_autotune_cache",
    "SparseTextEmbedding",
    "sparse_autotune",
    "ImageEmbedding",
    "image_autotune",
    "Reranker",
    "SparseEmbedding",
    "RerankResult",
    "ModelInfo",
    "ModelDesc",
    "Stats",
    "TuningResult",
    "RerankerTuningResult",
    "SparseTuningResult",
    "ImageTuningResult",
    "UnifiedTuningResult",
    "autotune_unified",
    "reranker_autotune",
    "reranker_autotune_constrained",
    "reranker_auto_config",
    "reranker_auto_config_profile",
    "clear_reranker_autotune_cache",
    "list_text_models",
    "list_sparse_models",
    "list_image_models",
    "list_reranker_models",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "LembedError",
    # Autotune modes
    "LEMBED_AUTOTUNE_QUICK",
    "LEMBED_AUTOTUNE_FULL",
    # Objectives
    "LEMBED_OBJECTIVE_LATENCY",
    "LEMBED_OBJECTIVE_THROUGHPUT",
    "LEMBED_OBJECTIVE_BALANCED",
    "LEMBED_OBJECTIVE_MEMORY",
    # Tasks
    "LEMBED_TASK_EMBEDDING",
    "LEMBED_TASK_RERANKING",
    "LEMBED_TASK_IMAGE",
    "LEMBED_TASK_SPARSE",
    # Profiles
    "LEMBED_PROFILE_INTERACTIVE",
    "LEMBED_PROFILE_BALANCED",
    "LEMBED_PROFILE_QUALITY",
]
