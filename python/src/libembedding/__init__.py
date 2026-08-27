"""libembedding — Fast ONNX-based text, image, and sparse embeddings for Python."""

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("libembedding-ng")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .text_embedding import TextEmbedding, TextEmbeddingPool
from .sparse_text_embedding import SparseTextEmbedding
from .image_embedding import ImageEmbedding
from .reranker import Reranker
from .types import SparseEmbedding, RerankResult, ModelInfo, ModelDesc, Stats
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

__all__ = [
    "TextEmbedding",
    "TextEmbeddingPool",
    "SparseTextEmbedding",
    "ImageEmbedding",
    "Reranker",
    "SparseEmbedding",
    "RerankResult",
    "ModelInfo",
    "ModelDesc",
    "Stats",
    "list_text_models",
    "list_sparse_models",
    "list_image_models",
    "list_reranker_models",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "LembedError",
]
