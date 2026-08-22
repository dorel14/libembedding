"""libembedding — Fast ONNX-based text, image, and sparse embeddings for Python."""

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("libembedding-ng")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .text_embedding import TextEmbedding
from .sparse_text_embedding import SparseTextEmbedding
from .image_embedding import ImageEmbedding
from .reranker import Reranker
from .types import SparseEmbedding, RerankResult, ModelInfo, ModelDesc
from .models import (
    list_text_models,
    list_sparse_models,
    list_image_models,
    list_reranker_models,
)
from .exceptions import LembedError

__all__ = [
    "TextEmbedding",
    "SparseTextEmbedding",
    "ImageEmbedding",
    "Reranker",
    "SparseEmbedding",
    "RerankResult",
    "ModelInfo",
    "ModelDesc",
    "list_text_models",
    "list_sparse_models",
    "list_image_models",
    "list_reranker_models",
    "LembedError",
]
