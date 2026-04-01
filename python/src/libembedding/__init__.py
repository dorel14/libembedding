"""libembedding — Fast ONNX-based text, image, and sparse embeddings for Python."""

from .text_embedding import TextEmbedding
from .sparse_text_embedding import SparseTextEmbedding
from .image_embedding import ImageEmbedding
from .reranker import Reranker
from .types import SparseEmbedding, RerankResult, ModelInfo
from .models import (
    list_text_models,
    list_sparse_models,
    list_image_models,
    list_reranker_models,
)
from .exceptions import LembedError

__version__ = "0.1.0"

__all__ = [
    "TextEmbedding",
    "SparseTextEmbedding",
    "ImageEmbedding",
    "Reranker",
    "SparseEmbedding",
    "RerankResult",
    "ModelInfo",
    "list_text_models",
    "list_sparse_models",
    "list_image_models",
    "list_reranker_models",
    "LembedError",
]
