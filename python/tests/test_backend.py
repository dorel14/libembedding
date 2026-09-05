"""Test backend auto-detection."""
from libembedding import TextEmbedding, detect_backend

# Test auto-detection
print("Auto-detection tests:")
print(f"  bge-small -> {detect_backend('BAAI/bge-small-en-v1.5')}")
print(f"  model.gguf -> {detect_backend('model.gguf')}")
print(f"  model.onnx -> {detect_backend('model.onnx')}")
print(f"  Xenova/model-Q4_K_M.gguf -> {detect_backend('Xenova/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2-Q4_K_M.gguf')}")
print(f"  Explicit onnx -> {detect_backend('anything', 'onnx')}")
print(f"  Explicit llama -> {detect_backend('anything', 'llama')}")
