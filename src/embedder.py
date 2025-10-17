# src/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
    return _model

def embed_texts(texts):
    if not texts: return np.zeros((0, 384), dtype="float32")
    m = _get_model()
    embs = m.encode(texts, normalize_embeddings=False)
    return np.asarray(embs, dtype="float32")
