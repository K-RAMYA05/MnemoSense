import numpy as np
from sentence_transformers import SentenceTransformer
_model=None
def _get_model():
    global _model
    if _model is None: _model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model
def embed_texts(texts):
    embs=_get_model().encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")
