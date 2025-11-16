from typing import List
_model = None

def _ensure_model():
    global _model
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        # ultra-light fallback if sentence-transformers can't load
        import numpy as np
        class _HashEmb:
            def encode(self, texts, normalize_embeddings=True):
                out = []
                for t in texts:
                    h = abs(hash(t)) % (10**8)
                    vec = np.array([(h >> i) & 1 for i in range(256)], dtype=float)
                    if normalize_embeddings:
                        n = np.linalg.norm(vec) + 1e-8
                        vec = vec / n
                    out.append(vec)
                return out
        _model = _HashEmb()

def embed_texts(texts: List[str]):
    if isinstance(texts, str):
        texts = [texts]
    _ensure_model()
    return _model.encode(texts, normalize_embeddings=True)
