# src/rag_store.py
from __future__ import annotations
from typing import List, Dict, Iterable
import threading
import numpy as np
from copy import deepcopy

from embedder import embed_texts  # must return (N, D) numpy array for N texts

# ---------------- In-memory vector store ----------------
_lock = threading.Lock()
_embs: np.ndarray | None = None   # shape (N, D)
_docs: List[Dict] = []            # parallel metadata list (len == N)


def _sanitize_docs(docs: List[Dict]) -> List[Dict]:
    """Keep only docs with non-empty 'text'."""
    cleaned = []
    for d in docs:
        t = (d.get("text") or "").strip()
        if t:
            cleaned.append({"text": t, "meta": d.get("meta", {})})
    return cleaned


def add_documents(docs: List[Dict]) -> List[int]:
    """
    Add documents to the vector store.
    Each doc is {"text": str, "meta": {...}}.
    Returns the list of assigned integer IDs.
    Robust against empty inputs and empty embedding batches.
    """
    global _embs, _docs

    docs = _sanitize_docs(docs)
    if not docs:
        return []

    texts = [d["text"] for d in docs]
    vecs = embed_texts(texts)  # expected shape (N, D)

    if vecs is None:
        return []
    if isinstance(vecs, list):
        vecs = np.array(vecs)
    if not hasattr(vecs, "shape") or vecs.size == 0:
        return []
    vecs = np.atleast_2d(vecs)

    with _lock:
        start_id = len(_docs)
        new_ids = list(range(start_id, start_id + vecs.shape[0]))

        if _embs is None or (_embs.size == 0):
            _embs = vecs
        else:
            # dimension guard
            if _embs.shape[1] != vecs.shape[1]:
                # Embedding dimension changed (e.g., different model) -> skip safely
                return []
            _embs = np.vstack([_embs, vecs])

        _docs.extend(docs)
        return new_ids


# ----------------- Query helpers -----------------
def search(query: str, k: int = 5) -> List[Dict]:
    """
    Simple cosine-similarity search over in-memory embeddings.
    Returns list of {"text": str, "meta": {...}, "score": float, "id": int}
    """
    global _embs, _docs
    if not query or _embs is None or len(_docs) == 0:
        return []

    qv = embed_texts([query])
    if qv is None or getattr(qv, "size", 0) == 0:
        return []
    if isinstance(qv, list):
        qv = np.array(qv)
    qv = np.atleast_2d(qv)[0]

    A = _embs
    denom = (np.linalg.norm(A, axis=1) * (np.linalg.norm(qv) + 1e-8)) + 1e-8
    sims = (A @ qv) / denom

    idxs = np.argsort(-sims)[:k]
    out = []
    for i in idxs:
        out.append({
            "id": int(i),
            "text": _docs[i]["text"],
            "meta": deepcopy(_docs[i].get("meta", {})),
            "score": float(sims[i]),
        })
    return out


def count() -> int:
    """Number of vectors currently stored."""
    return 0 if _embs is None else int(_embs.shape[0])


# ----------------- Introspection / For forgetting.py -----------------
def _load_meta() -> List[Dict]:
    """
    Return a shallow copy of document metadata, aligned with indices [0..N-1].
    Each item resembles {"text": str, "meta": {...}}.
    NOTE: This matches your forgetting.py import.
    """
    with _lock:
        return deepcopy(_docs)


def get_embedding_dim() -> int:
    """Return current embedding dimension (D) or 0 if uninitialized."""
    if _embs is None or _embs.size == 0:
        return 0
    return int(_embs.shape[1])


# ----------------- Delete support (for forgetting) -----------------
def delete_ids(ids: Iterable[int]) -> int:
    """
    Delete documents by integer ids (indices). Returns number of rows removed.
    This rebuilds the in-memory arrays and reindexes to 0..N-1.
    NOTE: Any external references to old ids become invalid after this call.
    """
    global _embs, _docs
    ids = sorted(set(int(i) for i in ids if isinstance(i, (int, np.integer)) and i >= 0))
    if not ids:
        return 0

    with _lock:
        n = 0 if _embs is None else _embs.shape[0]
        if n == 0 or len(_docs) == 0:
            return 0

        mask = np.ones(n, dtype=bool)
        for i in ids:
            if 0 <= i < n:
                mask[i] = False

        kept_idx = np.where(mask)[0]
        removed = n - kept_idx.size

        if kept_idx.size == 0:
            # store becomes empty
            _embs = None
            _docs = []
            return removed

        _embs = _embs[kept_idx, :]
        _docs = [_docs[i] for i in kept_idx.tolist()]
        return removed
