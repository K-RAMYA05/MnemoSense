# src/vector_db.py
from __future__ import annotations
from typing import List, Dict, Tuple
import os, json
from pathlib import Path
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except ImportError:
    faiss = None

from .embedder import embed_texts


# Files
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DBD  = DATA / "vectordb"
INDEX_PATH = DBD / "index.faiss"
META_PATH  = DBD / "meta.jsonl"

os.makedirs(DBD, exist_ok=True)

# Globals
_index = None
_dim   = None
_meta: List[Dict] = []

def _load_meta():
    global _meta
    _meta = []
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    _meta.append(json.loads(line))

def _save_meta(entries: List[Dict]):
    with open(META_PATH, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def _ensure_index(dim: int):
    global _index, _dim
    if faiss is None:
        raise RuntimeError("faiss-cpu not installed. pip install faiss-cpu")
    if _index is not None and _dim == dim:
        return
    if INDEX_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        _dim = _index.d   # dimension
    else:
        _index = faiss.IndexFlatIP(dim)  # cosine if normalized
        _dim = dim

def _persist_index():
    faiss.write_index(_index, str(INDEX_PATH))

def add_texts(texts: List[str], metas: List[Dict] | None = None) -> List[int]:
    """
    Embed and add to FAISS. Returns assigned ids (row positions in meta list).
    """
    if not texts:
        return []
    embs = embed_texts(texts)  # (N, D)
    if isinstance(embs, list):
        embs = np.array(embs)
    if embs is None or getattr(embs, "size", 0) == 0:
        return []
    embs = np.array(embs, dtype="float32")
    # L2-normalize to turn inner product into cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms

    _ensure_index(embs.shape[1])
    _index.add(embs)

    # create metadata rows
    base = len(_meta)
    if metas is None:
        metas = [{} for _ in texts]
    for t, m in zip(texts, metas):
        _meta.append({"text": t, "meta": m})
    _save_meta([{"text": t, "meta": m} for t, m in zip(texts, metas)])

    _persist_index()
    return list(range(base, base + len(texts)))

def search(query: str, k: int = 5) -> List[Dict]:
    if not query.strip():
        return []
    # lazy load meta & index
    if not _meta:
        _load_meta()
    # embed query
    q = embed_texts([query])
    if isinstance(q, list):
        q = np.array(q)
    if q is None or getattr(q, "size", 0) == 0:
        return []
    q = q.astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

    _ensure_index(q.shape[1])
    if _index.ntotal == 0:
        return []

    D, I = _index.search(q, min(k, _index.ntotal))  # inner product ~ cosine
    out=[]
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(_meta): continue
        row = _meta[idx]
        out.append({"id": int(idx), "text": row["text"], "meta": row.get("meta", {}), "score": float(score)})
    return out
