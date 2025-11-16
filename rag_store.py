import os, json
from typing import List, Dict, Any
import numpy as np
from embedder import embed_texts

BASE = os.path.dirname(__file__)
DB_DIR = os.path.join(BASE, "data")
META = os.path.join(DB_DIR, "transcripts.jsonl")
VEC = os.path.join(DB_DIR, "vec.npy")

os.makedirs(DB_DIR, exist_ok=True)

def _load_meta() -> List[Dict[str, Any]]:
    if not os.path.exists(META): return []
    with open(META, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def _append_meta(row: Dict[str, Any]):
    with open(META, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")

def _load_vecs() -> np.ndarray:
    if not os.path.exists(VEC): return np.zeros((0,256), dtype=np.float32)
    return np.load(VEC)

def _save_vecs(X: np.ndarray):
    np.save(VEC, X)

def add_text(text: str, meta: Dict[str, Any]):
    X = _load_vecs()
    emb = embed_texts([text])[0]
    emb = np.array(emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb)+1e-8)
    X_new = emb[None, :] if X.size==0 else np.vstack([X, emb])
    _save_vecs(X_new)
    _append_meta(meta | {"text": text})

def search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    rows = _load_meta()
    if not rows: return []
    X = _load_vecs()
    qv = np.array(embed_texts([query])[0], dtype=np.float32)
    qv = qv / (np.linalg.norm(qv)+1e-8)
    sims = (X @ qv).tolist() if X.size else []
    idx = np.argsort(sims)[::-1][:k] if sims else []
    hits = []
    for i in idx:
        if i < len(rows):
            r = dict(rows[i])
            r["score"] = float(sims[i])
            hits.append(r)
    return hits
