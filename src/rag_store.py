# src/rag_store.py
import json, os
import numpy as np
from utils import DB, now_iso, safe_mkdir
from embedder import embed_texts

META = DB / "meta.json"
EMBS = DB / "embeddings.npy"

def _load_meta():
    # Ensure DB dir exists even on first run
    safe_mkdir(DB)
    if not os.path.exists(META):
        return {"docs": []}
    return json.load(open(META, "r"))

def _save_meta(meta):
    safe_mkdir(DB)
    with open(META, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _load_embs():
    safe_mkdir(DB)
    if not os.path.exists(EMBS):
        return np.zeros((0, 384), dtype="float32")
    return np.load(EMBS)

def _save_embs(arr):
    safe_mkdir(DB)
    np.save(EMBS, arr)

def add_documents(docs):
    """
    docs: list of {"text": str, "ts": iso8601?, "tags": [..], "source": str}
    Returns: list of assigned ids.
    """
    meta = _load_meta()
    embs = _load_embs()

    texts = [d["text"] for d in docs]
    vecs = embed_texts(texts)  # normalized float32

    start_id = len(meta["docs"])
    ids = list(range(start_id, start_id + len(docs)))
    for i, d in enumerate(docs):
        meta["docs"].append({
            "id": ids[i],
            "ts": d.get("ts") or now_iso(),
            "text": d["text"],
            "tags": d.get("tags", []),
            "source": d.get("source", ""),
        })

    embs = np.vstack([embs, vecs]) if embs.size else vecs
    _save_meta(meta)
    _save_embs(embs)
    return ids

def search(query, top_k=6):
    meta = _load_meta()
    embs = _load_embs()
    if embs.size == 0:
        return []
    q = embed_texts([query])[0]  # normalized
    sims = embs @ q              # cosine because normalized
    idxs = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), meta["docs"][i]) for i in idxs]

def delete_ids(ids):
    """
    Remove a set/list of document ids from both meta and embeddings.
    Reassigns ids sequentially to keep arrays compact.
    """
    ids = set(ids)
    meta = _load_meta()
    embs = _load_embs()

    keep_idx = [i for i, d in enumerate(meta["docs"]) if d["id"] not in ids]
    new_docs = [meta["docs"][i] for i in keep_idx]
    new_embs = embs[keep_idx] if embs.size else embs

    # Reassign ids 0..N-1
    for i, d in enumerate(new_docs):
        d["id"] = i

    _save_meta({"docs": new_docs})
    _save_embs(new_embs)
