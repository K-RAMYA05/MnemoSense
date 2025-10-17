import re, numpy as np
from typing import List
from embedder import embed_texts
def split_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]
def mmr_summarize(text: str, max_sentences: int = 4, diversity: float = 0.6) -> str:
    sents = split_sentences(text)
    if not sents: return text.strip()
    if len(sents) <= max_sentences: return " ".join(sents)
    embs = embed_texts(sents); centroid = np.mean(embs, axis=0); centroid/= (np.linalg.norm(centroid)+1e-8)
    selected=[int(np.argmax(embs @ centroid))]
    while len(selected)<max_sentences:
        best=-1e9; idx=None
        for i in range(len(sents)):
            if i in selected: continue
            rel=float(embs[i] @ centroid)
            red=max(float(embs[i] @ embs[j]) for j in selected) if selected else 0.0
            score=diversity*rel - (1-diversity)*red
            if score>best: best, idx = score, i
        if idx is None: break
        selected.append(idx)
    selected.sort(); return " ".join(sents[i] for i in selected)
