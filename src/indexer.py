from typing import List, Tuple
import re
from .vector_db import add_texts


def _split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text.strip()) if s.strip()]

def _chunk_text(text: str, max_chunk_len: int = 512) -> List[str]:
    if not text or not text.strip():
        return []
    sents = _split_into_sentences(text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + 1 + len(s) <= max_chunk_len:
            cur = f"{cur} {s}".strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]

def index_transcript(text: str, source: str = "", tags: List[str] | None = None, max_chunk_len: int = 512) -> Tuple[List[int], int]:
    tags = tags or []
    if not text or not text.strip():
        return [], 0
    chunks = _chunk_text(text, max_chunk_len=max_chunk_len)
    if not chunks:
        return [], 0
    metas = [{"source": source, "tags": tags, "len": len(c)} for c in chunks]
    ids = add_texts(chunks, metas)
    return ids, len(chunks)
