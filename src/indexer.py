import re
from utils import now_iso
from rag_store import add_documents
def _split_into_chunks(text, max_len=512):
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur)+len(s) < max_len: cur=(cur+" "+s).strip()
        else: 
            if cur: chunks.append(cur); cur=s
    if cur: chunks.append(cur)
    return chunks
def index_transcript(text, source, tags=None, max_chunk_len=512):
    chunks = _split_into_chunks(text, max_len=max_chunk_len)
    docs = [ {"text": c, "ts": now_iso(), "source": source, "tags": tags or []} for c in chunks if c.strip() ]
    ids = add_documents(docs); return ids, len(chunks)
