import os, requests
from rag_store import search
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
SYS_PROMPT = """You are MnemoSense, a helpful assistant.
Answer ONLY from the provided notes. If it's not in the notes, say you don't have that information.
Be concise. Do not cite timestamps or sources."""
def _ollama_chat(messages, model=LLM_MODEL, temperature=0.1, max_tokens=300):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model, "messages": messages, "options": {"temperature": temperature, "num_predict": max_tokens}
        }, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()
    except Exception:
        return None
    return None
def answer(query, top_k=6):
    hits = search(query, top_k=top_k)
    context = "\n".join([f"- {d['text']}" for _, d in hits])
    if context.strip():
        resp = _ollama_chat(
            [{"role":"system","content":SYS_PROMPT},
             {"role":"user","content": f"Question: {query}\n\nNotes:\n{context}\n\nAnswer clearly."}])
        if resp: return resp, hits
    return (hits[0][1]["text"] if hits else "I don't have that yet."), hits
