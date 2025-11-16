import os
from typing import List
from summarize import mmr_summarize

def _join(ctxs: List[str], max_chars=4000) -> str:
    out, used = [], 0
    for c in (ctxs or []):
        c = (c or "").strip()
        if not c: continue
        if used + len(c) > max_chars: break
        out.append(c); used += len(c)
    return "\n\n".join(out) if out else "(no context)"

def local_answer(question: str, contexts: List[str]) -> str:
    ctx = _join(contexts, 3000)
    if not ctx or ctx == "(no context)":
        return "I don't have enough information yet."
    return mmr_summarize(ctx, max_sentences=4)

def openai_answer(question: str, contexts: List[str]) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL","gpt-4o-mini")
        system = "You are MnemoSense. Answer using ONLY the provided context. If insufficient, say you don't know."
        user = f"Context:\n{_join(contexts)}\n\nQuestion: {question}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "(local) " + local_answer(question, contexts)

def answer(question: str, contexts: List[str]) -> str:
    if os.getenv("OPENAI_API_KEY"):
        return openai_answer(question, contexts)
    return local_answer(question, contexts)
