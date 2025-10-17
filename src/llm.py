# src/llm.py
import os
from typing import List

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def generate_answer(query: str, contexts: List[str]) -> str:
    """
    Simple OpenAI Chat call. Provide your OPENAI_API_KEY in env.
    Fallback: return stitched context if no key.
    """
    if not OPENAI_API_KEY:
        # graceful fallback: extractive
        joined = "\n\n".join(contexts[:3])
        return f"(No LLM key configured) Here's what I found:\n\n{joined}"

    import openai
    openai.api_key = OPENAI_API_KEY

    system = (
        "You are MnemoSense, a helpful memory assistant. "
        "Answer concisely using ONLY the provided context. "
        "If context is insufficient, say you don't have that information."
    )
    ctx_block = "\n\n".join([f"- {c}" for c in contexts[:6]])
    user = f"Query: {query}\n\nContext:\n{ctx_block}"

    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp["choices"][0]["message"]["content"].strip()
