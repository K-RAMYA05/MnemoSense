import gradio as gr
import time, os, uuid, datetime as dt
from stt import transcribe_file
from summarize import mmr_summarize
from rag_store import add_text, search
from llm import answer

def ingest(audio_path: str, notes: str):
    if not audio_path:
        return "No audio provided.", ""
    t0 = time.time()
    text = transcribe_file(audio_path) or ""
    if not text.strip():
        return "Couldn't transcribe. Try speaking closer to the mic.", ""
    summary = mmr_summarize(text, max_sentences=4)
    meta = {
        "id": str(uuid.uuid4()),
        "ts": dt.datetime.utcnow().isoformat(),
        "tags": [t.strip() for t in (notes or "").split(",") if t.strip()]
    }
    add_text(summary, meta)
    dt_ms = int((time.time()-t0)*1000)
    return f"Indexed summary in {dt_ms} ms (text-only).", summary

def ask(q: str, audio_q: str):
    query = (q or "").strip()
    if (not query) and audio_q:
        query = transcribe_file(audio_q)
    if not query.strip():
        return "", "", "Please provide a question (text or audio)."
    hits = search(query, k=5)
    ctxs = [h.get("text","") for h in hits]
    ans = answer(query, ctxs)
    refs = "\n\n".join([f"- {h.get('text','')[:160]}…" for h in hits])
    return query, ans, refs if refs else "(no references yet)"

with gr.Blocks(title="MnemoSense — Spaces Demo") as demo:
    gr.Markdown("# MnemoSense — Text-only Memory (HF Spaces)")
    gr.Markdown("**Privacy-first**: Only summaries are stored. Try the **Ingest** tab, then ask questions.")

    with gr.Tab("Ingest"):
        with gr.Row():
            mic = gr.Audio(sources=["microphone","upload"], type="filepath", label="Record or Upload (<= 60s)")
            notes = gr.Textbox(label="Optional tags (comma-separated)", placeholder="demo, meeting, idea")
        btn_ingest = gr.Button("Transcribe → Summarize → Index")
        status = gr.Textbox(label="Status", interactive=False)
        summary = gr.Textbox(label="Summary stored", lines=4, interactive=False)
        btn_ingest.click(ingest, inputs=[mic, notes], outputs=[status, summary])

    with gr.Tab("Ask"):
        with gr.Row():
            q = gr.Textbox(label="Question", placeholder="What did we say about the mission?")
            q_audio = gr.Audio(sources=["microphone","upload"], type="filepath", label="Or ask by voice")
        btn_ask = gr.Button("Retrieve → Answer")
        out_q = gr.Textbox(label="You asked", interactive=False)
        out_ans = gr.Textbox(label="Answer", lines=6, interactive=False)
        out_refs = gr.Textbox(label="References (summaries)", lines=6, interactive=False)
        btn_ask.click(ask, inputs=[q, q_audio], outputs=[out_q, out_ans, out_refs])

if __name__ == "__main__":
    demo.launch()
