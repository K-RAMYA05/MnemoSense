from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import os, tempfile

try:
    from .tts import speak
except Exception:
    def speak(_: str): pass

from .stt import transcribe
from .vector_db import search
from .llm import generate_answer


TEMPLATE = """
<!doctype html><html><head><meta charset='utf-8'><title>MnemoSense</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:2rem;max-width:820px}
.card{border:1px solid #ddd;border-radius:14px;padding:1rem 1.25rem;margin-bottom:1rem;box-shadow:0 2px 6px rgba(0,0,0,.04)}
input[type=text]{width:100%;padding:.7rem;border-radius:10px;border:1px solid #aaa}
button{padding:.6rem 1rem;border-radius:10px;border:0;background:#111;color:#fff;cursor:pointer}
.muted{color:#666;font-size:.9rem}.row{display:grid;grid-template-columns:1fr auto auto;gap:.5rem;align-items:center}
#micBtn.recording{background:#ff3b30}
</style></head><body>
<h1>MnemoSense</h1>
<div class="card">
  <form method="POST" action="{{ url_for('ask') }}">
    <div class="row">
      <input id="q" type="text" name="q" placeholder="Ask anything remembered…" required>
      <button>Ask</button>
      <button type="button" id="micBtn" title="Hold to talk">🎙️ Hold</button>
    </div>
  </form>
  <p class="muted" id="status">Idle.</p>
  <div class="card">
    <div class="muted">You said</div>
    <div id="heard">—</div>
  </div>
  {% if answer %}<div class="card"><div class="muted">Answer</div><div id="answer">{{ answer }}</div></div>{% else %}
  <div class="card"><div class="muted">Answer</div><div id="answer">—</div></div>{% endif %}
</div>
<script>
(() => {
  const micBtn = document.getElementById('micBtn');
  const statusEl = document.getElementById('status');
  const heardEl  = document.getElementById('heard');
  const answerEl = document.getElementById('answer');
  let mediaRecorder, stream, chunks=[];

  function speakClient(text) {
    try { const u=new SpeechSynthesisUtterance(text); speechSynthesis.cancel(); speechSynthesis.speak(u); } catch(e){}
  }

  async function startRec() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio:true});
      const mime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' :
                   MediaRecorder.isTypeSupported('audio/ogg')  ? 'audio/ogg'  : '';
      mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      chunks = [];
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          const form = new FormData(); form.append('audio', blob, 'query.webm');
          statusEl.textContent = 'Transcribing → retrieving → LLM…';
          const res = await fetch('/api/ask/voice', { method:'POST', body: form });
          const out = await res.json();
          heardEl.textContent = out.text || '—';
          answerEl.textContent = out.answer || '—';
          statusEl.textContent = out.error ? ('Error: ' + out.error) : 'Done.';
          if (out.answer) speakClient(out.answer);
        } catch (e) {
          statusEl.textContent = 'Error (see console).'; console.error(e);
        } finally {
          if (stream) stream.getTracks().forEach(t => t.stop());
        }
      };
      mediaRecorder.start(); micBtn.classList.add('recording'); micBtn.textContent='● Recording…';
      statusEl.textContent='Listening…';
    } catch (e) { statusEl.textContent='Mic permission denied?'; }
  }
  function stopRec() {
    if (mediaRecorder && mediaRecorder.state!=='inactive') mediaRecorder.stop();
    micBtn.classList.remove('recording'); micBtn.textContent='🎙️ Hold';
  }
  micBtn.addEventListener('mousedown', startRec);
  micBtn.addEventListener('mouseup', stopRec);
  micBtn.addEventListener('mouseleave', ()=>{ if (mediaRecorder?.state==='recording') stopRec(); });
  micBtn.addEventListener('touchstart', (e)=>{ e.preventDefault(); startRec(); }, {passive:false});
  micBtn.addEventListener('touchend',   (e)=>{ e.preventDefault(); stopRec();  }, {passive:false});
})();
</script>
</body></html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(TEMPLATE, answer=None)

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("q","").strip()
    if not q:
        return redirect(url_for("home"))

    hits = search(q, k=6)
    contexts = [h["text"] for h in hits]
    ans = generate_answer(q, contexts)

    try: speak(ans)  # optional server-side TTS
    except Exception: pass

    return render_template_string(TEMPLATE, answer=ans)

@app.route("/api/ask/voice", methods=["POST"])
def ask_voice():
    f = request.files.get("audio")
    if not f: return jsonify({"error":"no audio"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tmp.write(f.read()); tmp.flush(); tmp.close()

    try:
        text, _ = transcribe(tmp.name)  # whisper can handle webm via ffmpeg
    except Exception as e:
        try: os.unlink(tmp.name)
        except: pass
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass

    q = (text or "").strip()
    if not q:
        return jsonify({"text":"", "answer":"Sorry, I didn’t catch that. Try again."})

    hits = search(q, k=6)
    contexts = [h["text"] for h in hits]
    ans = generate_answer(q, contexts)

    try: speak(ans)
    except Exception: pass

    return jsonify({"text": q, "answer": ans})

def run(port=5005):
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    run(5005)
