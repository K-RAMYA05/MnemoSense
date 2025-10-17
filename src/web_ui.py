from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import json, os, time
from utils import TRANS
from qa import answer
from tts import speak
from stt import transcribe
TEMPLATE = """
<!doctype html><html><head><meta charset='utf-8'><title>MnemoSense</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:2rem}
.card{border:1px solid #ddd;border-radius:14px;padding:1rem 1.25rem;margin-bottom:1rem;box-shadow:0 2px 6px rgba(0,0,0,.04)}
input[type=text]{width:100%;padding:.7rem;border-radius:10px;border:1px solid #aaa}
button{padding:.6rem 1rem;border-radius:10px;border:0;background:#111;color:#fff;cursor:pointer}
.muted{color:#666;font-size:.9rem}.row{display:grid;grid-template-columns:1fr auto auto;gap:.5rem;align-items:center}
</style></head><body>
<h1>MnemoSense</h1>
<div class="card">
  <form method="POST" action="{{ url_for('ask') }}">
    <div class="row">
      <input id="q" type="text" name="q" placeholder="Ask anything remembered…" required>
      <button>Ask</button>
      <button type="button" id="micBtn">🎙️</button>
    </div>
  </form>
  <p class="muted" id="recHint">Click 🎙️ to start/stop recording.</p>
  <div id="heard" class="muted"></div>
  {% if answer %}<p><strong>Answer:</strong> {{ answer }}</p>{% endif %}
</div>
<script>
let mediaRecorder; let chunks=[];
const micBtn=document.getElementById('micBtn'); const heard=document.getElementById('heard');
micBtn.addEventListener('click', async ()=>{
  if(!mediaRecorder || mediaRecorder.state==='inactive'){
    chunks=[]; const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    mediaRecorder=new MediaRecorder(stream);
    mediaRecorder.ondataavailable=e=>{ if(e.data.size>0) chunks.push(e.data); };
    mediaRecorder.onstop=async ()=>{
      const blob=new Blob(chunks,{type:mediaRecorder.mimeType||'audio/webm'});
      const form=new FormData(); form.append('audio', blob, 'query.webm');
      const res=await fetch('/api/ask/voice',{method:'POST', body:form}); const out=await res.json();
      heard.textContent=out.text?('[query] '+out.text):''; if(out.redirect) location.reload();
    };
    mediaRecorder.start(); micBtn.textContent='⏺️ recording...';
  } else if(mediaRecorder.state==='recording'){ mediaRecorder.stop(); micBtn.textContent='🎙️'; }
});
</script>
</body></html>
"""
app = Flask(__name__)
@app.route("/", methods=["GET"])
def home(): return render_template_string(TEMPLATE, answer=None)
@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("q","").strip()
    if not q: return redirect(url_for("home"))
    ans, _ = answer(q, top_k=6)
    try: speak(ans)
    except Exception: pass
    return render_template_string(TEMPLATE, answer=ans)
@app.route("/api/ask/voice", methods=["POST"])
def ask_voice():
    f = request.files.get('audio')
    if not f: return jsonify({"error":"no audio"}), 400
    tmp_dir = (TRANS.parent / "tmp"); os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"voice_{int(time.time())}.webm"); f.save(path)
    try: text, _ = transcribe(path)
    except Exception as e: return jsonify({"error": str(e)}), 500
    ans, _ = answer(text, top_k=6)
    try: speak(ans)
    except Exception: pass
    return jsonify({"text": text, "answer": ans, "redirect": True})
def run(port=5005): app.run(host="127.0.0.1", port=port, debug=False)
