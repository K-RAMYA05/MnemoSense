import os, subprocess

def _to_wav16k(path: str) -> str:
    if path.endswith(".wav"): return path
    wav = path + ".wav"
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", wav]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return wav if os.path.exists(wav) and os.path.getsize(wav) > 0 else path

def transcribe_file(path: str, model_size="base") -> str:
    # Prefer OpenAI Whisper API (fast on Spaces CPU)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            wav = _to_wav16k(path)
            with open(wav, "rb") as f:
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
            return (tr.text or "").strip()
        except Exception:
            pass
    # Fallback: local whisper (may be slow on CPU)
    import whisper
    wav = _to_wav16k(path)
    model = whisper.load_model(model_size)
    out = model.transcribe(wav, fp16=False)
    return (out or {}).get("text","").strip()
