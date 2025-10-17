import whisper, os, json
from .utils import TRANS, now_iso, safe_mkdir

# env hygiene
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

def transcribe(wav_path: str, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(wav_path, fp16=False)
    text = result.get("text", "").strip()
    segments = result.get("segments", [])
    return text, segments

def save_transcript(wav_path: str, text: str, segments):
    """
    Persist ONLY text (no audio/video paths). Also removes the temporary wav.
    """
    safe_mkdir(TRANS)
    rec = {
        "ts": now_iso(),
        "text": text,          # store summary or transcript text only
        "segments": [],        # keep empty to avoid leaking timing details if not needed
        "tags": []
    }
    # write into transcripts folder with neutral name (no wav reference)
    json_path = os.path.join(str(TRANS), f"{now_iso().replace(':','-').replace('.','-')}.json")
    with open(json_path, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    # also append to events log if you use it
    try:
        from utils import append_jsonl
        append_jsonl(TRANS / "events.jsonl", rec)
    except Exception:
        pass

    # remove audio immediately
    try:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception:
        pass

    return json_path
