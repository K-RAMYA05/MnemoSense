import whisper, os, json
from utils import TRANS, now_iso, safe_mkdir
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ["POSTHOG_DISABLED"]="1"; os.environ["CHROMA_TELEMETRY_DISABLED"]="1"
def transcribe(wav_path: str, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(wav_path, fp16=False)
    text = result.get("text","").strip()
    segments = result.get("segments", [])
    return text, segments
def save_transcript(wav_path: str, text: str, segments):
    safe_mkdir(TRANS)
    rec={"ts": now_iso(), "wav": wav_path, "text": text, "segments": [], "tags": []}
    json_path = wav_path.replace(".wav",".json").replace("/recordings/","/transcripts/")
    with open(json_path,"w") as f: json.dump(rec, f, ensure_ascii=False, indent=2)
    from utils import append_jsonl; append_jsonl(TRANS/"events.jsonl", rec)
    return json_path
