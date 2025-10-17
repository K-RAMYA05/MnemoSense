# src/utils.py
import os, json, yaml, datetime as dt, tempfile
from pathlib import Path

# Roots / data dirs (unchanged)
ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
DB    = DATA / "db"
RECS  = DATA / "recordings"
TRANS = DATA / "transcripts"

# NEW: temp directory for ephemeral audio (auto-created)
TEMP = Path(tempfile.gettempdir()) / "mnemosense_tmp"
os.makedirs(TEMP, exist_ok=True)

def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def load_config():
    # Keep original behavior; assume file exists.
    # If you want fallback, wrap in try/except and return {} on failure.
    return yaml.safe_load(open(DATA / "config.yaml"))

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    if not os.path.exists(path): return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]

def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# NEW: helper for unique temp wav paths (caller should delete after use)
def get_temp_wav_path(prefix: str = "livewindow") -> str:
    """
    Returns a unique .wav path inside TEMP, e.g.
    /tmp/mnemosense_tmp/livewindow_2025-10-16T23-59-12.wav
    """
    ts = now_iso().replace(":", "-")
    return str(TEMP / f"{prefix}_{ts}.wav")
