import os, json, yaml, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB   = DATA / "db"
RECS = DATA / "recordings"
TRANS= DATA / "transcripts"
def now_iso(): return dt.datetime.now().isoformat(timespec="seconds")
def load_config(): return yaml.safe_load(open(DATA/"config.yaml"))
def safe_mkdir(p): os.makedirs(p, exist_ok=True)
def read_jsonl(path):
    if not os.path.exists(path): return []
    return [json.loads(x) for x in open(path) if x.strip()]
def append_jsonl(path, obj):
    with open(path, "a") as f: f.write(json.dumps(obj, ensure_ascii=False) + "\n")
