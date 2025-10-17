# src/continuous_ingest.py
import os, time, traceback

from .utils import load_config
from .audio_capture import ContinuousAudioRecorder
from .stt import transcribe
from .summarize import mmr_summarize
from .indexer import index_transcript

def log(msg): print(msg, flush=True)

def warmup_camera():
    """Ask for macOS camera permission on main thread (headless)."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        if opened:
            cap.read()
            cap.release()
        log(f"[warmup] camera opened={opened}")
    except Exception as e:
        log(f"[warmup] camera warmup skipped: {e}")

def ingest_loop(rec: ContinuousAudioRecorder, cfg: dict):
    model_size     = cfg.get("stt", {}).get("model_size", "base")
    max_chunk_len  = cfg.get("policies", {}).get("max_chunk_len", 512)
    max_sentences  = cfg.get("summary", {}).get("max_sentences", 4)
    store_full_text= cfg.get("summary", {}).get("store_full_text", False)

    log(f"[ingest] whisper={model_size} • store_full_text={store_full_text}")

    while True:
        try:
            wav = rec.get_chunk(timeout=1.0)
            if not wav:
                continue

            log(f"[ingest] processing {os.path.basename(wav)} …")
            text, _ = transcribe(wav, model_size=model_size)

            summary = mmr_summarize(text, max_sentences=max_sentences) if text.strip() else ""
            to_store = text if store_full_text else summary

            try: os.unlink(wav)
            except: pass

            if to_store.strip():
                index_transcript(to_store, source="", tags=[], max_chunk_len=max_chunk_len)
                log("[ingest] indexed ✓")
            else:
                log("[ingest] empty text; skipped")

        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            time.sleep(0.3)

def main():
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    warmup_camera()

    cfg = load_config()
    win = int(cfg.get("audio", {}).get("window_seconds", 120))
    rec = ContinuousAudioRecorder(window_sec=win, keep_webcam=True)
    rec.start()
    try:
        ingest_loop(rec, cfg)
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()
        print("[ingest] stopped")

if __name__ == "__main__":
    main()
